clear; clc; close all;

%% =========================================================
%  uwb_ddpg_train.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 1: DDPG Adaptive Kalman Filter Training
%
%  Training path and episode length now MATCH deploy exactly:
%    - 8-lane boustrophedon (same as uwb_ddpg_deploy.m)
%    - 60s episodes (same as uwb_ddpg_deploy.m)
%    - rng seed randomised per episode (generalisation)
%
%  State  (8 inputs):
%    [r(k) r(k-1) r(k-2)]  last 3 raw ranges, normalised
%    [e(k) e(k-1) e(k-2)]  last 3 Kalman errors (cm), normalised
%    [vel]                  range-rate estimate, normalised
%    [nlos]                 NLOS flag (0 or 1)
%
%  Action (3 outputs): Q_pos | Q_vel | R
%  Reward: negative local RMSE + tracking penalty + turn bonus
%
%  Improvements over failed v1:
%    1. 8-lane path matches deploy (was 4-lane)
%    2. 60s episodes match deploy (was 30s)
%    3. R_max tightened 0.4 -> 0.15 (prevents ignoring measurements)
%    4. Tracking penalty (penalises sluggish LOS tracking)
%    5. PER + Checkpointing + Reward Shaping (from Task 1)
%
%  Saves: ddpg_trained.mat
%  NIT Patna | Shrinivas V (2350011) | Dr. Golak Bihari Mahanta
%% =========================================================

%% PARALLEL POOL
nCores = feature('numcores');
cl     = parcluster('local');
maxW   = cl.NumWorkers;
nUse   = min(maxW, max(1, floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n', nCores, nUse);
if isempty(gcp('nocreate'))
    parpool('local', nUse);
end

%% SYSTEM PARAMETERS
fs  = 10; dt = 1/fs;
W   = 20; H  = 15;
F_k = [1 dt; 0 1];
H_k = [1 0];

% DDPG action bounds
% R_max reduced from 0.4 to 0.15
% This prevents the agent from learning to ignore measurements entirely
Q_pos_min = 0.0005;  Q_pos_max = 0.05;
Q_vel_min = 0.005;   Q_vel_max = 0.5;
R_min     = 0.01;    R_max     = 0.15;

nS = 8; nA = 3;

%% REPLAY BUFFER — PER
buf_size  = 20000;
buf_s     = zeros(buf_size, nS);
buf_a     = zeros(buf_size, nA);
buf_r     = zeros(buf_size, 1);
buf_s2    = zeros(buf_size, nS);
buf_p     = ones(buf_size,  1);
buf_ptr   = 1; buf_count = 0;

alpha   = 0.6;
beta0   = 0.4;
per_eps = 1e-4;

%% NETWORKS
rng(42);
actor.W1  = randn(32, nS)*sqrt(2/nS);     actor.b1  = zeros(32,1);
actor.W2  = randn(32, 32)*sqrt(2/32);     actor.b2  = zeros(32,1);
actor.W3  = randn(nA, 32)*0.01;           actor.b3  = zeros(nA,1);
actor_t   = actor;
critic.W1 = randn(32, nS+nA)*sqrt(2/(nS+nA)); critic.b1 = zeros(32,1);
critic.W2 = randn(32, 32)*sqrt(2/32);          critic.b2 = zeros(32,1);
critic.W3 = randn(1,  32)*0.01;                critic.b3 = zeros(1,1);
critic_t  = critic;

%% HELPER FUNCTIONS
function a = act_fwd(n, s)
    a = tanh(n.W3 * max(0, n.W2 * max(0, n.W1*s + n.b1) + n.b2) + n.b3);
end
function q = crit_fwd(n, s, a)
    q = n.W3 * max(0, n.W2 * max(0, n.W1*[s;a] + n.b1) + n.b2) + n.b3;
end
function [qp, qv, rv] = decode_action(a, qpn,qpx, qvn,qvx, rn,rx)
    qp = max(qpn, min(qpx, qpn + (a(1)+1)/2 * (qpx-qpn)));
    qv = max(qvn, min(qvx, qvn + (a(2)+1)/2 * (qvx-qvn)));
    rv = max(rn,  min(rx,  rn  + (a(3)+1)/2 * (rx -rn )));
end
function s = build_state(r3, e3, vel, nlos, W, H)
    s = [r3/sqrt(W^2+H^2); e3/100; vel/10; double(nlos)];
end
function n = clip_net(n, cv)
    flds = {'W1','b1','W2','b2','W3','b3'};
    for i = 1:6, n.(flds{i}) = max(-cv, min(cv, n.(flds{i}))); end
end
function nt = soft_update(n, nt, tau)
    nt.W1=tau*n.W1+(1-tau)*nt.W1; nt.b1=tau*n.b1+(1-tau)*nt.b1;
    nt.W2=tau*n.W2+(1-tau)*nt.W2; nt.b2=tau*n.b2+(1-tau)*nt.b2;
    nt.W3=tau*n.W3+(1-tau)*nt.W3; nt.b3=tau*n.b3+(1-tau)*nt.b3;
end

%% PER SAMPLING
function [idx, is_w] = per_sample(priorities, batch, beta)
    p    = priorities .^ beta;
    p    = p / sum(p);
    idx  = randsample(length(p), batch, true, p);
    w    = (1 ./ (length(p) .* p(idx))) .^ beta;
    is_w = (w / max(w))';
end

%% CRITIC UPDATE
function [n, td_errs] = update_critic(n, sb, ab, yb, lr, is_w)
    bs=size(sb,2); cv=1.0; td_errs=zeros(1,bs);
    gW1=zeros(size(n.W1)); gb1=zeros(size(n.b1));
    gW2=zeros(size(n.W2)); gb2=zeros(size(n.b2));
    gW3=zeros(size(n.W3)); gb3=zeros(size(n.b3));
    for i=1:bs
        x=[sb(:,i);ab(:,i)];
        z1=max(0,n.W1*x+n.b1); z2=max(0,n.W2*z1+n.b2);
        td=n.W3*z2+n.b3-yb(i); td_errs(i)=td; w=is_w(i);
        gW3=gW3+w*td*z2'; gb3=gb3+w*td;
        d2=(n.W3'*(w*td)).*(z2>0);
        gW2=gW2+d2*z1'; gb2=gb2+d2;
        d1=(n.W2'*d2).*(z1>0);
        gW1=gW1+d1*x'; gb1=gb1+d1;
    end
    sc=1/bs;
    n.W3=n.W3-lr*max(-cv,min(cv,gW3*sc)); n.b3=n.b3-lr*max(-cv,min(cv,gb3*sc));
    n.W2=n.W2-lr*max(-cv,min(cv,gW2*sc)); n.b2=n.b2-lr*max(-cv,min(cv,gb2*sc));
    n.W1=n.W1-lr*max(-cv,min(cv,gW1*sc)); n.b1=n.b1-lr*max(-cv,min(cv,gb1*sc));
end

%% ACTOR UPDATE
function n = update_actor(n, crit, sb, lr)
    bs=size(sb,2); eps_fd=1e-3; cv=0.5;
    gW1=zeros(size(n.W1)); gb1=zeros(size(n.b1));
    gW2=zeros(size(n.W2)); gb2=zeros(size(n.b2));
    gW3=zeros(size(n.W3)); gb3=zeros(size(n.b3));
    for i=1:bs
        s=sb(:,i);
        z1=max(0,n.W1*s+n.b1); z2=max(0,n.W2*z1+n.b2);
        a=tanh(n.W3*z2+n.b3);
        dqda=zeros(size(a));
        for j=1:length(a)
            ap=a; ap(j)=min(1,a(j)+eps_fd);
            am=a; am(j)=max(-1,a(j)-eps_fd);
            dqda(j)=(crit_fwd(crit,s,ap)-crit_fwd(crit,s,am))/(2*eps_fd);
        end
        d3=dqda.*(1-a.^2);
        gW3=gW3+d3*z2'; gb3=gb3+d3;
        d2=(n.W3'*d3).*(z2>0);
        gW2=gW2+d2*z1'; gb2=gb2+d2;
        d1=(n.W2'*d2).*(z1>0);
        gW1=gW1+d1*s'; gb1=gb1+d1;
    end
    sc=1/bs;
    n.W3=n.W3+lr*max(-cv,min(cv,gW3*sc)); n.b3=n.b3+lr*max(-cv,min(cv,gb3*sc));
    n.W2=n.W2+lr*max(-cv,min(cv,gW2*sc)); n.b2=n.b2+lr*max(-cv,min(cv,gb2*sc));
    n.W1=n.W1+lr*max(-cv,min(cv,gW1*sc)); n.b1=n.b1+lr*max(-cv,min(cv,gb1*sc));
end

%% PATH GENERATOR — 8 lanes, matches deploy exactly
function [tx, ty, tr] = make_path(W, H, N)
    lanes = 8;
    ly    = linspace(1, H-1, lanes);
    px=[]; py=[];
    for L=1:lanes
        if mod(L,2)==1, px=[px 1 W-1]; else, px=[px W-1 1]; end
        py=[py ly(L) ly(L)];
    end
    dc=[0 cumsum(sqrt(diff(px).^2+diff(py).^2))];
    dq=linspace(0,dc(end),N);
    tx=interp1(dc,px,dq); ty=interp1(dc,py,dq);
    tr=sqrt(tx.^2+ty.^2);
end

%% LANE TURN DETECTOR
function is_turn = detect_turn(tr, k, fs)
    is_turn=false;
    if k<3||k>length(tr)-1, return; end
    v_now =(tr(k)  -tr(k-1))*fs;
    v_prev=(tr(k-1)-tr(k-2))*fs;
    if abs(v_now-v_prev)>1.5, is_turn=true; end
end

%% TRAINING HYPERPARAMETERS
n_ep  = 500;
T_ep  = 60; N_ep = T_ep*fs;   % 600 steps, matches deploy
batch = 64;
lr_a  = 5e-5; lr_c = 3e-4;
gamma = 0.97; tau  = 0.005;
ou_sig = 0.25; ou_th = 0.15;

turn_bonus   = 2.0;
turn_qp_thr  = 0.002;
turn_window  = 10;
track_pen_w  = 0.5;
track_lag_thr = 20;

best_rmse  = inf; best_actor = actor; best_ep = 0; ckpt_every = 25;
rew_hist   = zeros(1,n_ep); rmse_hist = zeros(1,n_ep);

%% TRAINING FIGURE
fig = figure('Color','w','Position',[50 350 1150 370],...
    'Name','DDPG Training | UWB Mine Monitoring | NIT Patna','NumberTitle','off');
tl  = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

ax_rew=nexttile(tl); hold on;
h_ep =animatedline(ax_rew,'Color',[0.15 0.45 0.80],'LineWidth',0.8);
h_avg=animatedline(ax_rew,'Color',[0.90 0.20 0.10],'LineWidth',1.4);
set(ax_rew,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_rew,'Episode'); ylabel(ax_rew,'Reward');
xlim(ax_rew,[0 n_ep]); ylim(ax_rew,[-1500 0]); grid(ax_rew,'on');
legend(ax_rew,'Episode','Avg-20','Location','best','FontSize',7,'Box','off');
title(ax_rew,'Training Reward','FontWeight','bold','FontSize',9);

ax_rmse=nexttile(tl); hold on;
h_rmse=animatedline(ax_rmse,'Color',[0.05 0.72 0.32],'LineWidth',1.2);
set(ax_rmse,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_rmse,'Episode'); ylabel(ax_rmse,'RMSE (cm)');
xlim(ax_rmse,[0 n_ep]); ylim(ax_rmse,[0 45]); grid(ax_rmse,'on');
yline(ax_rmse,39.1,'--','Color',[0.50 0.10 0.80],'LineWidth',1.0,...
    'Label','Fixed KF baseline','LabelHorizontalAlignment','left');
title(ax_rmse,'Episode RMSE','FontWeight','bold','FontSize',9);

ax_qstd=nexttile(tl); hold on;
h_qstd=animatedline(ax_qstd,'Color',[0.92 0.55 0.05],'LineWidth',1.0);
set(ax_qstd,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_qstd,'Episode'); ylabel(ax_qstd,'Q_{pos} std');
xlim(ax_qstd,[0 n_ep]); ylim(ax_qstd,[0 0.01]); grid(ax_qstd,'on');
title(ax_qstd,'Q_{pos} Std Dev','FontWeight','bold','FontSize',9);

sgtitle(tl,'DDPG Training  |  8-Lane 60s Episodes  |  NIT Patna',...
    'FontSize',10,'FontWeight','bold');

%% TRAINING LOOP
fprintf('\n%s\n',repmat('=',1,65));
fprintf('  DDPG Training | UWB Mine Monitoring | NIT Patna\n');
fprintf('  Episodes: %d  |  Steps/ep: %d  |  Batch: %d\n',n_ep,N_ep,batch);
fprintf('  Path: 8-lane boustrophedon | Duration: %ds\n',T_ep);
fprintf('  R_max tightened to %.2f (prevents degenerate policy)\n',R_max);
fprintf('%s\n\n',repmat('=',1,65));
fprintf('%-6s %-10s %-10s %-12s %-10s %-10s\n',...
    'Ep','Reward','RMSE(cm)','Qpos_std','Best_ep','ETA(min)');
fprintf('%s\n',repmat('-',1,62));

t_start=tic;

for ep=1:n_ep

    beta_now=min(1.0, beta0+(1.0-beta0)*(ep/n_ep));

    rng(ep*7+13);
    [tx,ty,tr]=make_path(W,H,N_ep);
    raw   =max(tr+0.25*randn(1,N_ep)+(rand(1,N_ep)<0.12).*(0.4+0.8*rand(1,N_ep)),0);
    nlos_m=(raw-tr)>0.25;

    x_k=[raw(1);0]; P_k=eye(2)*0.5;
    raw_hist=raw(1)*ones(3,1); err_hist=zeros(3,1);
    state=build_state(raw_hist,err_hist,0,0,W,H);
    ou_n=zeros(nA,1);
    ep_rew=0; ep_rmse=0; err_buf=zeros(1,10); qp_log=zeros(1,N_ep);
    turn_countdown=0; rmse_at_turn=inf;

    for k=1:N_ep

        a_raw=act_fwd(actor,state);
        a_raw(isnan(a_raw)|isinf(a_raw))=0;
        decay=max(0.05,1-ep/n_ep);
        ou_n=ou_n-ou_th*ou_n*dt+ou_sig*decay*randn(nA,1);
        a_n=max(-1,min(1,a_raw+ou_n));

        [qp,qv,rv]=decode_action(a_n,...
            Q_pos_min,Q_pos_max,Q_vel_min,Q_vel_max,R_min,R_max);
        qp_log(k)=qp;

        Q_k=diag([qp qv]); R_k=rv^2;
        x_k=F_k*x_k; P_k=F_k*P_k*F_k'+Q_k;
        Kg=P_k*H_k'/(H_k*P_k*H_k'+R_k);
        x_k=x_k+Kg*(raw(k)-H_k*x_k); P_k=(eye(2)-Kg*H_k)*P_k;
        kf_v=max(0,min(sqrt(W^2+H^2),x_k(1)));
        if isnan(kf_v)||isinf(kf_v)
            x_k=[raw(k);0]; P_k=eye(2)*0.5; kf_v=raw(k);
        end

        err_cm=abs(kf_v-tr(k))*100;
        err_buf=[err_buf(2:end) err_cm];
        rmse_l=sqrt(mean(err_buf.^2));
        rew=-rmse_l;
        if nlos_m(k)&&err_cm>15, rew=rew-3; end
        if ~nlos_m(k)&&err_cm>track_lag_thr
            rew=rew-track_pen_w*(err_cm-track_lag_thr)/10;
        end

        is_turn=detect_turn(tr,k,fs);
        if is_turn
            turn_countdown=turn_window; rmse_at_turn=rmse_l;
            if qp<turn_qp_thr, rew=rew-1.0; end
        end
        if turn_countdown>0
            turn_countdown=turn_countdown-1;
            if turn_countdown==0
                improvement=rmse_at_turn-rmse_l;
                qp_was_high=qp_log(max(1,k-turn_window))>turn_qp_thr;
                if improvement>0&&qp_was_high
                    rew=rew+min(turn_bonus,improvement*0.5);
                end
            end
        end

        vel_est=0; if k>1, vel_est=(raw(k)-raw(k-1))*fs; end
        err_now=(kf_v-tr(k))*100;
        raw_hist=[raw_hist(2:end);raw(k)];
        err_hist=[err_hist(2:end);err_now];
        ns=build_state(raw_hist,err_hist,vel_est,double(nlos_m(k)),W,H);

        max_p=max(buf_p(1:max(1,buf_count)));
        buf_s(buf_ptr,:)=state'; buf_a(buf_ptr,:)=a_n';
        buf_r(buf_ptr)=rew;      buf_s2(buf_ptr,:)=ns';
        buf_p(buf_ptr)=max_p;
        buf_ptr=mod(buf_ptr,buf_size)+1;
        buf_count=min(buf_count+1,buf_size);
        state=ns; ep_rew=ep_rew+rew; ep_rmse=ep_rmse+rmse_l;

        if buf_count>=batch
            [idx,is_w]=per_sample(buf_p(1:buf_count),batch,beta_now);
            sb=buf_s(idx,:)'; ab=buf_a(idx,:)';
            rb=buf_r(idx);    s2b=buf_s2(idx,:)';
            yb=zeros(1,batch);
            for i=1:batch
                an=act_fwd(actor_t,s2b(:,i));
                an(isnan(an)|isinf(an))=0;
                qn=crit_fwd(critic_t,s2b(:,i),an);
                if isnan(qn)||isinf(qn), qn=0; end
                yb(i)=rb(i)+gamma*qn;
            end
            yb=max(-600,min(0,yb));
            [critic,td_errs]=update_critic(critic,sb,ab,yb,lr_c,is_w);
            new_p=(abs(td_errs)+per_eps).^alpha;
            for i=1:batch, buf_p(idx(i))=new_p(i); end
            actor=update_actor(actor,critic,sb,lr_a);
            actor=clip_net(actor,10); critic=clip_net(critic,10);
            actor_t=soft_update(actor,actor_t,tau);
            critic_t=soft_update(critic,critic_t,tau);
        end

    end

    ep_rmse_mean=ep_rmse/N_ep;
    rew_hist(ep)=ep_rew; rmse_hist(ep)=ep_rmse_mean;
    if ep_rmse_mean<best_rmse
        best_rmse=ep_rmse_mean; best_actor=actor; best_ep=ep;
    end
    if mod(ep,ckpt_every)==0
        ckpt_actor=best_actor;
        save(sprintf('checkpoint_ep%04d.mat',ep),'ckpt_actor','best_rmse','best_ep');
    end

    qp_std=std(qp_log);
    addpoints(h_ep,ep,ep_rew); addpoints(h_rmse,ep,ep_rmse_mean);
    addpoints(h_qstd,ep,qp_std);
    if ep>=20, addpoints(h_avg,ep,mean(rew_hist(max(1,ep-19):ep))); end

    if mod(ep,10)==0
        elapsed=toc(t_start); eta=(elapsed/ep)*(n_ep-ep)/60;
        fprintf('%-6d %-10.1f %-10.2f %-12.5f %-10d %-10.1f\n',...
            ep,ep_rew,ep_rmse_mean,qp_std,best_ep,eta);
        drawnow;
    end

end

fprintf('%s\n',repmat('-',1,62));
fprintf('Training complete  |  %.1f min\n',toc(t_start)/60);
fprintf('Best episode : %d\n',best_ep);
fprintf('Best RMSE    : %.4f cm\n',best_rmse);
actor=best_actor;
fprintf('Actor restored from episode %d\n',best_ep);

save('ddpg_trained.mat','actor','actor_t',...
    'Q_pos_min','Q_pos_max','Q_vel_min','Q_vel_max','R_min','R_max',...
    'F_k','H_k','fs','dt','W','H','nS','nA',...
    'best_ep','best_rmse','rmse_hist','rew_hist');
fprintf('Saved: ddpg_trained.mat\n\n');