clear; clc; close all;

%% =========================================================
%  uwb_4anchor_train.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 2: 4-Anchor DDPG for 2D Localisation
%
%  Extension from Task 1:
%    Task 1 — DDPG tunes 1D ranging Kalman (1 anchor, scalar range)
%    Task 2 — DDPG tunes 2D position Kalman (4 anchors, XY position)
%
%  System:
%    4 UWB anchors at mine corners: A1(0,0) A2(20,0) A3(20,15) A4(0,15)
%    Each anchor independently measures range to drone at 10 Hz
%    Gauss-Newton trilateration fuses 4 ranges -> XY estimate
%    DDPG agent tunes 2D Kalman Q and R at every timestep
%    Reward = negative 2D position RMSE (cm) — the DST target metric
%
%  State (11 inputs):
%    [r1 r2 r3 r4]   raw ranges from 4 anchors, normalised by D_max
%    [e1 e2 e3 e4]   per-anchor range errors (cm), normalised by 100
%    [vx vy]         estimated 2D velocity, normalised by 5
%    [nlos]          mean NLOS flag across anchors (0 to 1)
%
%  Action (3 outputs):
%    Q_pos : position process noise  (same for X and Y)
%    Q_vel : velocity process noise  (same for Vx and Vy)
%    R_2d  : measurement noise       (same for X and Y)
%
%  Saves: ddpg_4anchor_trained.mat
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
anchors = [0 0; W 0; W H; 0 H];   % 4 corner anchors
nAnc    = 4;

% 2D Kalman matrices — constant velocity model
% State: [x; y; vx; vy]
Fk = [1 0 dt 0;
      0 1 0  dt;
      0 0 1  0;
      0 0 0  1];
Hk = [1 0 0 0;
      0 1 0 0];

% DDPG action bounds
Q_pos_min = 0.0001;  Q_pos_max = 0.05;
Q_vel_min = 0.001;   Q_vel_max = 0.5;
R_min     = 0.005;   R_max     = 0.20;

nS = 11; nA = 3;

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
% Wider hidden layers (64) for richer 11-input state
rng(42);
actor.W1  = randn(64, nS)*sqrt(2/nS);     actor.b1  = zeros(64,1);
actor.W2  = randn(64, 64)*sqrt(2/64);     actor.b2  = zeros(64,1);
actor.W3  = randn(nA, 64)*0.01;           actor.b3  = zeros(nA,1);
actor_t   = actor;
critic.W1 = randn(64, nS+nA)*sqrt(2/(nS+nA)); critic.b1 = zeros(64,1);
critic.W2 = randn(64, 64)*sqrt(2/64);          critic.b2 = zeros(64,1);
critic.W3 = randn(1,  64)*0.01;                critic.b3 = zeros(1,1);
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
function n = clip_net(n, cv)
    flds = {'W1','b1','W2','b2','W3','b3'};
    for i = 1:6, n.(flds{i}) = max(-cv, min(cv, n.(flds{i}))); end
end
function nt = soft_update(n, nt, tau)
    nt.W1=tau*n.W1+(1-tau)*nt.W1; nt.b1=tau*n.b1+(1-tau)*nt.b1;
    nt.W2=tau*n.W2+(1-tau)*nt.W2; nt.b2=tau*n.b2+(1-tau)*nt.b2;
    nt.W3=tau*n.W3+(1-tau)*nt.W3; nt.b3=tau*n.b3+(1-tau)*nt.b3;
end

%% =========================================================
%  STATE BUILDER — 11 inputs
%  r4   : (4x1) raw ranges from all 4 anchors
%  e4   : (4x1) per-anchor errors from last step (cm)
%  vel  : (2x1) [vx; vy] estimated velocity
%  nlos : scalar mean NLOS flag
%% =========================================================
function s = build_state(r4, e4, vel2, nlos_mean, W, H)
    D = sqrt(W^2 + H^2);
    s = [r4/D; e4/100; vel2/5; double(nlos_mean)];
end

%% =========================================================
%  GAUSS-NEWTON TRILATERATION
%  Solves overdetermined nonlinear LS:
%  Find XY position from 4 anchor ranges
%  15 iterations max, stops when update < 1e-6 m
%% =========================================================
function pos = gauss_newton(ranges, anch, p0)
    p = p0;
    for i = 1:15
        d  = sqrt(sum((anch - p).^2, 2));
        d  = max(d, 1e-6);
        J  = -(anch - p) ./ d;
        dp = (J'*J) \ (J'*(ranges - d));
        p  = p + dp';
        if norm(dp) < 1e-6, break; end
    end
    pos = p;
end

%% PER SAMPLING
function [idx, is_w] = per_sample(priorities, batch, beta)
    p    = priorities .^ beta;
    p    = p / sum(p);
    idx  = randsample(length(p), batch, true, p);
    w    = (1 ./ (length(p) .* p(idx))) .^ beta;
    is_w = (w / max(w))';
end

%% CRITIC UPDATE WITH IS WEIGHTS
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

%% PATH GENERATOR — 8-lane boustrophedon, returns XY and true ranges
function [tx, ty, true_r] = make_path(W, H, N, anchors)
    lanes=8; ly=linspace(1,H-1,lanes);
    px=[]; py=[];
    for L=1:lanes
        if mod(L,2)==1, px=[px 1 W-1]; else, px=[px W-1 1]; end
        py=[py ly(L) ly(L)];
    end
    dc=[0 cumsum(sqrt(diff(px).^2+diff(py).^2))];
    dq=linspace(0,dc(end),N);
    tx=interp1(dc,px,dq); ty=interp1(dc,py,dq);
    true_r=zeros(size(anchors,1),N);
    for a=1:size(anchors,1)
        true_r(a,:)=sqrt((tx-anchors(a,1)).^2+(ty-anchors(a,2)).^2);
    end
end

%% NOISE MODEL — 4 anchors, independent NLOS per anchor
function [raw_r, nlos_f] = add_noise(true_r, nAnc, N)
    raw_r  = zeros(nAnc, N);
    nlos_f = false(nAnc, N);
    for a=1:nAnc
        noise       = 0.20*randn(1,N) + ...
                      (rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
        raw_r(a,:)  = max(true_r(a,:)+noise, 0.01);
        nlos_f(a,:) = (raw_r(a,:)-true_r(a,:)) > 0.20;
    end
end

%% TRAINING HYPERPARAMETERS
n_ep  = 500;
T_ep  = 60; N_ep = T_ep*fs;
batch = 64;
lr_a  = 5e-5; lr_c = 3e-4;
gamma = 0.97; tau  = 0.005;
ou_sig = 0.20; ou_th = 0.15;

best_rmse  = inf; best_actor = actor;
best_ep    = 0;   ckpt_every = 25;
rew_hist   = zeros(1,n_ep);
rmse_hist  = zeros(1,n_ep);

%% TRAINING FIGURE
fig = figure('Color','w','Position',[50 350 1150 370],...
    'Name','4-Anchor DDPG Training | NIT Patna','NumberTitle','off');
tl  = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

ax_rew=nexttile(tl); hold on;
h_ep =animatedline(ax_rew,'Color',[0.15 0.45 0.80],'LineWidth',0.8);
h_avg=animatedline(ax_rew,'Color',[0.90 0.20 0.10],'LineWidth',1.4);
set(ax_rew,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_rew,'Episode'); ylabel(ax_rew,'Reward');
xlim(ax_rew,[0 n_ep]); ylim(ax_rew,[-500 0]); grid(ax_rew,'on');
legend(ax_rew,'Episode','Avg-20','Location','best','FontSize',7,'Box','off');
title(ax_rew,'Training Reward','FontWeight','bold','FontSize',9);

ax_rmse=nexttile(tl); hold on;
h_rmse=animatedline(ax_rmse,'Color',[0.05 0.72 0.32],'LineWidth',1.2);
set(ax_rmse,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_rmse,'Episode'); ylabel(ax_rmse,'2D RMSE (cm)');
xlim(ax_rmse,[0 n_ep]); ylim(ax_rmse,[0 40]); grid(ax_rmse,'on');
yline(ax_rmse,50,'--','Color',[0.85 0.20 0.20],'LineWidth',1.0,...
    'Label','DST Target 50 cm','LabelHorizontalAlignment','left');
title(ax_rmse,'2D Position RMSE','FontWeight','bold','FontSize',9);

ax_q=nexttile(tl); hold on;
h_qstd=animatedline(ax_q,'Color',[0.92 0.55 0.05],'LineWidth',1.0);
set(ax_q,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_q,'Episode'); ylabel(ax_q,'Q_{pos} std');
xlim(ax_q,[0 n_ep]); ylim(ax_q,[0 0.01]); grid(ax_q,'on');
title(ax_q,'Q_{pos} Std Dev','FontWeight','bold','FontSize',9);

sgtitle(tl,'4-Anchor DDPG Training  |  2D Localisation  |  NIT Patna',...
    'FontSize',10,'FontWeight','bold');

%% TRAINING LOOP
fprintf('\n%s\n',repmat('=',1,65));
fprintf('  4-Anchor DDPG Training | 2D Localisation | NIT Patna\n');
fprintf('  State: %d inputs | Action: %d outputs\n',nS,nA);
fprintf('  State: [r1 r2 r3 r4 | e1 e2 e3 e4 | vx vy | nlos]\n');
fprintf('  Action: [Q_pos | Q_vel | R_2D]\n');
fprintf('  Reward: -2D_position_RMSE (cm)\n');
fprintf('  Episodes: %d | Steps/ep: %d\n',n_ep,N_ep);
fprintf('%s\n\n',repmat('=',1,65));
fprintf('%-6s %-10s %-12s %-10s %-10s %-10s\n',...
    'Ep','Reward','2D_RMSE(cm)','Qpos_std','Best_ep','ETA(min)');
fprintf('%s\n',repmat('-',1,62));

t_start=tic;

for ep=1:n_ep

    beta_now=min(1.0, beta0+(1.0-beta0)*(ep/n_ep));

    % Randomise seed each episode
    rng(ep*11+7);
    [tx,ty,true_r]=make_path(W,H,N_ep,anchors);
    [raw_r,nlos_f]=add_noise(true_r,nAnc,N_ep);

    % Initial GN position estimate
    try
        p0=[W/2 H/2];
        p_init=gauss_newton(raw_r(:,1),anchors,p0);
        p_init(1)=max(min(p_init(1),W+1),-1);
        p_init(2)=max(min(p_init(2),H+1),-1);
    catch
        p_init=[W/2 H/2];
    end

    % 2D Kalman init
    xk=[p_init(1); p_init(2); 0; 0];
    Pk=eye(4);

    % Initial RL state
    r4_0     = raw_r(:,1);
    e4_0     = zeros(4,1);
    vel2_0   = [0;0];
    nlos_0   = mean(double(nlos_f(:,1)));
    state    = build_state(r4_0,e4_0,vel2_0,nlos_0,W,H);

    ou_n    = zeros(nA,1);
    ep_rew  = 0; ep_rmse=0;
    err_buf = zeros(1,10);
    qp_log  = zeros(1,N_ep);
    pos_prev= p_init;

    for k=1:N_ep

        %% ACTION
        a_raw=act_fwd(actor,state);
        a_raw(isnan(a_raw)|isinf(a_raw))=0;
        decay=max(0.05,1-ep/n_ep);
        ou_n=ou_n-ou_th*ou_n*dt+ou_sig*decay*randn(nA,1);
        a_n=max(-1,min(1,a_raw+ou_n));

        [qp,qv,rv]=decode_action(a_n,...
            Q_pos_min,Q_pos_max,Q_vel_min,Q_vel_max,R_min,R_max);
        qp_log(k)=qp;

        %% GAUSS-NEWTON TRILATERATION
        try
            pos_gn=gauss_newton(raw_r(:,k),anchors,pos_prev);
        catch
            pos_gn=pos_prev;
        end
        pos_gn(1)=max(min(pos_gn(1),W+2),-2);
        pos_gn(2)=max(min(pos_gn(2),H+2),-2);

        %% 2D KALMAN — DDPG tunes Q and R
        Q2d=diag([qp qp qv qv]);
        R2d=diag([rv rv]);
        xk=Fk*xk; Pk=Fk*Pk*Fk'+Q2d;
        Kk=Pk*Hk'/(Hk*Pk*Hk'+R2d);
        xk=xk+Kk*(pos_gn'-Hk*xk);
        Pk=(eye(4)-Kk*Hk)*Pk;

        pos_est=xk(1:2)';
        vel_est=xk(3:4);
        pos_est(1)=max(min(pos_est(1),W+2),-2);
        pos_est(2)=max(min(pos_est(2),H+2),-2);

        if any(isnan(pos_est))||any(isinf(pos_est))
            pos_est=pos_prev;
            xk=[pos_prev(1);pos_prev(2);0;0]; Pk=eye(4);
        end

        %% 2D POSITION REWARD
        err_2d  = sqrt((pos_est(1)-tx(k))^2+(pos_est(2)-ty(k))^2)*100;
        err_buf = [err_buf(2:end) err_2d];
        rmse_l  = sqrt(mean(err_buf.^2));
        rew     = -rmse_l;

        % Multi-anchor NLOS penalty
        n_nlos=sum(nlos_f(:,k));
        if n_nlos>=2 && err_2d>20
            rew=rew-2*n_nlos;
        end

        ep_rew=ep_rew+rew; ep_rmse=ep_rmse+rmse_l;

        %% NEXT STATE
        est_r=sqrt((pos_est(1)-anchors(:,1)).^2+...
                   (pos_est(2)-anchors(:,2)).^2);
        e4_now=(est_r-raw_r(:,k))*100;
        nlos_mean=mean(double(nlos_f(:,k)));
        ns=build_state(raw_r(:,k),e4_now,vel_est,nlos_mean,W,H);

        %% BUFFER
        max_p=max(buf_p(1:max(1,buf_count)));
        buf_s(buf_ptr,:)=state'; buf_a(buf_ptr,:)=a_n';
        buf_r(buf_ptr)=rew;      buf_s2(buf_ptr,:)=ns';
        buf_p(buf_ptr)=max_p;
        buf_ptr=mod(buf_ptr,buf_size)+1;
        buf_count=min(buf_count+1,buf_size);
        state=ns; pos_prev=pos_est;

        %% PER TRAINING
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
            yb=max(-500,min(0,yb));
            [critic,td_errs]=update_critic(critic,sb,ab,yb,lr_c,is_w);
            new_p=(abs(td_errs)+per_eps).^alpha;
            for i=1:batch, buf_p(idx(i))=new_p(i); end
            actor=update_actor(actor,critic,sb,lr_a);
            actor=clip_net(actor,10); critic=clip_net(critic,10);
            actor_t=soft_update(actor,actor_t,tau);
            critic_t=soft_update(critic,critic_t,tau);
        end

    end % timestep loop

    %% CHECKPOINTING
    ep_rmse_mean=ep_rmse/N_ep;
    rew_hist(ep)=ep_rew; rmse_hist(ep)=ep_rmse_mean;
    if ep_rmse_mean<best_rmse
        best_rmse=ep_rmse_mean; best_actor=actor; best_ep=ep;
    end
    if mod(ep,ckpt_every)==0
        ckpt_actor=best_actor;
        save(sprintf('checkpoint_4a_ep%04d.mat',ep),...
            'ckpt_actor','best_rmse','best_ep');
    end

    qp_std=std(qp_log);
    addpoints(h_ep,ep,ep_rew);
    addpoints(h_rmse,ep,ep_rmse_mean);
    addpoints(h_qstd,ep,qp_std);
    if ep>=20, addpoints(h_avg,ep,mean(rew_hist(max(1,ep-19):ep))); end

    if mod(ep,10)==0
        elapsed=toc(t_start); eta=(elapsed/ep)*(n_ep-ep)/60;
        fprintf('%-6d %-10.1f %-12.2f %-10.5f %-10d %-10.1f\n',...
            ep,ep_rew,ep_rmse_mean,qp_std,best_ep,eta);
        drawnow;
    end

end % episode loop

fprintf('%s\n',repmat('-',1,62));
fprintf('Training complete  |  %.1f min\n',toc(t_start)/60);
fprintf('Best episode : %d\n',best_ep);
fprintf('Best 2D RMSE : %.4f cm\n',best_rmse);
actor=best_actor;
fprintf('Actor restored from episode %d\n',best_ep);

save('ddpg_4anchor_trained.mat',...
    'actor','actor_t',...
    'Q_pos_min','Q_pos_max','Q_vel_min','Q_vel_max','R_min','R_max',...
    'Fk','Hk','fs','dt','W','H','nS','nA','anchors',...
    'best_ep','best_rmse','rmse_hist','rew_hist');
fprintf('Saved: ddpg_4anchor_trained.mat\n\n');