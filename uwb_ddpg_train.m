clear; clc; close all;

%% PARALLEL POOL
nCores=feature('numcores');
cl=parcluster('local'); maxW=cl.NumWorkers;
nUse=min(maxW,max(1,floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n',nCores,nUse);
if isempty(gcp('nocreate')), parpool('local',nUse); end

fs=10; dt=1/fs; W=20; H=15;
F_k=[1 dt;0 1]; H_k=[1 0];
Q_pos_min=0.0005; Q_pos_max=0.05;
Q_vel_min=0.005;  Q_vel_max=0.5;
R_min=0.05;       R_max=0.4;
nS=8; nA=3;

%% REPLAY BUFFER
buf_size=10000;
buf_s=zeros(buf_size,nS); buf_a=zeros(buf_size,nA);
buf_r=zeros(buf_size,1);  buf_s2=zeros(buf_size,nS);
buf_ptr=1; buf_count=0;

%% NETWORKS
rng(42);
actor.W1=randn(32,nS)*sqrt(2/nS);  actor.b1=zeros(32,1);
actor.W2=randn(32,32)*sqrt(2/32);  actor.b2=zeros(32,1);
actor.W3=randn(nA,32)*0.01;        actor.b3=zeros(nA,1);
actor_t=actor;
critic.W1=randn(32,nS+nA)*sqrt(2/(nS+nA)); critic.b1=zeros(32,1);
critic.W2=randn(32,32)*sqrt(2/32);          critic.b2=zeros(32,1);
critic.W3=randn(1,32)*0.01;                 critic.b3=zeros(1,1);
critic_t=critic;

%% FORWARD PASSES
function a=act_fwd(n,s)
    a=tanh(n.W3*max(0,n.W2*max(0,n.W1*s+n.b1)+n.b2)+n.b3);
end
function q=crit_fwd(n,s,a)
    q=n.W3*max(0,n.W2*max(0,n.W1*[s;a]+n.b1)+n.b2)+n.b3;
end
function [qp,qv,rv]=decode(a,qpn,qpx,qvn,qvx,rn,rx)
    qp=max(qpn,min(qpx, qpn+(a(1)+1)/2*(qpx-qpn)));
    qv=max(qvn,min(qvx, qvn+(a(2)+1)/2*(qvx-qvn)));
    rv=max(rn, min(rx,  rn +(a(3)+1)/2*(rx-rn)));
end
function s=norm_s(r3,e3,vel,nlos,W,H)
    s=[r3/sqrt(W^2+H^2); e3/100; vel/10; double(nlos)];
end
function n=clip_net(n,cv)
    flds={'W1','b1','W2','b2','W3','b3'};
    for i=1:6, n.(flds{i})=max(-cv,min(cv,n.(flds{i}))); end
end

%% CRITIC BACKPROP
function n=upd_critic(n,sb,ab,yb,lr)
    bs=size(sb,2); cv=1.0;
    gW1=zeros(size(n.W1)); gb1=zeros(size(n.b1));
    gW2=zeros(size(n.W2)); gb2=zeros(size(n.b2));
    gW3=zeros(size(n.W3)); gb3=zeros(size(n.b3));
    for i=1:bs
        x=[sb(:,i);ab(:,i)];
        z1=max(0,n.W1*x+n.b1); z2=max(0,n.W2*z1+n.b2);
        d=n.W3*z2+n.b3-yb(i);
        gW3=gW3+d*z2'; gb3=gb3+d;
        d2=(n.W3'*d).*(z2>0);
        gW2=gW2+d2*z1'; gb2=gb2+d2;
        d1=(n.W2'*d2).*(z1>0);
        gW1=gW1+d1*x'; gb1=gb1+d1;
    end
    sc=1/bs;
    n.W3=n.W3-lr*max(-cv,min(cv,gW3*sc)); n.b3=n.b3-lr*max(-cv,min(cv,gb3*sc));
    n.W2=n.W2-lr*max(-cv,min(cv,gW2*sc)); n.b2=n.b2-lr*max(-cv,min(cv,gb2*sc));
    n.W1=n.W1-lr*max(-cv,min(cv,gW1*sc)); n.b1=n.b1-lr*max(-cv,min(cv,gb1*sc));
end

%% ACTOR UPDATE (finite difference)
function n=upd_actor(n,crit,sb,lr)
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

function n=soft_upd(n,nt,tau)
    n.W1=tau*n.W1+(1-tau)*nt.W1; n.b1=tau*n.b1+(1-tau)*nt.b1;
    n.W2=tau*n.W2+(1-tau)*nt.W2; n.b2=tau*n.b2+(1-tau)*nt.b2;
    n.W3=tau*n.W3+(1-tau)*nt.W3; n.b3=tau*n.b3+(1-tau)*nt.b3;
end

function [tx,ty,tr]=make_path(W,H,N)
    lanes=4; ly=linspace(1,H-1,lanes); px=[]; py=[];
    for L=1:lanes
        if mod(L,2)==1,px=[px 1 W-1];else,px=[px W-1 1];end
        py=[py ly(L) ly(L)];
    end
    dc=[0 cumsum(sqrt(diff(px).^2+diff(py).^2))];
    dq=linspace(0,dc(end),N);
    tx=interp1(dc,px,dq); ty=interp1(dc,py,dq);
    tr=sqrt(tx.^2+ty.^2);
end

%% TRAINING PARAMS
n_ep=800; T_ep=30; N_ep=T_ep*fs;
batch=32; lr_a=1e-4; lr_c=5e-4;
gamma=0.97; tau=0.01;
ou_sig=0.35; ou_th=0.15;
rew_hist=zeros(1,n_ep);

%% TRAINING FIGURE
fig_t=figure('Color','w','Position',[50 400 860 360],...
    'Name','DDPG Training','NumberTitle','off');
tl=tiledlayout(fig_t,1,2,'TileSpacing','compact','Padding','compact');
ax_r=nexttile(tl); hold on;
hl1=animatedline(ax_r,'Color',[0.15 0.45 0.80],'LineWidth',0.8);
hl2=animatedline(ax_r,'Color',[0.90 0.20 0.10],'LineWidth',1.4);
set(ax_r,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_r,'Episode','FontSize',8); ylabel(ax_r,'Reward','FontSize',8);
xlim(ax_r,[0 n_ep]); ylim(ax_r,[-600 0]); grid(ax_r,'on');
legend(ax_r,'Episode','Avg(20)','Location','best','FontSize',7,'Box','off');
title(ax_r,'Training Reward','FontWeight','bold','FontSize',9);

ax_m=nexttile(tl); hold on;
hl3=animatedline(ax_m,'Color',[0.10 0.65 0.35],'LineWidth',1.2);
set(ax_m,'Color','w','GridAlpha',0.12,'Box','on','FontSize',8);
xlabel(ax_m,'Episode','FontSize',8); ylabel(ax_m,'RMSE (cm)','FontSize',8);
xlim(ax_m,[0 n_ep]); ylim(ax_m,[0 25]); grid(ax_m,'on');
title(ax_m,'Kalman RMSE per Episode','FontWeight','bold','FontSize',9);
sgtitle(tl,'DDPG Training Progress | NIT Patna','FontSize',10,'FontWeight','bold');

%% TRAINING LOOP
fprintf('\nTraining started  [%d episodes x %d steps]\n\n', n_ep, N_ep);
fprintf('%-6s %-10s %-10s %-12s\n','Ep','Reward','RMSE(cm)','ETA(min)');
fprintf('%s\n',repmat('-',1,42));
t_start=tic;

for ep=1:n_ep
    [tx,ty,tr]=make_path(W,H,N_ep);
    raw=max(tr+0.25*randn(1,N_ep)+(rand(1,N_ep)<0.12).*(0.4+0.8*rand(1,N_ep)),0);
    nlos_m=(raw-tr)>0.25;

    x_k=[raw(1);0]; P_k=eye(2)*0.5;
    raw_hist=raw(1)*ones(3,1); err_hist=zeros(3,1);
    state=norm_s(raw_hist,err_hist,0,0,W,H);
    ou_n=zeros(nA,1);
    ep_rew=0; ep_rmse=0; err_buf=zeros(1,10);

    for k=1:N_ep
        a_raw=act_fwd(actor,state);
        a_raw(isnan(a_raw)|isinf(a_raw))=0;
        decay=max(0.05,1-ep/n_ep);
        ou_n=ou_n-ou_th*ou_n*dt+ou_sig*decay*randn(nA,1);
        a_n=max(-1,min(1,a_raw+ou_n));

        [qp,qv,rv]=decode(a_n,Q_pos_min,Q_pos_max,Q_vel_min,Q_vel_max,R_min,R_max);
        Q_k=diag([qp qv]); R_k=rv^2;
        x_k=F_k*x_k; P_k=F_k*P_k*F_k'+Q_k;
        Kg=P_k*H_k'/(H_k*P_k*H_k'+R_k);
        x_k=x_k+Kg*(raw(k)-H_k*x_k); P_k=(eye(2)-Kg*H_k)*P_k;
        kf_v=max(0,min(sqrt(W^2+H^2),x_k(1)));
        if isnan(kf_v)|isinf(kf_v), x_k=[raw(k);0]; P_k=eye(2)*0.5; kf_v=raw(k); end

        err_cm=abs(kf_v-tr(k))*100;
        err_buf=[err_buf(2:end) err_cm];
        rmse_l=sqrt(mean(err_buf.^2));
        rew=-rmse_l;
        if nlos_m(k)&&err_cm>15, rew=rew-3; end
        ep_rew=ep_rew+rew; ep_rmse=ep_rmse+rmse_l;

        vel_est=0; if k>1, vel_est=(raw(k)-raw(k-1))*fs; end
        err_now=(kf_v-tr(k))*100;
        raw_hist=[raw_hist(2:end);raw(k)];
        err_hist=[err_hist(2:end);err_now];
        ns=norm_s(raw_hist,err_hist,vel_est,double(nlos_m(k)),W,H);

        buf_s(buf_ptr,:)=state'; buf_a(buf_ptr,:)=a_n';
        buf_r(buf_ptr)=rew; buf_s2(buf_ptr,:)=ns';
        buf_ptr=mod(buf_ptr,buf_size)+1;
        buf_count=min(buf_count+1,buf_size);
        state=ns;

        if buf_count>=batch
            idx=randperm(buf_count,batch);
            sb=buf_s(idx,:)'; ab=buf_a(idx,:)';
            rb=buf_r(idx); s2b=buf_s2(idx,:)';
            yb=zeros(1,batch);
            for i=1:batch
                an=act_fwd(actor_t,s2b(:,i));
                an(isnan(an)|isinf(an))=0;
                qn=crit_fwd(critic_t,s2b(:,i),an);
                if isnan(qn)|isinf(qn), qn=0; end
                yb(i)=rb(i)+gamma*qn;
            end
            yb=max(-300,min(0,yb));
            critic=upd_critic(critic,sb,ab,yb,lr_c);
            actor =upd_actor(actor,critic,sb,lr_a);
            actor =clip_net(actor,10);
            critic=clip_net(critic,10);
            actor_t =soft_upd(actor, actor_t, tau);
            critic_t=soft_upd(critic,critic_t,tau);
        end
    end

    rew_hist(ep)=ep_rew;
    addpoints(hl1,ep,ep_rew);
    addpoints(hl3,ep,ep_rmse/N_ep);
    if ep>=20, addpoints(hl2,ep,mean(rew_hist(max(1,ep-19):ep))); end

    if mod(ep,10)==0
        elapsed=toc(t_start);
        eta=(elapsed/ep)*(n_ep-ep)/60;
        fprintf('%-6d %-10.1f %-10.2f %-12.1f\n',...
            ep,ep_rew,ep_rmse/N_ep,eta);
        drawnow;
    end
end

fprintf('%s\n',repmat('-',1,42));
fprintf('Training done in %.1f min.\n',toc(t_start)/60);

save('ddpg_trained.mat','actor','actor_t',...
    'Q_pos_min','Q_pos_max','Q_vel_min','Q_vel_max','R_min','R_max',...
    'F_k','H_k','fs','dt','W','H','nS','nA');
fprintf('Saved: ddpg_trained.mat\n');