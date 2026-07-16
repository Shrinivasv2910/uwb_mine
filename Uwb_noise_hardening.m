clear; clc; close all;

%% =========================================================
%  uwb_noise_hardening.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 4: Noise Model Hardening
%
%  Scenario A — Clock Drift (20 ppm per anchor, sinusoidal bias)
%  Scenario B — Anchor Placement Error (±0.5m)
%  Scenario C — Single Anchor Dropout (A1 drops t=20–30s)
%
%  Scenario C fix vs previous version:
%    OLD: Feed stale frozen A1 reading to DDPG during dropout
%         -> agent state completely out-of-distribution -> diverges
%    NEW: Substitute missing range with estimated range from
%         GN position (physically plausible), flag NLOS=1 for A1
%         -> agent sees coherent state -> graceful degradation
%
%  NIT Patna | Shrinivas V (2350011) | Dr. Golak Bihari Mahanta
%% =========================================================

%% PARALLEL POOL
nCores=feature('numcores'); cl=parcluster('local');
maxW=cl.NumWorkers; nUse=min(maxW,max(1,floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n',nCores,nUse);
if isempty(gcp('nocreate')), parpool('local',nUse); end

%% LOAD TASK 2 AGENT
if ~exist('ddpg_4anchor_trained.mat','file')
    error('ddpg_4anchor_trained.mat not found.');
end
d2=load('ddpg_4anchor_trained.mat');
fprintf('Task 2 agent loaded  (best ep: %d | train RMSE: %.2f cm)\n\n',...
    d2.best_ep,d2.best_rmse);

%% SYSTEM PARAMETERS
fs=d2.fs; dt=d2.dt; W=d2.W; H=d2.H;
anchors_nom=d2.anchors; nAnc=4;
T=60; t_vec=0:1/fs:T; N=length(t_vec);

%% HELPER FUNCTIONS
function a=act_fwd(n,s)
    a=tanh(n.W3*max(0,n.W2*max(0,n.W1*s+n.b1)+n.b2)+n.b3);
end
function [qp,qv,rv]=decode_action(a,qpn,qpx,qvn,qvx,rn,rx)
    qp=max(qpn,min(qpx,qpn+(a(1)+1)/2*(qpx-qpn)));
    qv=max(qvn,min(qvx,qvn+(a(2)+1)/2*(qvx-qvn)));
    rv=max(rn, min(rx, rn +(a(3)+1)/2*(rx -rn )));
end
function s=build_state_4a(r4,e4,vel2,nlos_m,W,H)
    D=sqrt(W^2+H^2);
    s=[r4/D; e4/100; vel2/5; double(nlos_m)];
end
function pos=gauss_newton(ranges,anch,p0)
    p=p0;
    for i=1:15
        d=sqrt(sum((anch-p).^2,2)); d=max(d,1e-6);
        J=-(anch-p)./d;
        dp=(J'*J)\(J'*(ranges-d)); p=p+dp';
        if norm(dp)<1e-6, break; end
    end
    pos=p;
end
function s=pass_fail(c), if c, s='PASSED'; else, s='FAILED'; end, end

%% BOUSTROPHEDON PATH
lanes=8; ly=linspace(1,H-1,lanes); px=[]; py=[];
for L=1:lanes
    if mod(L,2)==1, px=[px 1 W-1]; else, px=[px W-1 1]; end
    py=[py ly(L) ly(L)];
end
dc=[0 cumsum(sqrt(diff(px).^2+diff(py).^2))];
dq=linspace(0,dc(end),N);
true_x=interp1(dc,px,dq,'linear');
true_y=interp1(dc,py,dq,'linear');

%% KALMAN MATRICES
Fk_2d=[1 0 dt 0;0 1 0 dt;0 0 1 0;0 0 0 1];
Hk_2d=[1 0 0 0;0 1 0 0];
Qk_fix=diag([0.001 0.001 0.01 0.01]);
Rk_fix=diag([0.12^2 0.12^2]);

%% =========================================================
%  STANDARD RUN FUNCTION — all 4 anchors available
%  Used for Baseline, Scenario A, Scenario B
%% =========================================================
function [kf2d_fix,kf2d_rl]=run_standard(raw_r,nlos_f,anch_used,...
        Fk_2d,Hk_2d,Qk_fix,Rk_fix,d2,W,H,N)

    %% Fixed 2D KF
    p0=[W/2 H/2]; est_raw=zeros(2,N);
    for k=1:N
        try, est_raw(:,k)=gauss_newton(raw_r(:,k),anch_used,p0)';
        catch, est_raw(:,k)=p0'; end
        p0=est_raw(:,k)';
    end
    est_raw(1,:)=max(min(est_raw(1,:),W+2),-2);
    est_raw(2,:)=max(min(est_raw(2,:),H+2),-2);
    xk=[est_raw(1,1);est_raw(2,1);0;0]; Pk=eye(4);
    kf2d_fix=zeros(2,N);
    for k=1:N
        xk=Fk_2d*xk; Pk=Fk_2d*Pk*Fk_2d'+Qk_fix;
        Kk=Pk*Hk_2d'/(Hk_2d*Pk*Hk_2d'+Rk_fix);
        xk=xk+Kk*(est_raw(:,k)-Hk_2d*xk);
        Pk=(eye(4)-Kk*Hk_2d)*Pk;
        kf2d_fix(:,k)=xk(1:2);
    end

    %% DDPG 4-Anchor KF
    try
        p_init=gauss_newton(raw_r(:,1),anch_used,[W/2 H/2]);
        p_init(1)=max(min(p_init(1),W+1),-1);
        p_init(2)=max(min(p_init(2),H+1),-1);
    catch
        p_init=[W/2 H/2];
    end
    xk_rl=[p_init(1);p_init(2);0;0]; Pk_rl=eye(4);
    kf2d_rl=zeros(2,N);
    st=build_state_4a(raw_r(:,1),zeros(4,1),[0;0],...
        mean(double(nlos_f(:,1))),W,H);
    pp=p_init;
    for k=1:N
        ao=act_fwd(d2.actor,st); ao(isnan(ao)|isinf(ao))=0;
        [qp,qv,rv]=decode_action(ao,...
            d2.Q_pos_min,d2.Q_pos_max,...
            d2.Q_vel_min,d2.Q_vel_max,...
            d2.R_min,d2.R_max);
        try, pg=gauss_newton(raw_r(:,k),anch_used,pp);
        catch, pg=pp; end
        pg(1)=max(min(pg(1),W+2),-2); pg(2)=max(min(pg(2),H+2),-2);
        Q2d=diag([qp qp qv qv]); R2d=diag([rv rv]);
        xk_rl=d2.Fk*xk_rl; Pk_rl=d2.Fk*Pk_rl*d2.Fk'+Q2d;
        Kk=Pk_rl*d2.Hk'/(d2.Hk*Pk_rl*d2.Hk'+R2d);
        xk_rl=xk_rl+Kk*(pg'-d2.Hk*xk_rl);
        Pk_rl=(eye(4)-Kk*d2.Hk)*Pk_rl;
        pe=xk_rl(1:2)';
        pe(1)=max(min(pe(1),W+2),-2); pe(2)=max(min(pe(2),H+2),-2);
        if any(isnan(pe))||any(isinf(pe))
            pe=pp; xk_rl=[pp(1);pp(2);0;0]; Pk_rl=eye(4);
        end
        kf2d_rl(:,k)=pe'; pp=pe;
        er=sqrt((pe(1)-d2.anchors(:,1)).^2+(pe(2)-d2.anchors(:,2)).^2);
        e4=(er-raw_r(:,k))*100; v2=xk_rl(3:4);
        st=build_state_4a(raw_r(:,k),e4,v2,...
            mean(double(nlos_f(:,k))),W,H);
    end
end

%% =========================================================
%  DROPOUT RUN FUNCTION — Scenario C
%
%  Fixed KF: GN uses only the 3 valid anchors during dropout
%
%  DDPG: Substitute missing anchor with estimated range from
%        current GN position estimate, flag that anchor NLOS=1
%        This keeps the DDPG state IN-DISTRIBUTION:
%        - The agent still receives 4 range values (no missing input)
%        - The substituted range is geometrically consistent
%        - NLOS=1 signals the agent to trust measurements less
%        - Agent can adapt Q/R gracefully rather than diverging
%% =========================================================
function [kf2d_fix,kf2d_rl]=run_dropout(raw_r,nlos_f,...
        dropout_anchor,k_start,k_end,anch_used,...
        Fk_2d,Hk_2d,Qk_fix,Rk_fix,d2,W,H,N)

    nAnc=size(anch_used,1);

    %% Fixed KF — GN on valid anchors only during dropout
    p0=[W/2 H/2]; est_gn=zeros(2,N);
    for k=1:N
        if k>=k_start && k<=k_end
            valid_idx=setdiff(1:nAnc, dropout_anchor);
        else
            valid_idx=1:nAnc;
        end
        try
            est_gn(:,k)=gauss_newton(...
                raw_r(valid_idx,k), anch_used(valid_idx,:), p0)';
        catch
            est_gn(:,k)=p0';
        end
        est_gn(1,k)=max(min(est_gn(1,k),W+2),-2);
        est_gn(2,k)=max(min(est_gn(2,k),H+2),-2);
        p0=est_gn(:,k)';
    end
    xk=[est_gn(1,1);est_gn(2,1);0;0]; Pk=eye(4);
    kf2d_fix=zeros(2,N);
    for k=1:N
        xk=Fk_2d*xk; Pk=Fk_2d*Pk*Fk_2d'+Qk_fix;
        Kk=Pk*Hk_2d'/(Hk_2d*Pk*Hk_2d'+Rk_fix);
        xk=xk+Kk*(est_gn(:,k)-Hk_2d*xk);
        Pk=(eye(4)-Kk*Hk_2d)*Pk;
        kf2d_fix(:,k)=xk(1:2);
    end

    %% DDPG — estimated range substitution during dropout
    try
        p_init=gauss_newton(raw_r(:,1),anch_used,[W/2 H/2]);
        p_init(1)=max(min(p_init(1),W+1),-1);
        p_init(2)=max(min(p_init(2),H+1),-1);
    catch
        p_init=[W/2 H/2];
    end
    xk_rl=[p_init(1);p_init(2);0;0]; Pk_rl=eye(4);
    kf2d_rl=zeros(2,N);
    st=build_state_4a(raw_r(:,1),zeros(4,1),[0;0],...
        mean(double(nlos_f(:,1))),W,H);
    pp=p_init;

    for k=1:N
        % Build measurement vector for DDPG
        r4_k   = raw_r(:,k);
        nlos_k = nlos_f(:,k);

        if k>=k_start && k<=k_end
            % Substitute dropped anchor with estimated range
            % from current position estimate (pp = last known position)
            est_range_drop = sqrt((pp(1)-anch_used(dropout_anchor,1))^2 + ...
                                  (pp(2)-anch_used(dropout_anchor,2))^2);
            r4_k(dropout_anchor)   = est_range_drop;
            nlos_k(dropout_anchor) = true;   % flag as NLOS

            % GN on 3 valid anchors only for position measurement
            valid_idx=setdiff(1:nAnc,dropout_anchor);
            try
                pg=gauss_newton(r4_k(valid_idx),anch_used(valid_idx,:),pp);
            catch
                pg=pp;
            end
        else
            % Normal: all 4 anchors
            try, pg=gauss_newton(r4_k,anch_used,pp);
            catch, pg=pp; end
        end

        pg(1)=max(min(pg(1),W+2),-2); pg(2)=max(min(pg(2),H+2),-2);

        ao=act_fwd(d2.actor,st); ao(isnan(ao)|isinf(ao))=0;
        [qp,qv,rv]=decode_action(ao,...
            d2.Q_pos_min,d2.Q_pos_max,...
            d2.Q_vel_min,d2.Q_vel_max,...
            d2.R_min,d2.R_max);

        Q2d=diag([qp qp qv qv]); R2d=diag([rv rv]);
        xk_rl=d2.Fk*xk_rl; Pk_rl=d2.Fk*Pk_rl*d2.Fk'+Q2d;
        Kk=Pk_rl*d2.Hk'/(d2.Hk*Pk_rl*d2.Hk'+R2d);
        xk_rl=xk_rl+Kk*(pg'-d2.Hk*xk_rl);
        Pk_rl=(eye(4)-Kk*d2.Hk)*Pk_rl;

        pe=xk_rl(1:2)';
        pe(1)=max(min(pe(1),W+2),-2); pe(2)=max(min(pe(2),H+2),-2);
        if any(isnan(pe))||any(isinf(pe))
            pe=pp; xk_rl=[pp(1);pp(2);0;0]; Pk_rl=eye(4);
        end
        kf2d_rl(:,k)=pe'; pp=pe;

        % Build next state with substituted r4_k
        er=sqrt((pe(1)-d2.anchors(:,1)).^2+(pe(2)-d2.anchors(:,2)).^2);
        e4=(er-r4_k)*100; v2=xk_rl(3:4);
        st=build_state_4a(r4_k,e4,v2,...
            mean(double(nlos_k)),W,H);
    end
end

%% MONTE CARLO SETUP
N_mc=30;

%% COMPUTE TRUE RANGES (same for all scenarios)
function true_r=compute_true_ranges(true_x,true_y,anchors,nAnc,N)
    true_r=zeros(nAnc,N);
    for a=1:nAnc
        true_r(a,:)=sqrt((true_x-anchors(a,1)).^2+...
                         (true_y-anchors(a,2)).^2);
    end
end

%% NOISE GENERATOR (standard)
function [raw_r,nlos_f]=gen_noise(true_r,nAnc,N)
    raw_r=zeros(nAnc,N); nlos_f=false(nAnc,N);
    for a=1:nAnc
        n=0.20*randn(1,N)+(rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
        raw_r(a,:)=max(true_r(a,:)+n,0.01);
        nlos_f(a,:)=(raw_r(a,:)-true_r(a,:))>0.20;
    end
end

%% DROPOUT PARAMETERS
dropout_anchor=1;
t_drop_start=20; t_drop_end=30;
k_start=round(t_drop_start*fs)+1;
k_end  =round(t_drop_end*fs)+1;

%% BASELINE
fprintf('%s\n',repmat('-',1,55));
fprintf('  BASELINE (standard noise)\n');
fprintf('%s\n',repmat('-',1,55));
rmse_base_fix=zeros(N_mc,1); rmse_base_rl=zeros(N_mc,1);
for mc=1:N_mc
    rng(mc*7+3);
    tr=compute_true_ranges(true_x,true_y,anchors_nom,nAnc,N);
    [rr,nf]=gen_noise(tr,nAnc,N);
    [kf_fix,kf_rl]=run_standard(rr,nf,anchors_nom,...
        Fk_2d,Hk_2d,Qk_fix,Rk_fix,d2,W,H,N);
    rmse_base_fix(mc)=sqrt(mean(sum((kf_fix-[true_x;true_y]).^2,1)*100^2));
    rmse_base_rl(mc) =sqrt(mean(sum((kf_rl -[true_x;true_y]).^2,1)*100^2));
end
fprintf('  Fixed=%.2f±%.2f cm  DDPG=%.2f±%.2f cm\n\n',...
    mean(rmse_base_fix),std(rmse_base_fix),...
    mean(rmse_base_rl),std(rmse_base_rl));

%% SCENARIO A — CLOCK DRIFT
fprintf('%s\n',repmat('-',1,55));
fprintf('  SCENARIO A: Clock Drift (~20 ppm per anchor)\n');
fprintf('%s\n',repmat('-',1,55));
rmse_A_fix=zeros(N_mc,1); rmse_A_rl=zeros(N_mc,1);
for mc=1:N_mc
    rng(mc*13+5);
    tr=compute_true_ranges(true_x,true_y,anchors_nom,nAnc,N);
    raw_r=zeros(nAnc,N); nlos_f=false(nAnc,N);
    for a=1:nAnc
        n=0.20*randn(1,N)+(rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
        A_d=0.02+0.05*rand(); f_d=0.005+0.015*rand(); phi_d=2*pi*rand();
        drift=A_d*sin(2*pi*f_d*(0:N-1)*dt+phi_d);
        raw_r(a,:)=max(tr(a,:)+n+drift,0.01);
        nlos_f(a,:)=(raw_r(a,:)-tr(a,:))>0.20;
    end
    [kf_fix,kf_rl]=run_standard(raw_r,nlos_f,anchors_nom,...
        Fk_2d,Hk_2d,Qk_fix,Rk_fix,d2,W,H,N);
    rmse_A_fix(mc)=sqrt(mean(sum((kf_fix-[true_x;true_y]).^2,1)*100^2));
    rmse_A_rl(mc) =sqrt(mean(sum((kf_rl -[true_x;true_y]).^2,1)*100^2));
    fprintf('  Seed %2d: Fixed=%.2f  DDPG=%.2f cm\n',mc,rmse_A_fix(mc),rmse_A_rl(mc));
end
fprintf('  Mean: Fixed=%.2f±%.2f  DDPG=%.2f±%.2f cm\n',...
    mean(rmse_A_fix),std(rmse_A_fix),mean(rmse_A_rl),std(rmse_A_rl));
fprintf('  DDPG better by: %.2f%%\n\n',...
    (1-mean(rmse_A_rl)/mean(rmse_A_fix))*100);

%% SCENARIO B — PLACEMENT ERROR
fprintf('%s\n',repmat('-',1,55));
fprintf('  SCENARIO B: Anchor Placement Error (±0.5 m)\n');
fprintf('%s\n',repmat('-',1,55));
rmse_B_fix=zeros(N_mc,1); rmse_B_rl=zeros(N_mc,1);
for mc=1:N_mc
    rng(mc*17+9);
    anc_disp=anchors_nom+0.5*(2*rand(nAnc,2)-1);
    tr=compute_true_ranges(true_x,true_y,anc_disp,nAnc,N);
    [raw_r,nlos_f]=gen_noise(tr,nAnc,N);
    % System uses NOMINAL positions (doesn't know true displaced positions)
    [kf_fix,kf_rl]=run_standard(raw_r,nlos_f,anchors_nom,...
        Fk_2d,Hk_2d,Qk_fix,Rk_fix,d2,W,H,N);
    rmse_B_fix(mc)=sqrt(mean(sum((kf_fix-[true_x;true_y]).^2,1)*100^2));
    rmse_B_rl(mc) =sqrt(mean(sum((kf_rl -[true_x;true_y]).^2,1)*100^2));
    off=mean(sqrt(sum((anc_disp-anchors_nom).^2,2)));
    fprintf('  Seed %2d: Fixed=%.2f  DDPG=%.2f cm  (offset=%.2f m)\n',...
        mc,rmse_B_fix(mc),rmse_B_rl(mc),off);
end
fprintf('  Mean: Fixed=%.2f±%.2f  DDPG=%.2f±%.2f cm\n',...
    mean(rmse_B_fix),std(rmse_B_fix),mean(rmse_B_rl),std(rmse_B_rl));
fprintf('  DST Target: Fixed=%s  DDPG=%s\n',...
    pass_fail(mean(rmse_B_fix)<50),pass_fail(mean(rmse_B_rl)<50));
fprintf('  DDPG better by: %.2f%%\n\n',...
    (1-mean(rmse_B_rl)/mean(rmse_B_fix))*100);

%% SCENARIO C — ANCHOR DROPOUT (FIXED)
fprintf('%s\n',repmat('-',1,55));
fprintf('  SCENARIO C: A1 Dropout (t=%.0f–%.0fs)\n',...
    t_drop_start,t_drop_end);
fprintf('  Fix: Substitute missing range with GN estimate, NLOS=1\n');
fprintf('%s\n',repmat('-',1,55));
rmse_C_fix=zeros(N_mc,1); rmse_C_rl=zeros(N_mc,1);
rmse_C_fix_d=zeros(N_mc,1); rmse_C_rl_d=zeros(N_mc,1);
idx_drop=k_start:k_end;

for mc=1:N_mc
    rng(mc*23+11);
    tr=compute_true_ranges(true_x,true_y,anchors_nom,nAnc,N);
    [raw_r,nlos_f]=gen_noise(tr,nAnc,N);

    [kf_fix,kf_rl]=run_dropout(raw_r,nlos_f,...
        dropout_anchor,k_start,k_end,anchors_nom,...
        Fk_2d,Hk_2d,Qk_fix,Rk_fix,d2,W,H,N);

    rmse_C_fix(mc)=sqrt(mean(sum((kf_fix-[true_x;true_y]).^2,1)*100^2));
    rmse_C_rl(mc) =sqrt(mean(sum((kf_rl -[true_x;true_y]).^2,1)*100^2));
    rmse_C_fix_d(mc)=sqrt(mean(sum(...
        (kf_fix(:,idx_drop)-[true_x(idx_drop);true_y(idx_drop)]).^2,1)*100^2));
    rmse_C_rl_d(mc) =sqrt(mean(sum(...
        (kf_rl(:,idx_drop) -[true_x(idx_drop);true_y(idx_drop)]).^2,1)*100^2));

    fprintf('  Seed %2d: Fixed=%.2f  DDPG=%.2f cm  (dropout: Fix=%.1f DDPG=%.1f)\n',...
        mc,rmse_C_fix(mc),rmse_C_rl(mc),rmse_C_fix_d(mc),rmse_C_rl_d(mc));
end
fprintf('  Full mission:   Fixed=%.2f±%.2f  DDPG=%.2f±%.2f cm\n',...
    mean(rmse_C_fix),std(rmse_C_fix),mean(rmse_C_rl),std(rmse_C_rl));
fprintf('  Dropout window: Fixed=%.2f±%.2f  DDPG=%.2f±%.2f cm\n',...
    mean(rmse_C_fix_d),std(rmse_C_fix_d),mean(rmse_C_rl_d),std(rmse_C_rl_d));
fprintf('  DST Target during dropout: Fixed=%s  DDPG=%s\n',...
    pass_fail(mean(rmse_C_fix_d)<50),pass_fail(mean(rmse_C_rl_d)<50));
fprintf('  DDPG better by: %.2f%% (full)  %.2f%% (dropout window)\n\n',...
    (1-mean(rmse_C_rl)/mean(rmse_C_fix))*100,...
    (1-mean(rmse_C_rl_d)/mean(rmse_C_fix_d))*100);

%% SUMMARY TABLE
fprintf('%s\n',repmat('=',1,75));
fprintf('  NOISE HARDENING SUMMARY  |  N=%d seeds per scenario\n',N_mc);
fprintf('%s\n',repmat('=',1,75));
scenarios={'Baseline','A: Clock Drift','B: Placement ±0.5m','C: A1 Dropout (full)'};
fix_means=[mean(rmse_base_fix) mean(rmse_A_fix) mean(rmse_B_fix) mean(rmse_C_fix)];
rl_means =[mean(rmse_base_rl)  mean(rmse_A_rl)  mean(rmse_B_rl)  mean(rmse_C_rl)];
fix_stds =[std(rmse_base_fix)  std(rmse_A_fix)  std(rmse_B_fix)  std(rmse_C_fix)];
rl_stds  =[std(rmse_base_rl)   std(rmse_A_rl)   std(rmse_B_rl)   std(rmse_C_rl)];
fprintf('  %-25s  %-16s  %-16s  %s\n','Scenario','Fixed 2D KF','DDPG 4-Anchor','Impr.');
fprintf('%s\n',repmat('-',1,75));
for i=1:4
    fprintf('  %-25s  %5.2f ± %4.2f cm  %5.2f ± %4.2f cm  %+.1f%%\n',...
        scenarios{i},fix_means(i),fix_stds(i),rl_means(i),rl_stds(i),...
        (1-rl_means(i)/fix_means(i))*100);
end
fprintf('%s\n\n',repmat('=',1,75));

%% COLOURS
Cfix=[0.50 0.10 0.80]; Crl=[0.05 0.72 0.32];
GS=[0.97 0.97 0.97]; RES=300;

%% FIG 1 — SCENARIO COMPARISON BAR CHART
figure('Color','w','Position',[50 50 820 440],'NumberTitle','off');
axes('Position',[0.11 0.18 0.85 0.70]); hold on;

n_sc=4; x_f=(1:n_sc)-0.18; x_r=(1:n_sc)+0.18;
bf=bar(x_f,fix_means,0.30,'FaceColor',Cfix,'EdgeColor','none');
br=bar(x_r,rl_means, 0.30,'FaceColor',Crl, 'EdgeColor','none');
errorbar(x_f,fix_means,fix_stds,'k','LineWidth',1.0,'LineStyle','none','CapSize',6);
errorbar(x_r,rl_means, rl_stds, 'k','LineWidth',1.0,'LineStyle','none','CapSize',6);
for i=1:n_sc
    text(x_f(i),fix_means(i)+fix_stds(i)+1.2,...
        sprintf('%.1f',fix_means(i)),'HorizontalAlignment','center',...
        'FontSize',7.5,'Color',Cfix,'FontWeight','bold');
    text(x_r(i),rl_means(i)+rl_stds(i)+1.2,...
        sprintf('%.1f',rl_means(i)),'HorizontalAlignment','center',...
        'FontSize',7.5,'Color',Crl,'FontWeight','bold');
end
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',1.0,...
    'Label','DST Target 50 cm','LabelHorizontalAlignment','right');
set(gca,'XTick',1:n_sc,'XTickLabel',...
    {'Baseline','Scenario A\nClock Drift',...
     'Scenario B\nPlacement ±0.5m','Scenario C\nA1 Dropout'},...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:60,'FontSize',8);
grid on; ylabel('2D RMSE (cm)'); ylim([0 60]);
legend([bf br],{'Fixed 2D KF','DDPG 4-Anchor'},...
    'Location','northwest','FontSize',9,'Box','off');
title('Noise Hardening: 2D RMSE Across All Scenarios  |  Mean ± Std  |  N=30 seeds',...
    'FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'nh_scenario_comparison.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 1 saved: nh_scenario_comparison.png\n'); close;

%% FIG 2 — DEGRADATION vs BASELINE
figure('Color','w','Position',[50 50 720 420],'NumberTitle','off');
axes('Position',[0.13 0.18 0.82 0.70]); hold on;
deg_f=fix_means(2:4)-fix_means(1);
deg_r=rl_means(2:4)-rl_means(1);
lbl_sc={'A: Clock Drift','B: Placement Error','C: A1 Dropout'};
bf2=bar((1:3)-0.18,deg_f,0.30,'FaceColor',Cfix,'EdgeColor','none');
br2=bar((1:3)+0.18,deg_r,0.30,'FaceColor',Crl, 'EdgeColor','none');
for i=1:3
    text(i-0.18,deg_f(i)+0.4,sprintf('+%.1f',deg_f(i)),...
        'HorizontalAlignment','center','FontSize',8,'Color',Cfix,'FontWeight','bold');
    text(i+0.18,deg_r(i)+0.4,sprintf('+%.1f',deg_r(i)),...
        'HorizontalAlignment','center','FontSize',8,'Color',Crl,'FontWeight','bold');
end
yline(0,'-','Color',[0.40 0.40 0.40],'LineWidth',0.8,'HandleVisibility','off');
set(gca,'XTick',1:3,'XTickLabel',lbl_sc,...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:2:20,'FontSize',8);
grid on; ylabel('RMSE Increase vs Baseline (cm)');
legend([bf2 br2],{'Fixed 2D KF','DDPG 4-Anchor'},...
    'Location','northwest','FontSize',9,'Box','off');
title('Degradation from Baseline per Hardening Scenario',...
    'FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'nh_degradation.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 2 saved: nh_degradation.png\n'); close;

%% FIG 3 — DROPOUT DETAIL
figure('Color','w','Position',[50 50 820 400],'NumberTitle','off');
axes('Position',[0.12 0.17 0.83 0.72]); hold on;
vd=[mean(rmse_C_fix_d) mean(rmse_C_rl_d)];
sd_=[std(rmse_C_fix_d)  std(rmse_C_rl_d)];
vf=[mean(rmse_C_fix)    mean(rmse_C_rl)];
sf_=[std(rmse_C_fix)     std(rmse_C_rl)];
b1=bar([1 2]-0.18,vd,0.30,'FaceColor','flat','EdgeColor','none');
b1.CData=[Cfix;Crl];
b2=bar([1 2]+0.18,vf,0.30,'FaceColor','flat','EdgeColor','none');
b2.CData=[Cfix*0.65;Crl*0.65];
errorbar([1 2]-0.18,vd,sd_,'k','LineStyle','none','CapSize',6,'LineWidth',1.0);
errorbar([1 2]+0.18,vf,sf_,'k','LineStyle','none','CapSize',6,'LineWidth',1.0);
for i=1:2
    text(i-0.18,vd(i)+sd_(i)+1.0,sprintf('%.1f',vd(i)),...
        'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
    text(i+0.18,vf(i)+sf_(i)+1.0,sprintf('%.1f',vf(i)),...
        'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
end
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',1.0,...
    'Label','DST Target 50 cm','LabelHorizontalAlignment','right');
set(gca,'XTick',[1 2],'XTickLabel',{'Fixed 2D KF','DDPG 4-Anchor'},...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:60,'FontSize',8);
grid on; ylabel('2D RMSE (cm)'); ylim([0 60]);
legend([b1(1) b2(1)],{'Dropout window (t=20–30s)','Full mission'},...
    'Location','northwest','FontSize',9,'Box','off');
title(sprintf('Scenario C: A1 Dropout  |  3-Anchor Fallback  |  Dropout window: Fixed=%s  DDPG=%s',...
    pass_fail(mean(rmse_C_fix_d)<50),pass_fail(mean(rmse_C_rl_d)<50)),...
    'FontSize',9,'FontWeight','bold');
exportgraphics(gcf,'nh_dropout_detail.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 3 saved: nh_dropout_detail.png\n'); close;

%% FIG 4 — SEED SCATTER
figure('Color','w','Position',[50 50 1000 380],'NumberTitle','off');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
sc_titles={'Scenario A: Clock Drift','Scenario B: Placement ±0.5m','Scenario C: A1 Dropout'};
fa={rmse_A_fix,rmse_B_fix,rmse_C_fix};
ra={rmse_A_rl, rmse_B_rl, rmse_C_rl};
for s=1:3
    nexttile; hold on;
    plot(1:N_mc,fa{s},'Color',Cfix,'LineWidth',0.9,...
        'DisplayName',sprintf('Fixed KF  %.1f cm',mean(fa{s})));
    plot(1:N_mc,ra{s},'Color',Crl,'LineWidth',1.2,...
        'DisplayName',sprintf('DDPG 4A  %.1f cm',mean(ra{s})));
    yline(mean(fa{s}),'--','Color',Cfix,'LineWidth',0.8,'Alpha',0.6,...
        'HandleVisibility','off');
    yline(mean(ra{s}),'--','Color',Crl,'LineWidth',0.8,'Alpha',0.6,...
        'HandleVisibility','off');
    yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',0.7,...
        'HandleVisibility','off','Label','DST 50cm');
    grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
    xlabel('Seed'); ylabel('RMSE (cm)'); ylim([0 60]);
    legend('Location','best','FontSize',7,'Box','off');
    title(sc_titles{s},'FontWeight','bold','FontSize',8);
end
sgtitle('Noise Hardening: RMSE per Seed  |  NIT Patna',...
    'FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'nh_seed_scatter.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 4 saved: nh_seed_scatter.png\n'); close;

%% EXPORT CSV
csv_file='uwb_noise_hardening_results.csv';
fid=fopen(csv_file,'w');
fprintf(fid,'Seed,Base_Fix,Base_DDPG,A_Fix,A_DDPG,B_Fix,B_DDPG,');
fprintf(fid,'C_Fix,C_DDPG,C_Fix_Drop,C_DDPG_Drop\n');
for mc=1:N_mc
    fprintf(fid,'%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n',...
        mc,rmse_base_fix(mc),rmse_base_rl(mc),...
        rmse_A_fix(mc),rmse_A_rl(mc),...
        rmse_B_fix(mc),rmse_B_rl(mc),...
        rmse_C_fix(mc),rmse_C_rl(mc),...
        rmse_C_fix_d(mc),rmse_C_rl_d(mc));
end
fclose(fid);
fprintf('Saved: %s  [%d rows]\n\n',csv_file,N_mc);

save('noise_hardening_results.mat',...
    'rmse_base_fix','rmse_base_rl',...
    'rmse_A_fix','rmse_A_rl',...
    'rmse_B_fix','rmse_B_rl',...
    'rmse_C_fix','rmse_C_rl',...
    'rmse_C_fix_d','rmse_C_rl_d',...
    'fix_means','rl_means','fix_stds','rl_stds','scenarios','N_mc');
fprintf('Saved: noise_hardening_results.mat\n\n');

fprintf('%s\n',repmat('=',1,65));
fprintf('  Task 4 complete.\n\n');
fprintf('  Scenario A (Clock Drift):      Fixed=%.2f  DDPG=%.2f cm  Impr=%.1f%%\n',...
    mean(rmse_A_fix),mean(rmse_A_rl),(1-mean(rmse_A_rl)/mean(rmse_A_fix))*100);
fprintf('  Scenario B (Placement ±0.5m):  Fixed=%.2f  DDPG=%.2f cm  Impr=%.1f%%\n',...
    mean(rmse_B_fix),mean(rmse_B_rl),(1-mean(rmse_B_rl)/mean(rmse_B_fix))*100);
fprintf('  Scenario C (A1 Dropout full):  Fixed=%.2f  DDPG=%.2f cm  Impr=%.1f%%\n',...
    mean(rmse_C_fix),mean(rmse_C_rl),(1-mean(rmse_C_rl)/mean(rmse_C_fix))*100);
fprintf('  Scenario C (dropout window):   Fixed=%.2f  DDPG=%.2f cm\n',...
    mean(rmse_C_fix_d),mean(rmse_C_rl_d));
all_pass=all([mean(rmse_A_rl) mean(rmse_B_rl) ...
              mean(rmse_C_rl) mean(rmse_C_rl_d)]<50);
fprintf('  All scenarios within DST 50cm: %s\n',pass_fail(all_pass));
fprintf('%s\n',repmat('=',1,65));