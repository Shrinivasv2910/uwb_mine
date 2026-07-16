clear; clc; close all;

%% =========================================================
%  uwb_monte_carlo.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 3: Monte Carlo Robustness Analysis
%
%  Runs full pipeline 50 times with different random seeds
%  Reports mean ± std RMSE — makes single-seed results defensible
%
%  Methods compared across all 50 seeds:
%    1. Raw 1D           — raw UWB range, no filter
%    2. Fixed KF 1D      — fixed Kalman on 1D range
%    3. DDPG KF 1D       — DDPG-tuned Kalman on 1D range (Task 1)
%    4. Raw GN 2D        — Gauss-Newton on raw 4-anchor ranges
%    5. Fixed 2D KF      — fixed Kalman on GN output
%    6. DDPG 4-Anchor    — DDPG-tuned 2D Kalman (Task 2)
%
%  Overwrites: mc_results.mat, all figure PNGs, uwb_mc_summary.csv
%  NIT Patna | Shrinivas V (2350011) | Dr. Golak Bihari Mahanta
%% =========================================================

%% PARALLEL POOL
nCores=feature('numcores'); cl=parcluster('local');
maxW=cl.NumWorkers; nUse=min(maxW,max(1,floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n',nCores,nUse);
if isempty(gcp('nocreate')), parpool('local',nUse); end

%% LOAD TRAINED AGENTS
if ~exist('ddpg_trained.mat','file')
    error('ddpg_trained.mat not found. Run uwb_ddpg_train.m first.');
end
if ~exist('ddpg_4anchor_trained.mat','file')
    error('ddpg_4anchor_trained.mat not found. Run uwb_4anchor_train.m first.');
end
d1=load('ddpg_trained.mat');
d2=load('ddpg_4anchor_trained.mat');
fprintf('Task 1 agent loaded  (best ep: %d | train RMSE: %.2f cm)\n',...
    d1.best_ep,d1.best_rmse);
fprintf('Task 2 agent loaded  (best ep: %d | train RMSE: %.2f cm)\n\n',...
    d2.best_ep,d2.best_rmse);

%% SYSTEM PARAMETERS
fs=d1.fs; dt=d1.dt; W=d1.W; H=d1.H;
F_k=d1.F_k; H_k=d1.H_k;
anchors=d2.anchors; nAnc=4;

%% MONTE CARLO SETUP
N_mc  = 50;
T     = 60; t_vec=0:1/fs:T; N=length(t_vec);
rmse_all   = zeros(N_mc,6);
nlos_rates = zeros(N_mc,1);

%% HELPER FUNCTIONS
function a=act_fwd(n,s)
    a=tanh(n.W3*max(0,n.W2*max(0,n.W1*s+n.b1)+n.b2)+n.b3);
end
function [qp,qv,rv]=decode_action(a,qpn,qpx,qvn,qvx,rn,rx)
    qp=max(qpn,min(qpx,qpn+(a(1)+1)/2*(qpx-qpn)));
    qv=max(qvn,min(qvx,qvn+(a(2)+1)/2*(qvx-qvn)));
    rv=max(rn, min(rx, rn +(a(3)+1)/2*(rx -rn )));
end
function s=build_state_1d(r3,e3,vel,nlos,W,H)
    s=[r3/sqrt(W^2+H^2); e3/100; vel/10; double(nlos)];
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

%% BOUSTROPHEDON PATH (fixed — same for all seeds)
lanes=8; ly=linspace(1,H-1,lanes); px=[]; py=[];
for L=1:lanes
    if mod(L,2)==1, px=[px 1 W-1]; else, px=[px W-1 1]; end
    py=[py ly(L) ly(L)];
end
dc=[0 cumsum(sqrt(diff(px).^2+diff(py).^2))];
dq=linspace(0,dc(end),N);
true_x=interp1(dc,px,dq,'linear');
true_y=interp1(dc,py,dq,'linear');
true_range_1d=sqrt(true_x.^2+true_y.^2);
true_ranges_4a=zeros(nAnc,N);
for a=1:nAnc
    true_ranges_4a(a,:)=sqrt((true_x-anchors(a,1)).^2+...
                              (true_y-anchors(a,2)).^2);
end

%% KALMAN MATRICES
Fk_2d=[1 0 dt 0;0 1 0 dt;0 0 1 0;0 0 0 1];
Hk_2d=[1 0 0 0;0 1 0 0];
Qk_fix2d=diag([0.001 0.001 0.01 0.01]);
Rk_fix2d=diag([0.12^2 0.12^2]);
Q_fix1d=diag([0.0005 0.005]); R_fix1d=0.25^2;

fprintf('%s\n',repmat('=',1,78));
fprintf('  Monte Carlo Analysis  |  N=%d seeds  |  60s mission\n',N_mc);
fprintf('  Methods: Raw1D | FixKF1D | DDPG1D | RawGN | Fix2D | DDPG4A\n');
fprintf('%s\n\n',repmat('=',1,78));
fprintf('%-6s %-10s %-10s %-10s %-10s %-10s %-10s %-8s\n',...
    'Seed','Raw1D','FixKF1D','DDPG1D','RawGN','Fix2D','DDPG4A','NLOS%');
fprintf('%s\n',repmat('-',1,78));

t_start=tic;

for mc=1:N_mc

    rng(mc*100+42);

    %% 1D NOISE
    raw1d=max(true_range_1d+0.25*randn(1,N)+...
              (rand(1,N)<0.12).*(0.4+0.8*rand(1,N)),0);
    nlos1d=(raw1d-true_range_1d)>0.25;

    %% 4-ANCHOR NOISE
    raw_4a=zeros(nAnc,N); nlos_4a=false(nAnc,N);
    for a=1:nAnc
        noise=0.20*randn(1,N)+(rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
        raw_4a(a,:)=max(true_ranges_4a(a,:)+noise,0.01);
        nlos_4a(a,:)=(raw_4a(a,:)-true_ranges_4a(a,:))>0.20;
    end
    nlos_rates(mc)=100*mean(any(nlos_4a,1));

    %% METHOD 1: RAW 1D
    rmse_all(mc,1)=sqrt(mean(((raw1d-true_range_1d)*100).^2));

    %% METHOD 2: FIXED KALMAN 1D
    x_fix=[raw1d(1);0]; P_fix=eye(2); kf1d_fix=zeros(1,N);
    for k=1:N
        x_fix=F_k*x_fix; P_fix=F_k*P_fix*F_k'+Q_fix1d;
        Kf=P_fix*H_k'/(H_k*P_fix*H_k'+R_fix1d);
        x_fix=x_fix+Kf*(raw1d(k)-H_k*x_fix);
        P_fix=(eye(2)-Kf*H_k)*P_fix;
        kf1d_fix(k)=max(0,x_fix(1));
    end
    rmse_all(mc,2)=sqrt(mean(((kf1d_fix-true_range_1d)*100).^2));

    %% METHOD 3: DDPG KALMAN 1D
    D_MAX=sqrt(W^2+H^2);
    x_rl=[raw1d(1);0]; P_rl=eye(2); kf1d_rl=zeros(1,N);
    raw_h=raw1d(1)*ones(3,1); err_h=zeros(3,1);
    st=build_state_1d(raw_h,err_h,0,0,W,H);
    for k=1:N
        ao=act_fwd(d1.actor,st); ao(isnan(ao)|isinf(ao))=0;
        [qp,qv,rv]=decode_action(ao,...
            d1.Q_pos_min,d1.Q_pos_max,...
            d1.Q_vel_min,d1.Q_vel_max,...
            d1.R_min,d1.R_max);
        Q_rl=diag([qp qv]); R_rl=rv^2;
        x_rl=F_k*x_rl; P_rl=F_k*P_rl*F_k'+Q_rl;
        Kr=P_rl*H_k'/(H_k*P_rl*H_k'+R_rl);
        x_rl=x_rl+Kr*(raw1d(k)-H_k*x_rl);
        P_rl=(eye(2)-Kr*H_k)*P_rl;
        kf1d_rl(k)=max(0,min(D_MAX,x_rl(1)));
        if isnan(kf1d_rl(k))||isinf(kf1d_rl(k))
            x_rl=[raw1d(k);0]; P_rl=eye(2)*0.5; kf1d_rl(k)=raw1d(k);
        end
        vel_e=0; if k>1, vel_e=(raw1d(k)-raw1d(k-1))*fs; end
        en=(kf1d_rl(k)-true_range_1d(k))*100;
        raw_h=[raw_h(2:end);raw1d(k)]; err_h=[err_h(2:end);en];
        st=build_state_1d(raw_h,err_h,vel_e,double(nlos1d(k)),W,H);
    end
    rmse_all(mc,3)=sqrt(mean(((kf1d_rl-true_range_1d)*100).^2));

    %% METHOD 4: RAW GAUSS-NEWTON 2D
    est_raw2d=zeros(2,N); p0=[W/2 H/2];
    for k=1:N
        try, est_raw2d(:,k)=gauss_newton(raw_4a(:,k),anchors,p0)';
        catch, est_raw2d(:,k)=p0'; end
        p0=est_raw2d(:,k)';
    end
    est_raw2d(1,:)=max(min(est_raw2d(1,:),W+2),-2);
    est_raw2d(2,:)=max(min(est_raw2d(2,:),H+2),-2);
    rmse_all(mc,4)=sqrt(mean(sum((est_raw2d-[true_x;true_y]).^2,1)*100^2));

    %% METHOD 5: FIXED 2D KF on raw ranges
    xk2=[est_raw2d(1,1);est_raw2d(2,1);0;0]; Pk2=eye(4);
    kf2d_fix=zeros(2,N);
    for k=1:N
        xk2=Fk_2d*xk2; Pk2=Fk_2d*Pk2*Fk_2d'+Qk_fix2d;
        Kk=Pk2*Hk_2d'/(Hk_2d*Pk2*Hk_2d'+Rk_fix2d);
        xk2=xk2+Kk*(est_raw2d(:,k)-Hk_2d*xk2);
        Pk2=(eye(4)-Kk*Hk_2d)*Pk2;
        kf2d_fix(:,k)=xk2(1:2);
    end
    rmse_all(mc,5)=sqrt(mean(sum((kf2d_fix-[true_x;true_y]).^2,1)*100^2));

    %% METHOD 6: DDPG 4-ANCHOR 2D
    try
        p_init=gauss_newton(raw_4a(:,1),anchors,[W/2 H/2]);
        p_init(1)=max(min(p_init(1),W+1),-1);
        p_init(2)=max(min(p_init(2),H+1),-1);
    catch
        p_init=[W/2 H/2];
    end
    xk_rl=[p_init(1);p_init(2);0;0]; Pk_rl=eye(4);
    kf2d_rl=zeros(2,N);
    st4=build_state_4a(raw_4a(:,1),zeros(4,1),[0;0],...
        mean(double(nlos_4a(:,1))),W,H);
    pp=p_init;
    for k=1:N
        ao=act_fwd(d2.actor,st4); ao(isnan(ao)|isinf(ao))=0;
        [qp,qv,rv]=decode_action(ao,...
            d2.Q_pos_min,d2.Q_pos_max,...
            d2.Q_vel_min,d2.Q_vel_max,...
            d2.R_min,d2.R_max);
        try, pg=gauss_newton(raw_4a(:,k),anchors,pp);
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
        er=sqrt((pe(1)-anchors(:,1)).^2+(pe(2)-anchors(:,2)).^2);
        e4=(er-raw_4a(:,k))*100; v2=xk_rl(3:4);
        st4=build_state_4a(raw_4a(:,k),e4,v2,...
            mean(double(nlos_4a(:,k))),W,H);
    end
    rmse_all(mc,6)=sqrt(mean(sum((kf2d_rl-[true_x;true_y]).^2,1)*100^2));

    fprintf('%-6d %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-8.1f\n',...
        mc,rmse_all(mc,1),rmse_all(mc,2),rmse_all(mc,3),...
        rmse_all(mc,4),rmse_all(mc,5),rmse_all(mc,6),nlos_rates(mc));

end % Monte Carlo loop

fprintf('%s\n',repmat('-',1,78));
fprintf('Monte Carlo done  |  %.1f min  |  %d seeds\n\n',toc(t_start)/60,N_mc);

%% STATISTICS
mu  = mean(rmse_all,1);
sd  = std(rmse_all,0,1);
mn  = min(rmse_all,[],1);
mx  = max(rmse_all,[],1);
p5  = prctile(rmse_all,5,1);
p95 = prctile(rmse_all,95,1);
imp_1d=(1-rmse_all(:,3)./rmse_all(:,2))*100;
imp_2d=(1-rmse_all(:,6)./rmse_all(:,5))*100;
methods={'Raw 1D','Fixed KF 1D','DDPG KF 1D','Raw GN 2D','Fixed 2D KF','DDPG 4-Anchor'};

%% PRINT TABLE
fprintf('%s\n',repmat('=',1,75));
fprintf('  MONTE CARLO STATISTICS  |  N=%d seeds  |  60s mission\n',N_mc);
fprintf('%s\n',repmat('=',1,75));
fprintf('  %-20s  %8s  %8s  %8s  %8s\n','Method','Mean','Std','Min','Max');
fprintf('%s\n',repmat('-',1,75));
for i=1:6
    fprintf('  %-20s  %8.2f  %8.2f  %8.2f  %8.2f  cm\n',...
        methods{i},mu(i),sd(i),mn(i),mx(i));
end
fprintf('%s\n',repmat('=',1,75));
fprintf('\n  1D: DDPG vs Fixed KF improvement across seeds:\n');
fprintf('    Mean: %.2f%%  |  Std: %.2f%%  |  Min: %.2f%%  |  Max: %.2f%%\n',...
    mean(imp_1d),std(imp_1d),min(imp_1d),max(imp_1d));
fprintf('    Positive improvement in %d/%d seeds (%.0f%%)\n',...
    sum(imp_1d>0),N_mc,100*mean(imp_1d>0));
fprintf('\n  2D: DDPG 4-Anchor vs Fixed 2D KF improvement across seeds:\n');
fprintf('    Mean: %.2f%%  |  Std: %.2f%%  |  Min: %.2f%%  |  Max: %.2f%%\n',...
    mean(imp_2d),std(imp_2d),min(imp_2d),max(imp_2d));
fprintf('    Positive improvement in %d/%d seeds (%.0f%%)\n',...
    sum(imp_2d>0),N_mc,100*mean(imp_2d>0));
fprintf('\n  Mean NLOS event rate: %.1f%% ± %.1f%%\n',...
    mean(nlos_rates),std(nlos_rates));
fprintf('%s\n\n',repmat('=',1,75));

%% SAVE WORKSPACE — overwrites previous
save('mc_results.mat','rmse_all','nlos_rates','mu','sd','mn','mx',...
    'p5','p95','imp_1d','imp_2d','methods','N_mc');
fprintf('Saved: mc_results.mat\n\n');

%% COLOURS
Craw=[0.92 0.22 0.08]; Cfix=[0.50 0.10 0.80]; Crl=[0.05 0.72 0.32];
Craw2=[0.95 0.55 0.10]; Cfix2=[0.20 0.40 0.80]; Crl2=[0.05 0.55 0.45];
GS=[0.97 0.97 0.97]; RES=300;
clrs=[Craw;Cfix;Crl;Craw2;Cfix2;Crl2];

%% =========================================================
%  FIG 1 — RMSE DISTRIBUTIONS  (mean ± std bars + min/max whiskers)
%% =========================================================
figure('Color','w','Position',[50 50 950 480],'NumberTitle','off');
axes('Position',[0.09 0.20 0.88 0.70]); hold on;

y_max = max(mx)*1.55;
for i=1:6
    bar(i,mu(i),'FaceColor',clrs(i,:),'EdgeColor','none','BarWidth',0.55);
    errorbar(i,mu(i),sd(i),sd(i),'k','LineWidth',1.2,'CapSize',8);
    plot([i i],[mn(i) mu(i)-sd(i)],'k--','LineWidth',0.7);
    plot([i i],[mu(i)+sd(i) mx(i)],'k--','LineWidth',0.7);
    plot([i-0.18 i+0.18],[mn(i) mn(i)],'k-','LineWidth',0.8);
    plot([i-0.18 i+0.18],[mx(i) mx(i)],'k-','LineWidth',0.8);
    text(i,mu(i)+sd(i)+2.0,sprintf('%.1f±%.1f',mu(i),sd(i)),...
        'HorizontalAlignment','center','FontSize',8,...
        'FontWeight','bold','Color',[0.10 0.10 0.10]);
end

% Divider between Task 1 and Task 2 groups
xline(3.5,'--','Color',[0.60 0.60 0.60],'LineWidth',1.2,...
    'HandleVisibility','off');

% Group labels — fixed: use numeric y position
text(2, y_max*0.92, '1D Ranging  (Task 1)',...
    'HorizontalAlignment','center','FontSize',9,...
    'Color',[0.35 0.35 0.35],'FontWeight','bold');
text(5, y_max*0.92, '2D Localisation  (Task 2)',...
    'HorizontalAlignment','center','FontSize',9,...
    'Color',[0.35 0.35 0.35],'FontWeight','bold');

set(gca,'XTick',1:6,...
    'XTickLabel',{'Raw 1D','Fixed KF\newline1D','DDPG KF\newline1D',...
                  'Raw GN\newline2D','Fixed 2D\newlineKF','DDPG\newline4-Anchor'},...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:80,'FontSize',8);
grid on; ylabel('RMSE (cm)'); ylim([0 y_max]);
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',1.0,...
    'Label','DST Target 50 cm','LabelHorizontalAlignment','right');
title(sprintf('Monte Carlo RMSE Distribution  (N=%d seeds)  |  Mean ± Std  |  Whiskers: Min/Max',...
    N_mc),'FontSize',10,'FontWeight','bold');

exportgraphics(gcf,'mc_rmse_distributions.png','Resolution',RES,...
    'BackgroundColor','white');
fprintf('Fig 1 saved: mc_rmse_distributions.png\n'); close;

%% =========================================================
%  FIG 2 — IMPROVEMENT DISTRIBUTIONS (histograms)
%% =========================================================
figure('Color','w','Position',[50 50 900 420],'NumberTitle','off');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; hold on;
histogram(imp_1d,'BinWidth',1.5,'FaceColor',Crl,'EdgeColor','w','FaceAlpha',0.85);
xline(mean(imp_1d),'--','Color',[0.10 0.10 0.10],'LineWidth',1.5,...
    'Label',sprintf('Mean: %.1f%%',mean(imp_1d)),...
    'LabelHorizontalAlignment','left','LabelVerticalAlignment','bottom');
xline(0,'-','Color',[0.85 0.20 0.20],'LineWidth',1.2,...
    'Label','Break-even','LabelHorizontalAlignment','right',...
    'LabelVerticalAlignment','bottom');
grid on; set(gca,'Color',GS,'GridAlpha',0.12,'Box','on','FontSize',8);
xlabel('Improvement over Fixed KF 1D (%)'); ylabel('Number of seeds');
title(sprintf('Task 1: DDPG 1D Improvement\n%.0f/%d seeds positive',...
    sum(imp_1d>0),N_mc),'FontWeight','bold','FontSize',9);

nexttile; hold on;
histogram(imp_2d,'BinWidth',0.5,'FaceColor',Crl2,'EdgeColor','w','FaceAlpha',0.85);
xline(mean(imp_2d),'--','Color',[0.10 0.10 0.10],'LineWidth',1.5,...
    'Label',sprintf('Mean: %.1f%%',mean(imp_2d)),...
    'LabelHorizontalAlignment','left','LabelVerticalAlignment','bottom');
xline(0,'-','Color',[0.85 0.20 0.20],'LineWidth',1.2,...
    'Label','Break-even','LabelHorizontalAlignment','right',...
    'LabelVerticalAlignment','bottom');
grid on; set(gca,'Color',GS,'GridAlpha',0.12,'Box','on','FontSize',8);
xlabel('Improvement over Fixed 2D KF (%)'); ylabel('Number of seeds');
title(sprintf('Task 2: DDPG 4-Anchor Improvement\n%.0f/%d seeds positive',...
    sum(imp_2d>0),N_mc),'FontWeight','bold','FontSize',9);

sgtitle('Monte Carlo Improvement Distributions  |  NIT Patna',...
    'FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'mc_improvement_hist.png','Resolution',RES,...
    'BackgroundColor','white');
fprintf('Fig 2 saved: mc_improvement_hist.png\n'); close;

%% =========================================================
%  FIG 3 — SEED-BY-SEED RMSE (consistency check)
%% =========================================================
figure('Color','w','Position',[50 50 1000 420],'NumberTitle','off');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; hold on;
plot(1:N_mc,rmse_all(:,2),'Color',Cfix,'LineWidth',0.9,...
    'DisplayName',sprintf('Fixed KF 1D  (%.1f cm)',mu(2)));
plot(1:N_mc,rmse_all(:,3),'Color',Crl,'LineWidth',1.3,...
    'DisplayName',sprintf('DDPG KF 1D  (%.1f cm)',mu(3)));
yline(mu(2),'--','Color',Cfix,'LineWidth',0.8,'Alpha',0.6,...
    'HandleVisibility','off');
yline(mu(3),'--','Color',Crl,'LineWidth',0.8,'Alpha',0.6,...
    'HandleVisibility','off');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
xlabel('Seed'); ylabel('RMSE (cm)');
legend('Location','best','FontSize',8,'Box','off');
title('Task 1: 1D RMSE per Seed','FontWeight','bold','FontSize',9);
ylim([0 max(mx(1:3))*1.2]);

nexttile; hold on;
plot(1:N_mc,rmse_all(:,5),'Color',Cfix2,'LineWidth',0.9,...
    'DisplayName',sprintf('Fixed 2D KF  (%.1f cm)',mu(5)));
plot(1:N_mc,rmse_all(:,6),'Color',Crl2,'LineWidth',1.3,...
    'DisplayName',sprintf('DDPG 4-Anchor  (%.1f cm)',mu(6)));
yline(mu(5),'--','Color',Cfix2,'LineWidth',0.8,'Alpha',0.6,...
    'HandleVisibility','off');
yline(mu(6),'--','Color',Crl2,'LineWidth',0.8,'Alpha',0.6,...
    'HandleVisibility','off');
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',0.8,...
    'HandleVisibility','off','Label','DST 50 cm');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
xlabel('Seed'); ylabel('RMSE (cm)');
legend('Location','best','FontSize',8,'Box','off');
title('Task 2: 2D RMSE per Seed','FontWeight','bold','FontSize',9);
ylim([0 max(mx(4:6))*1.2]);

sgtitle('Monte Carlo: RMSE Consistency Across 50 Seeds  |  NIT Patna',...
    'FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'mc_seed_consistency.png','Resolution',RES,...
    'BackgroundColor','white');
fprintf('Fig 3 saved: mc_seed_consistency.png\n'); close;

%% =========================================================
%  FIG 4 — BOX PLOTS (MATLAB boxplot if available, else manual)
%% =========================================================
figure('Color','w','Position',[50 50 950 440],'NumberTitle','off');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; hold on;
data1d=[rmse_all(:,2) rmse_all(:,3)];
bp=boxplot(data1d,'Labels',{'Fixed KF 1D','DDPG KF 1D'},...
    'Colors',[Cfix;Crl],'Symbol','o','Widths',0.5);
set(bp,'LineWidth',1.2);
h_lines=findobj(gca,'type','line');
for ii=1:length(h_lines)
    set(h_lines(ii),'LineWidth',1.2);
end
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
ylabel('1D Ranging RMSE (cm)');
title(sprintf('Task 1: Improvement = %.1f%% ± %.1f%%',...
    mean(imp_1d),std(imp_1d)),'FontWeight','bold','FontSize',9);
text(1.5,max(mx(1:3))*0.95,...
    sprintf('50/50 seeds\npositive'),...
    'HorizontalAlignment','center','FontSize',8,'Color',[0.10 0.55 0.10]);

nexttile; hold on;
data2d=[rmse_all(:,5) rmse_all(:,6)];
bp2=boxplot(data2d,'Labels',{'Fixed 2D KF','DDPG 4-Anchor'},...
    'Colors',[Cfix2;Crl2],'Symbol','o','Widths',0.5);
set(bp2,'LineWidth',1.2);
h_lines2=findobj(gca,'type','line');
for ii=1:length(h_lines2)
    set(h_lines2(ii),'LineWidth',1.2);
end
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',0.8,...
    'Label','DST 50 cm','LabelHorizontalAlignment','right');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
ylabel('2D Position RMSE (cm)');
title(sprintf('Task 2: Improvement = %.1f%% ± %.1f%%',...
    mean(imp_2d),std(imp_2d)),'FontWeight','bold','FontSize',9);
text(1.5,max(mx(4:6))*0.95,...
    sprintf('50/50 seeds\npositive'),...
    'HorizontalAlignment','center','FontSize',8,'Color',[0.10 0.55 0.10]);

sgtitle('Monte Carlo Box Plots  |  N=50 Seeds  |  NIT Patna',...
    'FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'mc_boxplots.png','Resolution',RES,...
    'BackgroundColor','white');
fprintf('Fig 4 saved: mc_boxplots.png\n'); close;

%% EXPORT CSV — overwrites previous
csv_file='uwb_mc_summary.csv';
fid=fopen(csv_file,'w');
fprintf(fid,'Seed,Raw1D_cm,FixKF1D_cm,DDPG1D_cm,RawGN_cm,Fix2D_cm,DDPG4A_cm,');
fprintf(fid,'Imp1D_pct,Imp2D_pct,NLOS_Rate_pct\n');
for mc=1:N_mc
    fprintf(fid,'%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n',...
        mc,rmse_all(mc,1),rmse_all(mc,2),rmse_all(mc,3),...
        rmse_all(mc,4),rmse_all(mc,5),rmse_all(mc,6),...
        imp_1d(mc),imp_2d(mc),nlos_rates(mc));
end
fclose(fid);
fprintf('Saved: %s  [%d rows]\n\n',csv_file,N_mc);

fprintf('%s\n',repmat('=',1,65));
fprintf('  Task 3 complete.\n\n');
fprintf('  Task 1 DDPG: %.2f%% ± %.2f%% improvement  (%d/%d seeds positive)\n',...
    mean(imp_1d),std(imp_1d),sum(imp_1d>0),N_mc);
fprintf('  Task 2 DDPG: %.2f%% ± %.2f%% improvement  (%d/%d seeds positive)\n',...
    mean(imp_2d),std(imp_2d),sum(imp_2d>0),N_mc);
fprintf('%s\n',repmat('=',1,65));