clear; clc; close all;

%% =========================================================
%  uwb_4anchor_deploy.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 2: Deploy 4-Anchor DDPG, Evaluate 2D Localisation
%
%  Compares four methods on identical raw UWB data (rng=7):
%    1. Raw GN          — Gauss-Newton on raw ranges, no filter
%    2. Fixed 2D KF     — GN on raw ranges + fixed Kalman
%    3. DDPG 4-Anchor   — GN on raw ranges + DDPG-tuned 2D Kalman
%    4. FIR + Fixed KF  — Month 1 full pipeline (reference only)
%
%  Primary comparison: methods 1,2,3 all use raw ranges (fair)
%  Method 4 shown as Month 1 reference (uses FIR pre-filter)
%
%  Saves: 5 figures + uwb_4anchor_results.csv
%  NIT Patna | Shrinivas V (2350011) | Dr. Golak Bihari Mahanta
%% =========================================================

%% PARALLEL POOL
nCores=feature('numcores'); cl=parcluster('local');
maxW=cl.NumWorkers; nUse=min(maxW,max(1,floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n',nCores,nUse);
if isempty(gcp('nocreate')), parpool('local',nUse); end

%% LOAD 4-ANCHOR AGENT
if ~exist('ddpg_4anchor_trained.mat','file')
    error('ddpg_4anchor_trained.mat not found. Run uwb_4anchor_train.m first.');
end
load('ddpg_4anchor_trained.mat');
fprintf('4-Anchor Agent loaded  (best ep: %d | train RMSE: %.4f cm)\n\n',...
    best_ep,best_rmse);

%% SIMULATION SETUP
T=60; t=0:1/fs:T; N=length(t); nAnc=4;

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

true_ranges=zeros(nAnc,N);
for a=1:nAnc
    true_ranges(a,:)=sqrt((true_x-anchors(a,1)).^2+(true_y-anchors(a,2)).^2);
end

%% NOISE
rng(7);
raw_r=zeros(nAnc,N); nlos_flags=false(nAnc,N);
for a=1:nAnc
    noise=0.20*randn(1,N)+(rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
    raw_r(a,:)=max(true_ranges(a,:)+noise,0.01);
    nlos_flags(a,:)=(raw_r(a,:)-true_ranges(a,:))>0.20;
end
nlos_any=any(nlos_flags,1);
fprintf('Mission: %.0fs | %d samples | %d NLOS events (%.1f%%)\n\n',...
    T,N,sum(nlos_any),100*mean(nlos_any));

%% HELPER FUNCTIONS
function a=act_fwd(n,s)
    a=tanh(n.W3*max(0,n.W2*max(0,n.W1*s+n.b1)+n.b2)+n.b3);
end
function [qp,qv,rv]=decode_action(a,qpn,qpx,qvn,qvx,rn,rx)
    qp=max(qpn,min(qpx,qpn+(a(1)+1)/2*(qpx-qpn)));
    qv=max(qvn,min(qvx,qvn+(a(2)+1)/2*(qvx-qvn)));
    rv=max(rn, min(rx, rn +(a(3)+1)/2*(rx -rn )));
end
function s=build_state(r4,e4,vel2,nlos_mean,W,H)
    D=sqrt(W^2+H^2);
    s=[r4/D; e4/100; vel2/5; double(nlos_mean)];
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

%% 2D KALMAN MATRICES (shared by fixed and DDPG methods)
Fk_kf=[1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
Hk_kf=[1 0 0 0; 0 1 0 0];

%% METHOD 1: RAW GAUSS-NEWTON
fprintf('Running Raw GN...\n');
est_raw=zeros(2,N); p0=[W/2 H/2];
for k=1:N
    try, est_raw(:,k)=gauss_newton(raw_r(:,k),anchors,p0)';
    catch, est_raw(:,k)=p0'; end
    p0=est_raw(:,k)';
end
est_raw(1,:)=max(min(est_raw(1,:),W+2),-2);
est_raw(2,:)=max(min(est_raw(2,:),H+2),-2);

%% METHOD 2: FIXED 2D KALMAN on raw ranges
fprintf('Running Fixed 2D KF (raw)...\n');
Qk_fix=diag([0.001 0.001 0.01 0.01]);
Rk_fix=diag([0.12^2 0.12^2]);
xk2=[est_raw(1,1);est_raw(2,1);0;0]; Pk2=eye(4);
kf2d_fix=zeros(2,N);
for k=1:N
    xk2=Fk_kf*xk2; Pk2=Fk_kf*Pk2*Fk_kf'+Qk_fix;
    Kk=Pk2*Hk_kf'/(Hk_kf*Pk2*Hk_kf'+Rk_fix);
    xk2=xk2+Kk*(est_raw(:,k)-Hk_kf*xk2);
    Pk2=(eye(4)-Kk*Hk_kf)*Pk2;
    kf2d_fix(:,k)=xk2(1:2);
end

%% METHOD 3: FIR + FIXED KF (Month 1 reference)
fprintf('Running FIR+KF (Month 1 reference)...\n');
b_fir=fir1(8,0.3); fir_r=zeros(nAnc,N);
for a=1:nAnc
    fir_r(a,:)=filtfilt(b_fir,1,medfilt1(raw_r(a,:),5));
end
est_fir=zeros(2,N); p0=[W/2 H/2];
for k=1:N
    try, est_fir(:,k)=gauss_newton(fir_r(:,k),anchors,p0)';
    catch, est_fir(:,k)=p0'; end
    p0=est_fir(:,k)';
end
est_fir(1,:)=max(min(est_fir(1,:),W+2),-2);
est_fir(2,:)=max(min(est_fir(2,:),H+2),-2);
xk3=[est_fir(1,1);est_fir(2,1);0;0]; Pk3=eye(4);
kf2d_fir=zeros(2,N);
for k=1:N
    xk3=Fk_kf*xk3; Pk3=Fk_kf*Pk3*Fk_kf'+Qk_fix;
    Kk=Pk3*Hk_kf'/(Hk_kf*Pk3*Hk_kf'+Rk_fix);
    xk3=xk3+Kk*(est_fir(:,k)-Hk_kf*xk3);
    Pk3=(eye(4)-Kk*Hk_kf)*Pk3;
    kf2d_fir(:,k)=xk3(1:2);
end

%% METHOD 4: DDPG 4-ANCHOR
fprintf('Running DDPG 4-Anchor...\n');
try
    p_init=gauss_newton(raw_r(:,1),anchors,[W/2 H/2]);
    p_init(1)=max(min(p_init(1),W+1),-1);
    p_init(2)=max(min(p_init(2),H+1),-1);
catch
    p_init=[W/2 H/2];
end
xk_rl=[p_init(1);p_init(2);0;0]; Pk_rl=eye(4);
kf2d_rl=zeros(2,N); qp_hist=zeros(1,N); rv_hist=zeros(1,N);
state=build_state(raw_r(:,1),zeros(4,1),[0;0],...
    mean(double(nlos_flags(:,1))),W,H);
pos_prev=p_init;

for k=1:N
    a_out=act_fwd(actor,state);
    a_out(isnan(a_out)|isinf(a_out))=0;
    [qp,qv,rv]=decode_action(a_out,...
        Q_pos_min,Q_pos_max,Q_vel_min,Q_vel_max,R_min,R_max);
    qp_hist(k)=qp; rv_hist(k)=rv;

    try, pos_gn=gauss_newton(raw_r(:,k),anchors,pos_prev);
    catch, pos_gn=pos_prev; end
    pos_gn(1)=max(min(pos_gn(1),W+2),-2);
    pos_gn(2)=max(min(pos_gn(2),H+2),-2);

    Q2d=diag([qp qp qv qv]); R2d=diag([rv rv]);
    xk_rl=Fk*xk_rl; Pk_rl=Fk*Pk_rl*Fk'+Q2d;
    Kk=Pk_rl*Hk'/(Hk*Pk_rl*Hk'+R2d);
    xk_rl=xk_rl+Kk*(pos_gn'-Hk*xk_rl);
    Pk_rl=(eye(4)-Kk*Hk)*Pk_rl;

    pos_est=xk_rl(1:2)';
    pos_est(1)=max(min(pos_est(1),W+2),-2);
    pos_est(2)=max(min(pos_est(2),H+2),-2);
    if any(isnan(pos_est))||any(isinf(pos_est))
        pos_est=pos_prev;
        xk_rl=[pos_prev(1);pos_prev(2);0;0]; Pk_rl=eye(4);
    end
    kf2d_rl(:,k)=pos_est';
    pos_prev=pos_est;

    est_r=sqrt((pos_est(1)-anchors(:,1)).^2+(pos_est(2)-anchors(:,2)).^2);
    e4=(est_r-raw_r(:,k))*100;
    vel2=xk_rl(3:4);
    state=build_state(raw_r(:,k),e4,vel2,...
        mean(double(nlos_flags(:,k))),W,H);
end
fprintf('All methods done.\n\n');

%% ERRORS AND RUNNING RMSE
err_raw2d=sqrt(sum((est_raw -[true_x;true_y]).^2,1))*100;
err_fix2d=sqrt(sum((kf2d_fix-[true_x;true_y]).^2,1))*100;
err_fir2d=sqrt(sum((kf2d_fir-[true_x;true_y]).^2,1))*100;
err_rl2d =sqrt(sum((kf2d_rl -[true_x;true_y]).^2,1))*100;

rmse_raw2d=zeros(1,N); rmse_fix2d=zeros(1,N);
rmse_fir2d=zeros(1,N); rmse_rl2d=zeros(1,N);
a1=0;a2=0;a3=0;a4=0;
for k=1:N
    a1=a1+err_raw2d(k)^2; rmse_raw2d(k)=sqrt(a1/k);
    a2=a2+err_fix2d(k)^2; rmse_fix2d(k)=sqrt(a2/k);
    a3=a3+err_fir2d(k)^2; rmse_fir2d(k)=sqrt(a3/k);
    a4=a4+err_rl2d(k)^2;  rmse_rl2d(k) =sqrt(a4/k);
end

%% PRINT SUMMARY
fprintf('%s\n',repmat('=',1,65));
fprintf('  4-ANCHOR 2D LOCALISATION RESULTS  (60s, rng=7)\n');
fprintf('%s\n',repmat('=',1,65));
fprintf('  %-28s  %8s  %12s\n','Method','RMSE(cm)','vs Raw GN');
fprintf('  %-28s  %8.2f  %12s\n','Raw GN (no filter)',rmse_raw2d(N),'Baseline');
fprintf('  %-28s  %8.2f  %+9.1f%%\n','Fixed 2D KF (raw)',...
    rmse_fix2d(N),(1-rmse_fix2d(N)/rmse_raw2d(N))*100);
fprintf('  %-28s  %8.2f  %+9.1f%%\n','DDPG 4-Anchor',...
    rmse_rl2d(N),(1-rmse_rl2d(N)/rmse_raw2d(N))*100);
fprintf('  %-28s  %8.2f  %+9.1f%%  [Month 1 ref]\n','FIR + Fixed KF',...
    rmse_fir2d(N),(1-rmse_fir2d(N)/rmse_raw2d(N))*100);
fprintf('\n  DDPG vs Fixed 2D KF (fair): %+.2f%%\n',...
    (1-rmse_rl2d(N)/rmse_fix2d(N))*100);
fprintf('  DST Target (<=50 cm): DDPG = %.2f cm  -->  ',rmse_rl2d(N));
if rmse_rl2d(N)<50, fprintf('PASSED\n'); else, fprintf('FAILED\n'); end
fprintf('\n  Q_pos: mean=%.6f  std=%.6f\n',mean(qp_hist),std(qp_hist));
fprintf('  R_2D : mean=%.6f  std=%.6f\n',mean(rv_hist),std(rv_hist));
fprintf('%s\n\n',repmat('=',1,65));

%% COLOURS
Craw=[0.92 0.22 0.08]; Cfix=[0.50 0.10 0.80];
Cfir=[0.60 0.15 0.85]; Crl=[0.05 0.72 0.32];
Cq=[0.92 0.55 0.05];   Cr=[0.08 0.45 0.88];
GS=[0.97 0.97 0.97];   RES=300;

%% FIG 1 — TRAJECTORY
figure('Color','w','Position',[50 50 640 580],'NumberTitle','off');
axes('Position',[0.12 0.10 0.84 0.84]); hold on;
rectangle('Position',[0 0 W H],'EdgeColor',[0.70 0.70 0.70],...
    'LineWidth',1.0,'LineStyle','--');
plot(true_x,true_y,'Color',[0.95 0.82 0.05],'LineWidth',1.2,...
    'DisplayName','Planned Path');
plot(kf2d_fix(1,:),kf2d_fix(2,:),'Color',Cfix,'LineWidth',0.9,...
    'DisplayName',sprintf('Fixed 2D KF  (%.1f cm)',rmse_fix2d(N)));
plot(kf2d_rl(1,:),kf2d_rl(2,:),'Color',Crl,'LineWidth',1.3,...
    'DisplayName',sprintf('DDPG 4-Anchor  (%.1f cm)',rmse_rl2d(N)));
for a=1:nAnc
    scatter(anchors(a,1),anchors(a,2),120,'r','s','filled');
    text(anchors(a,1)+0.5,anchors(a,2)+0.7,sprintf('A%d',a),...
        'FontSize',9,'FontWeight','bold','Color',[0.75 0.08 0.08]);
end
xlim([-2 W+3]); ylim([-2 H+3]); axis equal;
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on');
xlabel('X (m)'); ylabel('Y (m)');
title('2D Trajectory: Fixed KF vs DDPG 4-Anchor','FontSize',10,'FontWeight','bold');
legend('Location','best','FontSize',8,'Box','off');
exportgraphics(gcf,'4a_trajectory.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 1 saved: 4a_trajectory.png\n'); close;

%% FIG 2 — RUNNING RMSE
figure('Color','w','Position',[50 50 1000 380],'NumberTitle','off');
axes('Position',[0.08 0.14 0.87 0.78]); hold on;
patch([t fliplr(t)],[rmse_fix2d fliplr(rmse_rl2d)],...
    [0.80 0.90 0.80],'FaceAlpha',0.20,'EdgeColor','none');
plot(t,rmse_raw2d,'Color',Craw,'LineWidth',0.8,...
    'DisplayName',sprintf('Raw GN  (%.1f cm)',rmse_raw2d(N)));
plot(t,rmse_fix2d,'Color',Cfix,'LineWidth',1.2,...
    'DisplayName',sprintf('Fixed 2D KF  (%.1f cm)',rmse_fix2d(N)));
plot(t,rmse_rl2d, 'Color',Crl, 'LineWidth',1.6,...
    'DisplayName',sprintf('DDPG 4-Anchor  (%.1f cm)',rmse_rl2d(N)));
plot(t,rmse_fir2d,'Color',Cfir,'LineWidth',1.0,'LineStyle','--',...
    'DisplayName',sprintf('FIR+KF Month 1  (%.1f cm)',rmse_fir2d(N)));
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',1.0,...
    'Label','DST 50 cm','LabelHorizontalAlignment','left','HandleVisibility','off');
text(61.5,rmse_fix2d(N),sprintf('%.1f',rmse_fix2d(N)),...
    'FontSize',8,'Color',Cfix,'FontWeight','bold','VerticalAlignment','middle');
text(61.5,rmse_rl2d(N),sprintf('%.1f',rmse_rl2d(N)),...
    'FontSize',8,'Color',Crl,'FontWeight','bold','VerticalAlignment','middle');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
xlabel('Time (s)'); ylabel('2D Running RMSE (cm)');
title('2D Position RMSE: All Methods','FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',8,'Box','off');
xlim([0 64]);
exportgraphics(gcf,'4a_running_rmse.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 2 saved: 4a_running_rmse.png\n'); close;

%% FIG 3 — RMSE BAR
figure('Color','w','Position',[50 50 580 460],'NumberTitle','off');
axes('Position',[0.14 0.20 0.82 0.68]); hold on;
vals=[rmse_raw2d(N) rmse_fix2d(N) rmse_rl2d(N) rmse_fir2d(N)];
clrs=[Craw;Cfix;Crl;Cfir];
lbls={'Raw GN','Fixed 2D KF','DDPG 4-Anchor','FIR+KF (M1)'};
bh=bar(vals,'FaceColor','flat','EdgeColor','none','BarWidth',0.55);
bh.CData=clrs;
set(gca,'XTick',1:4,'XTickLabel',lbls,'Color',GS,...
    'GridAlpha',0.12,'Box','on','YTick',0:5:80,...
    'XTickLabelRotation',15);
grid on; ylabel('2D RMSE (cm)');
title('2D Localisation RMSE Comparison','FontSize',10,'FontWeight','bold');
ylim([0 max(vals)*1.45]);
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',1.0,...
    'Label','DST Target','LabelHorizontalAlignment','right');
for i=1:4
    text(i,vals(i)+0.8,sprintf('%.1f',vals(i)),...
        'HorizontalAlignment','center','FontSize',9,...
        'FontWeight','bold','Color',[0.15 0.15 0.15]);
end
exportgraphics(gcf,'4a_rmse_bar.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 3 saved: 4a_rmse_bar.png\n'); close;

%% FIG 4 — Q AND R ADAPTATION
figure('Color','w','Position',[50 50 1000 420],'NumberTitle','off');
subplot(2,1,1); hold on;
plot(t,qp_hist,'Color',Cq,'LineWidth',1.0,'DisplayName','Q_{pos}');
scatter(t(nlos_any),qp_hist(nlos_any),10,[0.75 0.10 0.10],...
    'o','filled','MarkerFaceAlpha',0.4,'DisplayName','NLOS event');
set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
ylabel('Q_{pos}'); xlim([0 60]); grid on;
legend('Location','northeast','FontSize',8,'Box','off');
title('DDPG 4-Anchor: Adaptive 2D Kalman Parameters','FontSize',10,'FontWeight','bold');

subplot(2,1,2); hold on;
plot(t,rv_hist,'Color',Cr,'LineWidth',1.0,'DisplayName','R_{2D}');
scatter(t(nlos_any),rv_hist(nlos_any),10,[0.75 0.10 0.10],...
    'o','filled','MarkerFaceAlpha',0.4,'DisplayName','NLOS event');
set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
ylabel('R_{2D}'); xlabel('Time (s)'); xlim([0 60]); grid on;
legend('Location','northeast','FontSize',8,'Box','off');
exportgraphics(gcf,'4a_qr_adaptation.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 4 saved: 4a_qr_adaptation.png\n'); close;

%% FIG 5 — 2D POSITION ERROR
figure('Color','w','Position',[50 50 1000 360],'NumberTitle','off');
axes('Position',[0.08 0.14 0.88 0.78]); hold on;
plot(t,err_raw2d,'Color',Craw,'LineWidth',0.6,'DisplayName','Raw GN');
plot(t,err_fix2d,'Color',Cfix,'LineWidth',1.0,'DisplayName','Fixed 2D KF');
plot(t,err_rl2d, 'Color',Crl, 'LineWidth',1.4,'DisplayName','DDPG 4-Anchor');
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',0.8,...
    'HandleVisibility','off','Label','DST 50 cm');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
xlabel('Time (s)'); ylabel('2D Position Error (cm)');
title('2D Position Error Over Time','FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',8,'Box','off');
xlim([0 60]);
exportgraphics(gcf,'4a_position_error.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 5 saved: 4a_position_error.png\n'); close;

%% EXPORT CSV
csv_file='uwb_4anchor_results.csv';
fid=fopen(csv_file,'w');
fprintf(fid,'Sample,Time_s,TrueX_m,TrueY_m,');
fprintf(fid,'RawGN_X,RawGN_Y,FixKF_X,FixKF_Y,DDPG_X,DDPG_Y,');
fprintf(fid,'Err_Raw_cm,Err_Fix_cm,Err_DDPG_cm,');
fprintf(fid,'RMSE_Raw_cm,RMSE_Fix_cm,RMSE_DDPG_cm,');
fprintf(fid,'Qpos,R_2D,NLOS_Count\n');
for k=1:N
    fprintf(fid,'%d,%.4f,%.4f,%.4f,',k,t(k),true_x(k),true_y(k));
    fprintf(fid,'%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,',...
        est_raw(1,k),est_raw(2,k),...
        kf2d_fix(1,k),kf2d_fix(2,k),...
        kf2d_rl(1,k),kf2d_rl(2,k));
    fprintf(fid,'%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,',...
        err_raw2d(k),err_fix2d(k),err_rl2d(k),...
        rmse_raw2d(k),rmse_fix2d(k),rmse_rl2d(k));
    fprintf(fid,'%.8f,%.8f,%d\n',...
        qp_hist(k),rv_hist(k),sum(nlos_flags(:,k)));
end
fclose(fid);
fprintf('Saved: %s  [%d rows]\n\n',csv_file,N);

fprintf('%s\n',repmat('=',1,65));
fprintf('  Task 2 complete.\n');
fprintf('  Raw GN RMSE        : %.2f cm\n',rmse_raw2d(N));
fprintf('  Fixed 2D KF RMSE   : %.2f cm\n',rmse_fix2d(N));
fprintf('  DDPG 4-Anchor RMSE : %.2f cm\n',rmse_rl2d(N));
fprintf('  DDPG vs Fixed KF   : %+.2f%%\n',...
    (1-rmse_rl2d(N)/rmse_fix2d(N))*100);
fprintf('  FIR+KF Month 1 ref : %.2f cm\n',rmse_fir2d(N));
if rmse_rl2d(N)<50
    fprintf('  DST Target (50cm)  : PASSED\n');
else
    fprintf('  DST Target (50cm)  : FAILED\n');
end
fprintf('%s\n',repmat('=',1,65));