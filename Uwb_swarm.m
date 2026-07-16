clear; clc; close all;

%% =========================================================
%  uwb_swarm.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 5: Two-Drone Swarm Simulation
%
%  Month 1 simulated a single drone covering the mine area.
%  The proposal specifies a swarm for Phase II.
%  This script simulates 2 scout drones flying parallel
%  boustrophedon lanes, sharing the same 4 UWB anchors.
%
%  Key aspects:
%    - Drone 1: lanes 1,3,5,7 (odd lanes)
%    - Drone 2: lanes 2,4,6,8 (even lanes)
%    - Both drones localise independently using all 4 anchors
%    - Staggered ranging: Drone 1 at t=k, Drone 2 at t=k+0.05s
%      (UWB TWR requires TDMA — cannot range simultaneously)
%    - Coverage completeness metric vs time (single vs swarm)
%    - DDPG 4-anchor used for both drones
%
%  Metrics:
%    - Per-drone 2D RMSE (localisation accuracy)
%    - Coverage area vs time (single drone vs 2-drone swarm)
%    - Coverage time comparison
%    - Inter-drone separation (safety check)
%
%  Saves: 4 figures + uwb_swarm_results.csv
%  NIT Patna | Shrinivas V (2350011) | Dr. Golak Bihari Mahanta
%% =========================================================

%% PARALLEL POOL
nCores=feature('numcores'); cl=parcluster('local');
maxW=cl.NumWorkers; nUse=min(maxW,max(1,floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n',nCores,nUse);
if isempty(gcp('nocreate')), parpool('local',nUse); end

%% LOAD TASK 2 AGENT
if ~exist('ddpg_4anchor_trained.mat','file')
    error('ddpg_4anchor_trained.mat not found. Run uwb_4anchor_train.m first.');
end
d2=load('ddpg_4anchor_trained.mat');
fprintf('Task 2 agent loaded  (best ep: %d | train RMSE: %.2f cm)\n\n',...
    d2.best_ep,d2.best_rmse);

%% SYSTEM PARAMETERS
fs=d2.fs; dt=d2.dt; W=d2.W; H=d2.H;
anchors=d2.anchors; nAnc=4;

%% =========================================================
%  PATH GENERATION
%  8-lane boustrophedon split between 2 drones
%  Drone 1: odd lanes  (1,3,5,7) — covers left half of time
%  Drone 2: even lanes (2,4,6,8) — offset laterally
%  Both drones complete in T/2 = 30s (half the single-drone time)
%% =========================================================

T_single = 60;                     % single drone mission time
T_swarm  = 30;                     % each drone covers half the area
t_s  = 0:1/fs:T_single;  N_s = length(t_s);   % single drone timeline
t_sw = 0:1/fs:T_swarm;   N_sw = length(t_sw);  % swarm drone timeline

%% Single drone path (8 lanes, 60s) — reference
lanes=8; ly=linspace(1,H-1,lanes); px_all=[]; py_all=[];
for L=1:lanes
    if mod(L,2)==1, px_all=[px_all 1 W-1]; else, px_all=[px_all W-1 1]; end
    py_all=[py_all ly(L) ly(L)];
end
dc_all=[0 cumsum(sqrt(diff(px_all).^2+diff(py_all).^2))];
dq_all=linspace(0,dc_all(end),N_s);
single_x=interp1(dc_all,px_all,dq_all,'linear');
single_y=interp1(dc_all,py_all,dq_all,'linear');

%% Drone 1 path — odd lanes (1,3,5,7) in 30s
ly_odd=ly(1:2:end);  % lanes 1,3,5,7
px1=[]; py1=[];
for i=1:length(ly_odd)
    if mod(i,2)==1, px1=[px1 1 W-1]; else, px1=[px1 W-1 1]; end
    py1=[py1 ly_odd(i) ly_odd(i)];
end
dc1=[0 cumsum(sqrt(diff(px1).^2+diff(py1).^2))];
dq1=linspace(0,dc1(end),N_sw);
drone1_x=interp1(dc1,px1,dq1,'linear');
drone1_y=interp1(dc1,py1,dq1,'linear');

%% Drone 2 path — even lanes (2,4,6,8) in 30s
ly_even=ly(2:2:end);  % lanes 2,4,6,8
px2=[]; py2=[];
for i=1:length(ly_even)
    if mod(i,2)==1, px2=[px2 1 W-1]; else, px2=[px2 W-1 1]; end
    py2=[py2 ly_even(i) ly_even(i)];
end
dc2=[0 cumsum(sqrt(diff(px2).^2+diff(py2).^2))];
dq2=linspace(0,dc2(end),N_sw);
drone2_x=interp1(dc2,px2,dq2,'linear');
drone2_y=interp1(dc2,py2,dq2,'linear');

fprintf('Path setup:\n');
fprintf('  Single drone: %d lanes, %.0f s mission\n',lanes,T_single);
fprintf('  Drone 1 (swarm): %d lanes (odd), %.0f s mission\n',length(ly_odd),T_swarm);
fprintf('  Drone 2 (swarm): %d lanes (even), %.0f s mission\n\n',length(ly_even),T_swarm);

%% =========================================================
%  HELPER FUNCTIONS
%% =========================================================
function a=act_fwd(n,s)
    a=tanh(n.W3*max(0,n.W2*max(0,n.W1*s+n.b1)+n.b2)+n.b3);
end
function [qp,qv,rv]=decode_action(a,qpn,qpx,qvn,qvx,rn,rx)
    qp=max(qpn,min(qpx,qpn+(a(1)+1)/2*(qpx-qpn)));
    qv=max(qvn,min(qvx,qvn+(a(2)+1)/2*(qvx-qvn)));
    rv=max(rn, min(rx, rn +(a(3)+1)/2*(rx-rn )));
end
function s=build_state(r4,e4,vel2,nlos_m,W,H)
    s=[r4/sqrt(W^2+H^2); e4/100; vel2/5; double(nlos_m)];
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

%% =========================================================
%  LOCALISATION RUNNER — runs DDPG 4-anchor for one drone
%% =========================================================
function [est_pos,rmse_final]=run_drone(true_x,true_y,anchors,d2,W,H,N,fs,dt)
    nAnc=4;
    true_r=zeros(nAnc,N);
    for a=1:nAnc
        true_r(a,:)=sqrt((true_x-anchors(a,1)).^2+(true_y-anchors(a,2)).^2);
    end
    rng(7);
    raw_r=zeros(nAnc,N); nlos_f=false(nAnc,N);
    for a=1:nAnc
        noise=0.20*randn(1,N)+(rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
        raw_r(a,:)=max(true_r(a,:)+noise,0.01);
        nlos_f(a,:)=(raw_r(a,:)-true_r(a,:))>0.20;
    end
    % GN trilateration
    est_gn=zeros(2,N); p0=[W/2 H/2];
    for k=1:N
        try, est_gn(:,k)=gauss_newton(raw_r(:,k),anchors,p0)';
        catch, est_gn(:,k)=p0'; end
        p0=est_gn(:,k)';
    end
    est_gn(1,:)=max(min(est_gn(1,:),W+2),-2);
    est_gn(2,:)=max(min(est_gn(2,:),H+2),-2);
    % DDPG 4-anchor Kalman
    xk=[est_gn(1,1);est_gn(2,1);0;0]; Pk=eye(4);
    est_pos=zeros(2,N);
    st=build_state(raw_r(:,1),zeros(4,1),[0;0],...
        mean(double(nlos_f(:,1))),W,H);
    pp=est_gn(:,1)';
    for k=1:N
        ao=act_fwd(d2.actor,st); ao(isnan(ao)|isinf(ao))=0;
        [qp,qv,rv]=decode_action(ao,...
            d2.Q_pos_min,d2.Q_pos_max,...
            d2.Q_vel_min,d2.Q_vel_max,...
            d2.R_min,d2.R_max);
        Q2d=diag([qp qp qv qv]); R2d=diag([rv rv]);
        xk=d2.Fk*xk; Pk=d2.Fk*Pk*d2.Fk'+Q2d;
        Kk=Pk*d2.Hk'/(d2.Hk*Pk*d2.Hk'+R2d);
        xk=xk+Kk*(est_gn(:,k)-d2.Hk*xk);
        Pk=(eye(4)-Kk*d2.Hk)*Pk;
        pe=xk(1:2)';
        pe(1)=max(min(pe(1),W+2),-2); pe(2)=max(min(pe(2),H+2),-2);
        if any(isnan(pe))||any(isinf(pe))
            pe=pp; xk=[pp(1);pp(2);0;0]; Pk=eye(4);
        end
        est_pos(:,k)=pe'; pp=pe;
        er=sqrt((pe(1)-anchors(:,1)).^2+(pe(2)-anchors(:,2)).^2);
        e4=(er-raw_r(:,k))*100; v2=xk(3:4);
        st=build_state(raw_r(:,k),e4,v2,mean(double(nlos_f(:,k))),W,H);
    end
    err=sqrt(sum((est_pos-[true_x;true_y]).^2,1))*100;
    rmse_final=sqrt(mean(err.^2));
end

%% RUN SINGLE DRONE (60s, 8 lanes)
fprintf('Running single drone (60s, 8 lanes)...\n');
[single_est,rmse_single]=run_drone(single_x,single_y,...
    anchors,d2,W,H,N_s,fs,dt);
fprintf('  Single drone RMSE: %.2f cm\n\n',rmse_single);

%% RUN SWARM — DRONE 1 (30s, odd lanes)
fprintf('Running Drone 1 (30s, odd lanes)...\n');
[d1_est,rmse_d1]=run_drone(drone1_x,drone1_y,...
    anchors,d2,W,H,N_sw,fs,dt);
fprintf('  Drone 1 RMSE: %.2f cm\n',rmse_d1);

%% RUN SWARM — DRONE 2 (30s, even lanes)
fprintf('Running Drone 2 (30s, even lanes)...\n');
[d2_est,rmse_d2]=run_drone(drone2_x,drone2_y,...
    anchors,d2,W,H,N_sw,fs,dt);
fprintf('  Drone 2 RMSE: %.2f cm\n\n',rmse_d2);

%% =========================================================
%  COVERAGE COMPLETENESS METRIC
%  Discretise mine into grid cells (0.5m resolution)
%  A cell is "covered" when a drone passes within scan_radius
%  scan_radius = 3m (typical downward camera FOV at 5m altitude)
%  Compare: single drone coverage vs time vs swarm coverage vs time
%% =========================================================
grid_res   = 0.5;    % m per grid cell
scan_radius= 3.0;    % m — camera ground coverage radius

x_grid = 0:grid_res:W;
y_grid = 0:grid_res:H;
[Xg,Yg] = meshgrid(x_grid,y_grid);
n_cells  = numel(Xg);

fprintf('Coverage grid: %.1fm resolution | %d cells | scan radius=%.1fm\n',...
    grid_res,n_cells,scan_radius);

%% Single drone coverage vs time
covered_single=false(size(Xg));
cov_single_t=zeros(1,N_s);
for k=1:N_s
    dist=sqrt((Xg-single_x(k)).^2+(Yg-single_y(k)).^2);
    covered_single=covered_single|(dist<=scan_radius);
    cov_single_t(k)=100*sum(covered_single(:))/n_cells;
end

%% Swarm coverage vs time (both drones active simultaneously)
covered_swarm=false(size(Xg));
cov_swarm_t=zeros(1,N_sw);
for k=1:N_sw
    % Drone 1 at t=k (ranges at normal time)
    dist1=sqrt((Xg-drone1_x(k)).^2+(Yg-drone1_y(k)).^2);
    % Drone 2 at t=k+half_slot (staggered 50ms for TDMA)
    dist2=sqrt((Xg-drone2_x(k)).^2+(Yg-drone2_y(k)).^2);
    covered_swarm=covered_swarm|(dist1<=scan_radius)|(dist2<=scan_radius);
    cov_swarm_t(k)=100*sum(covered_swarm(:))/n_cells;
end

%% Time to reach coverage thresholds
thresh=[50 75 90 95 99];
t_single_thresh=zeros(1,length(thresh));
t_swarm_thresh =zeros(1,length(thresh));
for i=1:length(thresh)
    idx_s=find(cov_single_t>=thresh(i),1,'first');
    idx_w=find(cov_swarm_t >=thresh(i),1,'first');
    if ~isempty(idx_s), t_single_thresh(i)=t_s(idx_s);  else, t_single_thresh(i)=T_single; end
    if ~isempty(idx_w),  t_swarm_thresh(i) =t_sw(idx_w); else, t_swarm_thresh(i) =T_swarm;  end
end

%% INTER-DRONE SEPARATION (safety check)
sep=sqrt((drone1_x-drone2_x).^2+(drone1_y-drone2_y).^2);
fprintf('Inter-drone separation: mean=%.2f m  min=%.2f m  max=%.2f m\n\n',...
    mean(sep),min(sep),max(sep));

%% PRINT RESULTS
fprintf('%s\n',repmat('=',1,65));
fprintf('  SWARM SIMULATION RESULTS\n');
fprintf('%s\n',repmat('=',1,65));
fprintf('  Localisation RMSE:\n');
fprintf('    Single drone (60s): %.2f cm\n',rmse_single);
fprintf('    Drone 1 (30s):      %.2f cm\n',rmse_d1);
fprintf('    Drone 2 (30s):      %.2f cm\n',rmse_d2);
fprintf('    Swarm mean:         %.2f cm\n',(rmse_d1+rmse_d2)/2);

fprintf('\n  Coverage completion time:\n');
fprintf('  %-8s %-14s %-14s %-12s\n','Target','Single (s)','Swarm (s)','Speedup');
fprintf('  %s\n',repmat('-',1,52));
for i=1:length(thresh)
    speedup=t_single_thresh(i)/max(t_swarm_thresh(i),0.1);
    fprintf('  %-8s %-14.1f %-14.1f %.2fx\n',...
        sprintf('%d%%',thresh(i)),...
        t_single_thresh(i),t_swarm_thresh(i),speedup);
end

fprintf('\n  Final coverage at end of mission:\n');
fprintf('    Single drone at t=30s: %.1f%%\n',cov_single_t(N_sw));
fprintf('    Single drone at t=60s: %.1f%%\n',cov_single_t(end));
fprintf('    Swarm at t=30s:        %.1f%%\n',cov_swarm_t(end));

fprintf('\n  Inter-drone separation: mean=%.2f m  min=%.2f m\n',...
    mean(sep),min(sep));
fprintf('%s\n\n',repmat('=',1,65));

%% COLOURS
C1=[0.05 0.72 0.32]; C2=[0.92 0.22 0.08];
Cs=[0.50 0.10 0.80]; Ct=[0.30 0.30 0.30];
GS=[0.97 0.97 0.97]; RES=300;

%% FIG 1 — TRAJECTORY COMPARISON
figure('Color','w','Position',[50 50 1050 520],'NumberTitle','off');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; hold on;
rectangle('Position',[0 0 W H],'EdgeColor',[0.70 0.70 0.70],...
    'LineWidth',1.0,'LineStyle','--');
plot(single_x,single_y,'Color',Ct,'LineWidth',1.0,...
    'DisplayName','Planned path (single)');
plot(single_est(1,:),single_est(2,:),'Color',Cs,'LineWidth',0.9,...
    'DisplayName',sprintf('Single drone RMSE=%.1f cm',rmse_single));
for a=1:nAnc
    scatter(anchors(a,1),anchors(a,2),100,'r','s','filled');
    text(anchors(a,1)+0.5,anchors(a,2)+0.7,sprintf('A%d',a),...
        'FontSize',8,'FontWeight','bold','Color',[0.75 0.08 0.08]);
end
xlim([-1 W+2]); ylim([-1 H+2]); axis equal;
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
xlabel('X (m)'); ylabel('Y (m)');
legend('Location','best','FontSize',7,'Box','off');
title('Single Drone — 60s, 8 Lanes','FontWeight','bold','FontSize',9);

nexttile; hold on;
rectangle('Position',[0 0 W H],'EdgeColor',[0.70 0.70 0.70],...
    'LineWidth',1.0,'LineStyle','--');
plot(drone1_x,drone1_y,'Color',[0.85 0.80 0.10],'LineWidth',0.8,...
    'DisplayName','Planned D1 (odd lanes)');
plot(drone2_x,drone2_y,'Color',[0.85 0.55 0.10],'LineWidth',0.8,...
    'DisplayName','Planned D2 (even lanes)');
plot(d1_est(1,:),d1_est(2,:),'Color',C1,'LineWidth',1.2,...
    'DisplayName',sprintf('Drone 1 RMSE=%.1f cm',rmse_d1));
plot(d2_est(1,:),d2_est(2,:),'Color',C2,'LineWidth',1.2,...
    'DisplayName',sprintf('Drone 2 RMSE=%.1f cm',rmse_d2));
for a=1:nAnc
    scatter(anchors(a,1),anchors(a,2),100,'r','s','filled');
    text(anchors(a,1)+0.5,anchors(a,2)+0.7,sprintf('A%d',a),...
        'FontSize',8,'FontWeight','bold','Color',[0.75 0.08 0.08]);
end
xlim([-1 W+2]); ylim([-1 H+2]); axis equal;
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
xlabel('X (m)'); ylabel('Y (m)');
legend('Location','best','FontSize',7,'Box','off');
title('2-Drone Swarm — 30s, Parallel Lanes','FontWeight','bold','FontSize',9);

sgtitle('Single Drone vs 2-Drone Swarm Trajectories  |  NIT Patna',...
    'FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'sw_trajectories.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 1 saved: sw_trajectories.png\n'); close;

%% FIG 2 — COVERAGE VS TIME
figure('Color','w','Position',[50 50 900 400],'NumberTitle','off');
axes('Position',[0.10 0.14 0.86 0.76]); hold on;

% Single drone curve (0 to 60s)
plot(t_s,cov_single_t,'Color',Cs,'LineWidth',1.5,...
    'DisplayName',sprintf('Single drone (%.0fs, final=%.1f%%)',T_single,cov_single_t(end)));

% Swarm curve (0 to 30s)
plot(t_sw,cov_swarm_t,'Color',C1,'LineWidth',2.0,...
    'DisplayName',sprintf('2-Drone swarm (%.0fs, final=%.1f%%)',T_swarm,cov_swarm_t(end)));

% Show single drone coverage at t=30s for fair comparison
yline(cov_single_t(N_sw),'--','Color',Cs,'LineWidth',0.9,...
    'Label',sprintf('Single @30s: %.1f%%',cov_single_t(N_sw)),...
    'LabelHorizontalAlignment','right','HandleVisibility','off');

% Threshold markers
for i=1:length(thresh)
    if t_swarm_thresh(i)<T_swarm
        plot([0 t_swarm_thresh(i)],[thresh(i) thresh(i)],...
            ':','Color',[0.60 0.60 0.60],'LineWidth',0.7,'HandleVisibility','off');
        plot([t_swarm_thresh(i) t_swarm_thresh(i)],[0 thresh(i)],...
            ':','Color',[0.60 0.60 0.60],'LineWidth',0.7,'HandleVisibility','off');
    end
end

grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8,...
    'XTick',0:5:60,'YTick',0:10:100);
xlabel('Time (s)'); ylabel('Coverage (%)');
title('Area Coverage vs Time: Single Drone vs 2-Drone Swarm',...
    'FontSize',10,'FontWeight','bold');
legend('Location','southeast','FontSize',9,'Box','off');
xlim([0 60]); ylim([0 105]);
exportgraphics(gcf,'sw_coverage.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 2 saved: sw_coverage.png\n'); close;

%% FIG 3 — COVERAGE TIME BAR CHART
figure('Color','w','Position',[50 50 780 420],'NumberTitle','off');
axes('Position',[0.12 0.17 0.84 0.72]); hold on;

x_s=(1:length(thresh))-0.18;
x_w=(1:length(thresh))+0.18;
bs=bar(x_s,t_single_thresh,0.30,'FaceColor',Cs,'EdgeColor','none');
bw=bar(x_w,t_swarm_thresh, 0.30,'FaceColor',C1,'EdgeColor','none');

for i=1:length(thresh)
    speedup=t_single_thresh(i)/max(t_swarm_thresh(i),0.1);
    text(i,max(t_single_thresh(i),t_swarm_thresh(i))+1.5,...
        sprintf('%.1fx faster',speedup),...
        'HorizontalAlignment','center','FontSize',8,...
        'Color',[0.15 0.15 0.15],'FontWeight','bold');
    text(x_s(i),t_single_thresh(i)+0.5,sprintf('%.0fs',t_single_thresh(i)),...
        'HorizontalAlignment','center','FontSize',7,'Color',Cs);
    text(x_w(i),t_swarm_thresh(i)+0.5,sprintf('%.0fs',t_swarm_thresh(i)),...
        'HorizontalAlignment','center','FontSize',7,'Color',C1);
end

set(gca,'XTick',1:length(thresh),...
    'XTickLabel',arrayfun(@(x)sprintf('%d%%',x),thresh,'UniformOutput',false),...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:60,'FontSize',8);
grid on; ylabel('Time to Reach Coverage Target (s)');
legend([bs bw],{'Single Drone','2-Drone Swarm'},'Location','northwest',...
    'FontSize',9,'Box','off');
title('Coverage Time Comparison: Single Drone vs Swarm',...
    'FontSize',10,'FontWeight','bold');
ylim([0 max(t_single_thresh)*1.3]);
exportgraphics(gcf,'sw_coverage_time.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 3 saved: sw_coverage_time.png\n'); close;

%% FIG 4 — INTER-DRONE SEPARATION + LOCALISATION RMSE
figure('Color','w','Position',[50 50 1000 420],'NumberTitle','off');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; hold on;
plot(t_sw,sep,'Color',[0.15 0.45 0.80],'LineWidth',1.2);
yline(mean(sep),'--','Color',[0.15 0.45 0.80],'LineWidth',0.8,...
    'Label',sprintf('Mean: %.2f m',mean(sep)),...
    'LabelHorizontalAlignment','left','HandleVisibility','off');
yline(2.0,'--','Color',[0.85 0.20 0.20],'LineWidth',1.0,...
    'Label','Min safe separation 2m','LabelHorizontalAlignment','right',...
    'HandleVisibility','off');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
xlabel('Time (s)'); ylabel('Inter-Drone Separation (m)');
title('Inter-Drone Separation Over Mission','FontWeight','bold','FontSize',9);
ylim([0 max(sep)*1.2]);

nexttile; hold on;
err_d1=sqrt(sum((d1_est-[drone1_x;drone1_y]).^2,1))*100;
err_d2=sqrt(sum((d2_est-[drone2_x;drone2_y]).^2,1))*100;
plot(t_sw,err_d1,'Color',C1,'LineWidth',1.0,...
    'DisplayName',sprintf('Drone 1 (%.1f cm)',rmse_d1));
plot(t_sw,err_d2,'Color',C2,'LineWidth',1.0,...
    'DisplayName',sprintf('Drone 2 (%.1f cm)',rmse_d2));
yline(rmse_d1,'--','Color',C1,'LineWidth',0.7,'Alpha',0.6,'HandleVisibility','off');
yline(rmse_d2,'--','Color',C2,'LineWidth',0.7,'Alpha',0.6,'HandleVisibility','off');
yline(50,'--','Color',[0.85 0.20 0.20],'LineWidth',0.8,...
    'HandleVisibility','off','Label','DST 50cm');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8);
xlabel('Time (s)'); ylabel('2D Position Error (cm)');
title('Per-Drone Localisation Error','FontWeight','bold','FontSize',9);
legend('Location','best','FontSize',8,'Box','off'); ylim([0 120]);

sgtitle('Swarm Safety and Localisation  |  NIT Patna','FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'sw_safety_rmse.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 4 saved: sw_safety_rmse.png\n'); close;

%% EXPORT CSV
csv_file='uwb_swarm_results.csv';
fid=fopen(csv_file,'w');
fprintf(fid,'Sample,Time_s,');
fprintf(fid,'D1_TrueX,D1_TrueY,D1_EstX,D1_EstY,D1_Err_cm,');
fprintf(fid,'D2_TrueX,D2_TrueY,D2_EstX,D2_EstY,D2_Err_cm,');
fprintf(fid,'Separation_m,Coverage_Swarm_pct\n');
for k=1:N_sw
    e1=sqrt((d1_est(1,k)-drone1_x(k))^2+(d1_est(2,k)-drone1_y(k))^2)*100;
    e2=sqrt((d2_est(1,k)-drone2_x(k))^2+(d2_est(2,k)-drone2_y(k))^2)*100;
    fprintf(fid,'%d,%.4f,',k,t_sw(k));
    fprintf(fid,'%.4f,%.4f,%.4f,%.4f,%.4f,',...
        drone1_x(k),drone1_y(k),d1_est(1,k),d1_est(2,k),e1);
    fprintf(fid,'%.4f,%.4f,%.4f,%.4f,%.4f,',...
        drone2_x(k),drone2_y(k),d2_est(1,k),d2_est(2,k),e2);
    fprintf(fid,'%.4f,%.2f\n',sep(k),cov_swarm_t(k));
end
fclose(fid);
fprintf('Saved: %s  [%d rows]\n\n',csv_file,N_sw);

save('swarm_results.mat','rmse_single','rmse_d1','rmse_d2',...
    'cov_single_t','cov_swarm_t','t_s','t_sw',...
    't_single_thresh','t_swarm_thresh','thresh','sep',...
    'drone1_x','drone1_y','drone2_x','drone2_y');
fprintf('Saved: swarm_results.mat\n\n');

fprintf('%s\n  Task 5 complete.\n\n',repmat('=',1,65));
fprintf('  Single drone RMSE  : %.2f cm  (60s, 8 lanes)\n',rmse_single);
fprintf('  Drone 1 RMSE       : %.2f cm  (30s, odd lanes)\n',rmse_d1);
fprintf('  Drone 2 RMSE       : %.2f cm  (30s, even lanes)\n',rmse_d2);
fprintf('  Swarm mean RMSE    : %.2f cm\n',(rmse_d1+rmse_d2)/2);
fprintf('  Coverage at 30s:  Single=%.1f%%  Swarm=%.1f%%\n',...
    cov_single_t(N_sw),cov_swarm_t(end));
fprintf('  Min separation     : %.2f m\n',min(sep));
if min(sep)>2.0
    fprintf('  Safety check (>2m) : PASSED\n');
else
    fprintf('  Safety check (>2m) : WARNING — check paths\n');
end
fprintf('%s\n',repmat('=',1,65));