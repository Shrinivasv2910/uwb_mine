clear; clc; close all;

%% =========================================================
%  uwb_ddpg_deploy.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 1: Deploy and Evaluate Trained DDPG Agent
%
%  Loads ddpg_trained.mat, runs full 60s simulation,
%  Compares three methods on identical raw UWB data (rng=7):
%    1. Raw UWB          — no filtering
%    2. Fixed Kalman     — constant Q and R (Month 1 baseline)
%    3. DDPG Kalman      — agent-tuned Q and R (Month 2)
%
%  Outputs:
%    5 figures exported as PNG (300 dpi)
%    uwb_ddpg_results.csv  (601 rows x 16 columns)
%
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

%% LOAD TRAINED AGENT
if ~exist('ddpg_trained.mat','file')
    error('ddpg_trained.mat not found. Run uwb_ddpg_train.m first.');
end
load('ddpg_trained.mat');
fprintf('Agent loaded  (best ep: %d  |  train RMSE: %.4f cm)\n\n',...
    best_ep, best_rmse);

%% =========================================================
%  SIMULATION SETUP
%  Same seed (rng=7) as Month 1 for direct comparison
%% =========================================================
T  = 60;
t  = 0:dt:T;
N  = length(t);

%% BOUSTROPHEDON PATH — 8 lanes
lanes  = 8;
ly     = linspace(1, H-1, lanes);
px     = []; py = [];
for L  = 1:lanes
    if mod(L,2)==1, px=[px 1 W-1]; else, px=[px W-1 1]; end
    py = [py ly(L) ly(L)];
end
dc        = [0 cumsum(sqrt(diff(px).^2 + diff(py).^2))];
dq        = linspace(0, dc(end), N);
true_x    = interp1(dc, px, dq);
true_y    = interp1(dc, py, dq);
true_range = sqrt(true_x.^2 + true_y.^2);

%% RAW UWB MEASUREMENTS (same seed as Month 1)
rng(7);
raw       = max(true_range + 0.25*randn(1,N) + ...
                (rand(1,N)<0.12).*(0.4+0.8*rand(1,N)), 0);
nlos_mask = (raw - true_range) > 0.25;

fprintf('Mission: %.0f s  |  %d samples  |  %d NLOS events (%.1f%%)\n\n',...
    T, N, sum(nlos_mask), 100*mean(nlos_mask));

%% =========================================================
%  HELPER FUNCTIONS
%% =========================================================
function a = act_fwd(n, s)
    a = tanh(n.W3 * max(0, n.W2 * max(0, n.W1*s + n.b1) + n.b2) + n.b3);
end
function [qp, qv, rv] = decode_action(a, qpn,qpx, qvn,qvx, rn,rx)
    qp = max(qpn, min(qpx,  qpn + (a(1)+1)/2 * (qpx-qpn)));
    qv = max(qvn, min(qvx,  qvn + (a(2)+1)/2 * (qvx-qvn)));
    rv = max(rn,  min(rx,   rn  + (a(3)+1)/2 * (rx -rn )));
end
function s = build_state(r3, e3, vel, nlos, W, H)
    s = [r3/sqrt(W^2+H^2); e3/100; vel/10; double(nlos)];
end

%% =========================================================
%  METHOD 1: FIXED KALMAN (Month 1 baseline)
%  Q and R are constant throughout the mission
%% =========================================================
Q_fix = diag([0.0005 0.005]);
R_fix = 0.25^2;
x_fix = [raw(1); 0];
P_fix = eye(2);
kf_fix = zeros(1, N);

for k = 1:N
    x_fix = F_k*x_fix;
    P_fix = F_k*P_fix*F_k' + Q_fix;
    Kf    = P_fix*H_k' / (H_k*P_fix*H_k' + R_fix);
    x_fix = x_fix + Kf*(raw(k) - H_k*x_fix);
    P_fix = (eye(2) - Kf*H_k)*P_fix;
    kf_fix(k) = max(0, x_fix(1));
end

%% =========================================================
%  METHOD 2: DDPG KALMAN (Month 2)
%  Q and R tuned at every timestep by trained actor network
%% =========================================================
fprintf('Running DDPG agent...\n');
D_MAX    = sqrt(W^2 + H^2);
kf_ddpg  = zeros(1, N);
qp_hist  = zeros(1, N);
qv_hist  = zeros(1, N);
rv_hist  = zeros(1, N);

x_rl     = [raw(1); 0];
P_rl     = eye(2);
raw_hist = raw(1)*ones(3,1);
err_hist = zeros(3,1);
state    = build_state(raw_hist, err_hist, 0, 0, W, H);

for k = 1:N
    % Actor inference
    a_out = act_fwd(actor, state);
    a_out(isnan(a_out)|isinf(a_out)) = 0;

    [qp, qv, rv] = decode_action(a_out,...
        Q_pos_min, Q_pos_max,...
        Q_vel_min, Q_vel_max,...
        R_min, R_max);
    qp_hist(k) = qp;
    qv_hist(k) = qv;
    rv_hist(k) = rv;

    % Kalman step with agent-selected Q and R
    Q_rl  = diag([qp qv]);
    R_rl  = rv^2;
    x_rl  = F_k*x_rl;
    P_rl  = F_k*P_rl*F_k' + Q_rl;
    K_rl  = P_rl*H_k' / (H_k*P_rl*H_k' + R_rl);
    x_rl  = x_rl + K_rl*(raw(k) - H_k*x_rl);
    P_rl  = (eye(2) - K_rl*H_k)*P_rl;

    kf_ddpg(k) = max(0, min(D_MAX, x_rl(1)));
    if isnan(kf_ddpg(k)) || isinf(kf_ddpg(k))
        x_rl = [raw(k); 0]; P_rl = eye(2)*0.5; kf_ddpg(k) = raw(k);
    end

    % Build next state
    vel_est  = 0;
    if k > 1, vel_est = (raw(k) - raw(k-1))*fs; end
    err_now  = (kf_ddpg(k) - true_range(k))*100;
    raw_hist = [raw_hist(2:end); raw(k)];
    err_hist = [err_hist(2:end); err_now];
    state    = build_state(raw_hist, err_hist, vel_est,...
                           double(nlos_mask(k)), W, H);
end
fprintf('DDPG agent done.\n\n');

%% =========================================================
%  ERRORS AND RUNNING RMSE
%% =========================================================
err_raw  = (raw      - true_range)*100;
err_fix  = (kf_fix   - true_range)*100;
err_ddpg = (kf_ddpg  - true_range)*100;

rmse_fix  = zeros(1,N);
rmse_ddpg = zeros(1,N);
af = 0; ad = 0;
for k = 1:N
    af = af + err_fix(k)^2;   rmse_fix(k)  = sqrt(af/k);
    ad = ad + err_ddpg(k)^2;  rmse_ddpg(k) = sqrt(ad/k);
end

rmse_raw_val = sqrt(mean(err_raw.^2));

%% PRINT RESULTS
fprintf('%s\n', repmat('=',1,55));
fprintf('  PERFORMANCE SUMMARY  (60s mission, rng seed = 7)\n');
fprintf('%s\n', repmat('=',1,55));
fprintf('  %-20s  %10s  %15s\n','Method','RMSE (cm)','vs Fixed KF');
fprintf('  %-20s  %10.4f  %15s\n','Raw UWB',   rmse_raw_val,  'Baseline (raw)');
fprintf('  %-20s  %10.4f  %15s\n','Fixed KF',  rmse_fix(N),   'Baseline');
fprintf('  %-20s  %10.4f  %12.2f%%\n','DDPG KF', rmse_ddpg(N),...
    (1 - rmse_ddpg(N)/rmse_fix(N))*100);
fprintf('%s\n', repmat('=',1,55));
fprintf('\n  Q_pos:  mean = %.6f  |  std = %.6f\n',...
    mean(qp_hist), std(qp_hist));
fprintf('  R    :  mean = %.6f  |  std = %.6f\n',...
    mean(rv_hist), std(rv_hist));
fprintf('%s\n\n', repmat('=',1,55));

%% =========================================================
%  COLOUR PALETTE
%% =========================================================
Ctr   = [0.30 0.30 0.30];   % ground truth — dark grey
Craw  = [0.92 0.22 0.08];   % raw UWB      — red
Cfix  = [0.50 0.10 0.80];   % fixed KF     — purple
Cddpg = [0.05 0.72 0.32];   % DDPG KF      — green
Cq    = [0.92 0.55 0.05];   % Q_pos        — orange
Cr    = [0.08 0.45 0.88];   % R            — blue
GS    = [0.97 0.97 0.97];   % grid/bg grey
RES   = 300;                  % export resolution (dpi)

%% =========================================================
%  FIGURE 1 — TRAJECTORY
%% =========================================================
figure('Color','w','Position',[50 50 600 560],'NumberTitle','off');
axes('Position',[0.12 0.10 0.84 0.84]); hold on;

rectangle('Position',[0 0 W H],'EdgeColor',[0.70 0.70 0.70],...
    'LineWidth',1.0,'LineStyle','--');

% Planned path (yellow)
plot(true_x, true_y, 'Color',[0.95 0.82 0.05],'LineWidth',1.2,...
    'DisplayName','Planned Path');

% Anchors
anc = [0 0; W 0; W H; 0 H];
off = [0.6 0.8; W-2.5 0.8; W-2.5 H-1.6; 0.6 H-1.6];
for a = 1:4
    scatter(anc(a,1), anc(a,2), 120, 'r','s','filled');
    text(off(a,1), off(a,2), sprintf('A%d',a),...
        'FontSize',9,'FontWeight','bold','Color',[0.70 0.05 0.05]);
end

% Pink dashed lines from anchors to path centroid
cx = mean(true_x); cy = mean(true_y);
for a = 1:4
    plot([anc(a,1) cx],[anc(a,2) cy],'Color',[0.95 0.35 0.65],...
        'LineWidth',0.8,'LineStyle','--','HandleVisibility','off');
end

% Start / End markers
scatter(true_x(1),   true_y(1),   90,[0.10 0.60 0.10],'^','filled');
scatter(true_x(end), true_y(end), 90,[0.10 0.10 0.80],'o','filled');
text(true_x(1)+0.5,   true_y(1)+0.5,   'Start','FontSize',8,'Color',[0.10 0.60 0.10]);
text(true_x(end)+0.5, true_y(end)+0.5, 'End',  'FontSize',8,'Color',[0.10 0.10 0.80]);

xlim([-2 W+3]); ylim([-2 H+3]); axis equal;
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on');
xlabel('X (m)'); ylabel('Y (m)');
title('Boustrophedon Scan Path with UWB Anchor Layout',...
    'FontSize',10,'FontWeight','bold');

exportgraphics(gcf,'fig_trajectory.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 1 saved: fig_trajectory.png\n'); close;

%% =========================================================
%  FIGURE 2 — RANGE COMPARISON (full 60s)
%% =========================================================
figure('Color','w','Position',[50 50 1000 360],'NumberTitle','off');
axes('Position',[0.07 0.13 0.90 0.79]); hold on;

plot(t, true_range,'Color',Ctr,   'LineWidth',1.6,'DisplayName','Ground Truth');
plot(t, raw,        'Color',Craw,  'LineWidth',0.6,'DisplayName','Raw UWB');
plot(t, kf_fix,     'Color',Cfix,  'LineWidth',1.1,'DisplayName','Fixed Kalman');
plot(t, kf_ddpg,    'Color',Cddpg, 'LineWidth',1.5,'DisplayName','DDPG Kalman');

grid on;
set(gca,'Color',GS,'GridAlpha',0.10,'Box','on',...
    'XTick',0:5:60,'YTick',0:2:22);
xlabel('Time (s)'); ylabel('Range (m)');
title('UWB Range: Ground Truth vs Raw vs Fixed Kalman vs DDPG Kalman',...
    'FontSize',10,'FontWeight','bold');
legend('Location','best','FontSize',9,'Box','off');
xlim([0 60]); ylim([0 22]);

exportgraphics(gcf,'fig_range.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 2 saved: fig_range.png\n'); close;

%% =========================================================
%  FIGURE 3 — RANGING ERROR (0 to 30s with NLOS markers)
%% =========================================================
figure('Color','w','Position',[50 50 1000 380],'NumberTitle','off');
axes('Position',[0.07 0.13 0.90 0.79]); hold on;

idx = t <= 30;

% NLOS vertical lines
nl_idx = find(nlos_mask(idx));
for i = 1:length(nl_idx)
    xline(t(nl_idx(i)),'Color',[0.92 0.55 0.05],...
        'LineWidth',0.5,'Alpha',0.35,'HandleVisibility','off');
end

plot(t(idx), err_raw(idx),  'Color',Craw,  'LineWidth',0.6,'DisplayName','Raw UWB');
plot(t(idx), err_fix(idx),  'Color',Cfix,  'LineWidth',1.1,'DisplayName','Fixed Kalman');
plot(t(idx), err_ddpg(idx), 'Color',Cddpg, 'LineWidth',1.5,'DisplayName','DDPG Kalman');
yline(0,'--','Color',[0.50 0.50 0.50],'LineWidth',0.7,'HandleVisibility','off');

% Dummy scatter for NLOS legend entry
scatter(-999, 0, 20,[0.92 0.55 0.05],'v','filled',...
    'DisplayName',sprintf('NLOS events (n=%d)',sum(nlos_mask(idx))));

grid on;
set(gca,'Color',GS,'GridAlpha',0.10,'Box','on',...
    'XTick',0:5:30,'YTick',-30:10:90);
xlabel('Time (s)'); ylabel('Error (cm)');
title('Ranging Error Comparison with NLOS Events Highlighted  (0–30 s)',...
    'FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',9,'Box','off');
xlim([0 30]); ylim([-30 90]);

exportgraphics(gcf,'fig_error.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 3 saved: fig_error.png\n'); close;

%% =========================================================
%  FIGURE 4 — RMSE BAR CHART
%% =========================================================
figure('Color','w','Position',[50 50 520 440],'NumberTitle','off');
axes('Position',[0.16 0.15 0.80 0.70]); hold on;

vals = [rmse_raw_val  rmse_fix(N)  rmse_ddpg(N)];
bh   = bar(vals,'FaceColor','flat','EdgeColor','none','BarWidth',0.50);
bh.CData = [Craw; Cfix; Cddpg];

set(gca,'XTick',1:3,...
    'XTickLabel',{'Raw UWB','Fixed Kalman','DDPG Kalman'},...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:55);
grid on; ylabel('RMSE (cm)');
title('RMSE Comparison','FontSize',10,'FontWeight','bold');
ylim([0 max(vals)*1.45]);

for i = 1:3
    text(i, vals(i)+0.8, sprintf('%.2f cm', vals(i)),...
        'HorizontalAlignment','center',...
        'FontSize',9.5,'FontWeight','bold','Color',[0.15 0.15 0.15]);
end
text(2.5, max(vals)*1.30,...
    sprintf('\\downarrow %.1f%% improvement',...
    (1-rmse_ddpg(N)/rmse_fix(N))*100),...
    'HorizontalAlignment','center','FontSize',9,'Color',[0.20 0.20 0.20]);

exportgraphics(gcf,'fig_rmse_bar.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 4 saved: fig_rmse_bar.png\n'); close;

%% =========================================================
%  FIGURE 5 — Q_pos AND R ADAPTATION OVER TIME
%% =========================================================
figure('Color','w','Position',[50 50 1000 420],'NumberTitle','off');

subplot(2,1,1); hold on;
plot(t, qp_hist,'Color',Cq,'LineWidth',1.0,'DisplayName','Q_{pos}');
scatter(t(nlos_mask), qp_hist(nlos_mask), 10,[0.75 0.10 0.10],...
    'o','filled','MarkerFaceAlpha',0.5,'DisplayName','NLOS event');
set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
ylabel('Q_{pos}'); xlim([0 60]); grid on;
legend('Location','northeast','FontSize',8,'Box','off');
title('DDPG Agent: Adaptive Kalman Parameters Over Time',...
    'FontSize',10,'FontWeight','bold');

subplot(2,1,2); hold on;
plot(t, rv_hist,'Color',Cr,'LineWidth',1.0,'DisplayName','R');
scatter(t(nlos_mask), rv_hist(nlos_mask), 10,[0.75 0.10 0.10],...
    'o','filled','MarkerFaceAlpha',0.5,'DisplayName','NLOS event');
set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
ylabel('R'); xlabel('Time (s)'); xlim([0 60]); grid on;
legend('Location','northeast','FontSize',8,'Box','off');

exportgraphics(gcf,'fig_qr_adaptation.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 5 saved: fig_qr_adaptation.png\n'); close;

%% =========================================================
%  FIGURE 6 — RUNNING RMSE
%% =========================================================
figure('Color','w','Position',[50 50 1000 360],'NumberTitle','off');
axes('Position',[0.07 0.13 0.88 0.79]); hold on;

% Shaded improvement band
patch([t fliplr(t)],[rmse_fix fliplr(rmse_ddpg)],...
    [0.80 0.90 0.80],'FaceAlpha',0.20,'EdgeColor','none');

plot(t, rmse_fix,  'Color',Cfix,  'LineWidth',1.3,...
    'DisplayName',sprintf('Fixed Kalman  (%.2f cm)',  rmse_fix(N)));
plot(t, rmse_ddpg, 'Color',Cddpg, 'LineWidth',1.6,...
    'DisplayName',sprintf('DDPG Kalman  (%.2f cm)', rmse_ddpg(N)));

yline(rmse_fix(N),  '--','Color',Cfix,  'LineWidth',0.7,...
    'Alpha',0.6,'HandleVisibility','off');
yline(rmse_ddpg(N), '--','Color',Cddpg, 'LineWidth',0.7,...
    'Alpha',0.6,'HandleVisibility','off');

text(61.5, rmse_fix(N),  sprintf('%.2f cm',rmse_fix(N)),...
    'FontSize',8.5,'Color',Cfix, 'FontWeight','bold',...
    'VerticalAlignment','middle');
text(61.5, rmse_ddpg(N), sprintf('%.2f cm',rmse_ddpg(N)),...
    'FontSize',8.5,'Color',Cddpg,'FontWeight','bold',...
    'VerticalAlignment','middle');

grid on;
set(gca,'Color',GS,'GridAlpha',0.10,'Box','on',...
    'XTick',0:5:60,'YTick',0:2:50);
xlabel('Time (s)'); ylabel('Running RMSE (cm)');
title(sprintf('Running RMSE: Fixed Kalman vs DDPG Kalman  (%.1f%% Improvement)',...
    (1-rmse_ddpg(N)/rmse_fix(N))*100),...
    'FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',9,'Box','off');
xlim([0 64]);

exportgraphics(gcf,'fig_running_rmse.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 6 saved: fig_running_rmse.png\n'); close;

%% =========================================================
%  FIGURE 7 — NLOS EVENT ANALYSIS + ERROR DISTRIBUTION
%% =========================================================
figure('Color','w','Position',[50 50 1020 400],'NumberTitle','off');

% Left: NLOS median error bar chart
nlos_err_raw  = abs(err_raw( nlos_mask));
nlos_err_fix  = abs(err_fix( nlos_mask));
nlos_err_ddpg = abs(err_ddpg(nlos_mask));
mv = [median(nlos_err_raw) median(nlos_err_fix) median(nlos_err_ddpg)];

subplot(1,2,1);
bh2 = bar(mv,'FaceColor','flat','EdgeColor','none','BarWidth',0.50);
bh2.CData = [Craw; Cfix; Cddpg];
set(gca,'XTick',1:3,'XTickLabel',{'Raw','Fixed KF','DDPG KF'},...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:55);
grid on; ylabel('Median |Error| at NLOS Events (cm)');
title('NLOS Event Error','FontSize',10,'FontWeight','bold');
ylim([0 max(mv)*1.42]);
for i = 1:3
    text(i, mv(i)+0.8, sprintf('%.1f cm',mv(i)),...
        'HorizontalAlignment','center','FontSize',9.5,...
        'FontWeight','bold','Color',[0.15 0.15 0.15]);
end

% Right: Error probability distribution
subplot(1,2,2); hold on;
edges = -50:4:110; ctr = edges(1:end-1) + 2;
hr  = histcounts(err_raw,  edges,'Normalization','probability');
hf  = histcounts(err_fix,  edges,'Normalization','probability');
hd  = histcounts(err_ddpg, edges,'Normalization','probability');
plot(ctr, hr, 'Color',Craw,  'LineWidth',1.0,'DisplayName','Raw UWB');
plot(ctr, hf, 'Color',Cfix,  'LineWidth',1.2,'DisplayName','Fixed KF');
plot(ctr, hd, 'Color',Cddpg, 'LineWidth',1.5,'DisplayName','DDPG KF');
xline(0,'--','Color',[0.45 0.45 0.45],'LineWidth',0.8,...
    'HandleVisibility','off');
grid on;
set(gca,'Color',GS,'GridAlpha',0.12,'Box','on',...
    'XTick',-50:25:100,'YTick',0:0.02:0.20);
xlabel('Error (cm)'); ylabel('Probability');
title('Error Distribution','FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',8,'Box','off');
xlim([-50 110]); ylim([0 0.20]);

sgtitle('NLOS Event Analysis and Error Distribution',...
    'FontSize',11,'FontWeight','bold');

exportgraphics(gcf,'fig_nlos_analysis.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 7 saved: fig_nlos_analysis.png\n'); close;

%% =========================================================
%  EXPORT CSV
%  601 rows x 16 columns — same format as Month 1 for consistency
%% =========================================================
fprintf('\nExporting results to CSV...\n');
csv_file = 'uwb_ddpg_results.csv';
fid = fopen(csv_file,'w');
fprintf(fid,'Sample,Time_s,TrueRange_m,RawUWB_m,FixedKF_m,DDPG_KF_m,');
fprintf(fid,'Qpos,Qvel,R,');
fprintf(fid,'Error_Raw_cm,Error_FixKF_cm,Error_DDPG_cm,');
fprintf(fid,'RMSE_FixKF_cm,RMSE_DDPG_cm,NLOS_Flag,TrueX_m,TrueY_m\n');

for k = 1:N
    fprintf(fid,'%d,%.4f,%.6f,%.6f,%.6f,%.6f,',...
        k,t(k),true_range(k),raw(k),kf_fix(k),kf_ddpg(k));
    fprintf(fid,'%.8f,%.8f,%.8f,',...
        qp_hist(k),qv_hist(k),rv_hist(k));
    fprintf(fid,'%.4f,%.4f,%.4f,',...
        err_raw(k),err_fix(k),err_ddpg(k));
    fprintf(fid,'%.4f,%.4f,%d,%.4f,%.4f\n',...
        rmse_fix(k),rmse_ddpg(k),double(nlos_mask(k)),...
        true_x(k),true_y(k));
end
fclose(fid);
fprintf('Saved: %s  [%d rows x 17 columns]\n\n', csv_file, N);

fprintf('%s\n', repmat('=',1,55));
fprintf('  Task 1 complete.\n');
fprintf('  Fixed KF RMSE : %.4f cm\n', rmse_fix(N));
fprintf('  DDPG KF RMSE  : %.4f cm\n', rmse_ddpg(N));
fprintf('  Improvement   : %.2f%%\n',...
    (1-rmse_ddpg(N)/rmse_fix(N))*100);
fprintf('%s\n', repmat('=',1,55));