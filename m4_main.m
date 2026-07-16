% M4_MAIN  Month 4, Pillar B: Learned NLOS Detection front-end.
% Run this top to bottom. Each section is a cell (Ctrl+Enter to run one).
%
% Requires (all in your full license): Deep Learning Toolbox, Statistics and
% Machine Learning Toolbox. Everything else is base MATLAB.

clear; clc; close all;
cfg = m4_config();
fprintf('Month 4 - Pillar B: Learned NLOS Detection\n');
fprintf('Mine %dx%d m, %d anchors, %d Hz, %d s missions\n', ...
    cfg.W, cfg.H, cfg.nAnchors, cfg.fs, cfg.T);

%% 1. Sanity check: visualise one mission's noise + NLOS profile
sim = m4_simulate_mission(cfg, cfg.seedOffset+1);
f=figure('Color','w','Position',[100 100 900 360]);
plot(sim.t, sim.rng_raw(:,5),'r.','MarkerSize',4); hold on;
plot(sim.t, sim.rng_true(:,5),'k','LineWidth',1.5);
xlabel('Time (s)'); ylabel('Range to A5 (m)');
legend('raw (noisy)','true','Location','best');
title('Sanity check: raw vs true range, ceiling anchor A5');
grid on;
exportgraphics(f,fullfile(cfg.figdir,'fig_sanity_ranges.png'),'Resolution',200);

%% 2. Train the detector  (~minutes on GPU, longer on CPU)
net = m4_train_detector(cfg);

%% 3. Evaluate: ROC, conpfusion, RMSE comparison, example mission
results = m4_evaluate(cfg, net);
m4_running_rmse(cfg, net);

%% 4. Print a compact results table for the report
fprintf('\n========== MONTH 4 PILLAR B SUMMARY ==========\n');
fprintf('Detector AUC        : %.3f\n', results.AUC);
fprintf('Detector F1         : %.3f\n', results.f1);
fprintf('RMSE Fixed-R   (cm) : %.2f\n', 100*mean(results.rmseFix));
fprintf('RMSE Heuristic (cm) : %.2f\n', 100*mean(results.rmseHeu));
fprintf('RMSE Learned   (cm) : %.2f\n', 100*mean(results.rmseLrn));
fprintf('RMSE Oracle    (cm) : %.2f\n', 100*mean(results.rmseOra));
gainVsHeu = 100*(mean(results.rmseHeu)-mean(results.rmseLrn))/mean(results.rmseHeu);
capture  = (mean(results.rmseFix)-mean(results.rmseLrn)) / ...
    max(mean(results.rmseFix)-mean(results.rmseOra),eps);
fprintf('Learned vs Heuristic: %.1f%% RMSE reduction\n', gainVsHeu);
fprintf('Oracle-gap captured : %.1f%%\n', 100*capture);
fprintf('Wilcoxon p          : %.4g\n', results.pWilcoxon);
fprintf('==============================================\n');

%% 5. Animation (run last - takes a minute to render the MP4)
m4_animate(cfg, net);

disp('Done. See ./figures for all outputs.');