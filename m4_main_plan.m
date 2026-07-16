% M4_MAIN_PLAN  Month 4 Pillar A: NLOS-aware coverage path optimisation.
% New vs Months 1-3: the learned detector becomes a MAPPER (m4_risk_map),
% whose risk field drives an intelligent coverage planner (m4_plan_path) that
% routes around high-NLOS regions. m4_evaluate_planner reports the
% coverage-cost vs RMSE Pareto. Soft R-inflation replaces hard reject.
% Requires the trained detector nlos_detector.mat. Run top to bottom.

clear; clc; close all;
cfg = m4_config_plan();
fprintf('Month 4 - NLOS-aware coverage planning\n');
fprintf('Mine %dx%d m, %d anchors, %d s missions\n', cfg.W, cfg.H, cfg.nAnchors, cfg.T);

%% 1. Load (or train) the NLOS detector
if exist('nlos_detector.mat','file')
    S = load('nlos_detector.mat'); net = S.net;
    fprintf('Loaded trained detector from nlos_detector.mat\n');
else
    fprintf('No detector found - training (Pillar B)...\n');
    net = m4_train_detector(cfg);
end

%% 2. Build the predicted NLOS risk map (the enabling artifact)
[Rmap,Gmap,gx,gy] = m4_risk_map(cfg, net, cfg.seedOffset+900);
fprintf('Risk map: mean P(NLOS) = %.3f, peak = %.3f\n', mean(Rmap(:)), max(Rmap(:)));

%% 3. Headline study
results = m4_evaluate_planner(cfg, net);

%% 4. Compact summary
fprintf('\n========== MONTH 4 PLANNER SUMMARY ==========\n');
for s=1:numel(results.strategies)
    fprintf('%-12s : %5.2f cm   exposure %.3f\n', results.strategies{s}, ...
        100*mean(results.RMSE(:,s)), mean(results.EXPO(:,s)));
end
fprintf('Risk-aware gain vs lawnmower : %.1f%%\n', results.gain);
fprintf('Wilcoxon p                   : %.4g\n', results.pWilcoxon);
fprintf('=============================================\n');
disp('Done. See ./figures for the risk map and Pareto plots.');