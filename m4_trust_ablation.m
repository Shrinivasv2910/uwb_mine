% M4_TRUST_ABLATION  The paper's core experiment: on the FIXED lawnmower path,
% with the SAME detector, compare measurement-trust policies. Isolates the
% value of soft-trust (R-inflation) against hard-reject and naive baselines.
% No path planning here - path is held constant so trust is the only variable.

clear; clc; close all;
cfg = m4_config_plan();
S = load('nlos_detector.mat'); net = S.net;
fprintf('Month 4 - Trust-policy ablation (fixed lawnmower path)\n');

path = m4_trajectory(cfg);                 % the fixed Month 1-3 lawnmower
policies = {'fixed','hard','soft'};        % see m4_run_trust below
labels   = {'Fixed-R (trust all)','Hard-reject (P>=0.5)','Soft-trust (R-inflate)'};
nP = numel(policies);

testSeeds = cfg.seedOffset + 500 + (1:cfg.nPlanSeeds);
RMSE = zeros(cfg.nPlanSeeds, nP);
for si=1:numel(testSeeds)
    for p=1:nP
        o = m4_run_trust(cfg, net, path, testSeeds(si), policies{p});
        RMSE(si,p) = o.rmse;
    end
end

fprintf('\n=== Trust-policy RMSE (cm), N=%d missions, fixed path ===\n', cfg.nPlanSeeds);
for p=1:nP
    fprintf('%-22s : %5.2f +/- %4.2f  (median %5.2f)\n', labels{p}, ...
        100*mean(RMSE(:,p)), 100*std(RMSE(:,p)), 100*median(RMSE(:,p)));
end

% headline tests, both vs the hard-reject baseline (column 2)
pSoftHard = signrank(RMSE(:,3), RMSE(:,2));
gSoftHard = 100*(mean(RMSE(:,2))-mean(RMSE(:,3)))/mean(RMSE(:,2));
pSoftFix  = signrank(RMSE(:,3), RMSE(:,1));
gSoftFix  = 100*(mean(RMSE(:,1))-mean(RMSE(:,3)))/mean(RMSE(:,1));
fprintf('\nSoft-trust vs Hard-reject: %.1f%% RMSE reduction, p = %.4g\n', gSoftHard, pSoftHard);
fprintf('Soft-trust vs Fixed-R    : %.1f%% RMSE reduction, p = %.4g\n', gSoftFix,  pSoftFix);

% ---- box/bar figure ----
f=figure('Color','w','Position',[100 100 640 480]);
m=100*mean(RMSE); e=100*std(RMSE);
b=bar(m,'FaceColor','flat'); hold on;
b.CData=[0.5 0.5 0.5; 0.85 0.4 0.3; 0.2 0.6 0.3];
errorbar(1:nP,m,e,'k','LineStyle','none','LineWidth',1.2);
set(gca,'XTickLabel',{'Fixed-R','Hard-reject','Soft-trust'});
ylabel('3D Position RMSE (cm)'); grid on;
title('Measurement-trust ablation (fixed lawnmower path)');
for i=1:nP, text(i,m(i)+e(i)+0.5,sprintf('%.1f',m(i)),'HorizontalAlignment','center','FontWeight','bold'); end
exportgraphics(f,fullfile(cfg.figdir,'fig_trust_ablation.png'),'Resolution',200);

save(fullfile(pwd,'m4_trust_results.mat'),'RMSE','policies','labels');
fprintf('Saved m4_trust_results.mat and fig_trust_ablation.png\n');