function results = m4_evaluate_planner(cfg, net)
% M4_EVALUATE_PLANNER  Which routing OBJECTIVE improves accuracy over a fixed
% lawnmower, holding detector and filter fixed?
%   results = m4_evaluate_planner(cfg, net)
% Builds the NLOS risk map and the GDOP map, plans four paths, flies each over
% cfg.nPlanSeeds missions. Compares NLOS-objective vs GDOP-objective routing.
% Reports mean+median RMSE + exposure, Wilcoxon gdop_astar vs lawnmower, and a
% coverage-cost vs RMSE Pareto figure plus the risk/geometry map overlay.

strategies = {'lawnmower','nlos_only','gdop_astar','gdop_global'};
nS = numel(strategies);

[Rmap,Gmap,gx,gy] = m4_risk_map(cfg, net, cfg.seedOffset+900);

paths = cell(nS,1); plen = zeros(nS,1);
for s=1:nS
    paths{s} = m4_plan_path(cfg, Rmap, Gmap, gx, gy, strategies{s});
    plen(s)  = sum(sqrt(sum(diff(paths{s}).^2,2)));
end

testSeeds = cfg.seedOffset + 500 + (1:cfg.nPlanSeeds);
RMSE = zeros(cfg.nPlanSeeds, nS);
EXPO = zeros(cfg.nPlanSeeds, nS);
for si=1:numel(testSeeds)
    for s=1:nS
        o = m4_run_on_path(cfg, net, paths{s}, testSeeds(si), 'soft');
        RMSE(si,s)=o.rmse; EXPO(si,s)=o.exposure;
    end
end

fprintf('\n=== Month 4: routing-objective study (N=%d missions) ===\n', cfg.nPlanSeeds);
fprintf('%-12s  mean RMSE cm   median cm   NLOS expo   len (m)\n','strategy');
for s=1:nS
    fprintf('%-12s  %5.2f +/- %4.2f  %6.2f      %.3f       %.1f\n', strategies{s}, ...
        100*mean(RMSE(:,s)), 100*std(RMSE(:,s)), 100*median(RMSE(:,s)), ...
        mean(EXPO(:,s)), plen(s));
end

pW = signrank(RMSE(:,3), RMSE(:,1));          % gdop_astar vs lawnmower
gain     = 100*(mean(RMSE(:,1))-mean(RMSE(:,3)))/mean(RMSE(:,1));
gainMed  = 100*(median(RMSE(:,1))-median(RMSE(:,3)))/median(RMSE(:,1));
fprintf('gdop_astar vs lawnmower: %.1f%% mean / %.1f%% median RMSE reduction, p = %.4g\n', ...
    gain, gainMed, pW);

% ---- Pareto figure: path length vs median RMSE ------------------------
f=figure('Color','w','Position',[100 100 720 520]);
cols=[0.5 0.5 0.5; 0.85 0.5 0.2; 0.2 0.6 0.3; 0.2 0.3 0.7]; hold on;
for s=1:nS
    mu = 100*median(RMSE(:,s));
    lo = mu - 100*quantile(RMSE(:,s),0.25);
    hi = 100*quantile(RMSE(:,s),0.75) - mu;
    errorbar(plen(s), mu, lo, hi, ...
        'o','MarkerSize',10,'MarkerFaceColor',cols(s,:), ...
        'Color',cols(s,:),'LineWidth',1.4,'CapSize',8);
    text(plen(s)+0.3, mu, strategies{s}, 'Interpreter','none','FontWeight','bold');
end
xlabel('Path length (m)  ~  coverage time'); ylabel('3D Position RMSE (cm, median \pm IQR)');
title('Routing objective vs accuracy: NLOS vs GDOP'); grid on;
exportgraphics(f,fullfile(cfg.figdir,'fig_planner_pareto.png'),'Resolution',200);

% ---- GDOP map heatmap with overlaid paths -----------------------------
g=figure('Color','w','Position',[100 100 760 560]);
imagesc(gx, gy, Gmap); set(gca,'YDir','normal'); axis image; hold on;
colormap(parula); cb=colorbar; cb.Label.String='normalised GDOP (geometry quality)';
plot(paths{1}(:,1),paths{1}(:,2),'--','Color',[0.85 0.85 0.85],'LineWidth',1.6);
plot(paths{3}(:,1),paths{3}(:,2),'-','Color',[0.95 0.3 0.3],'LineWidth',2.0);
plot(cfg.anchors(:,1),cfg.anchors(:,2),'k^','MarkerFaceColor','w','MarkerSize',9);
legend({'lawnmower','GDOP-aware A*','anchors'},'Location','northoutside','Orientation','horizontal');
xlabel('X (m)'); ylabel('Y (m)'); title('GDOP map with planned paths');
exportgraphics(g,fullfile(cfg.figdir,'fig_gdop_map.png'),'Resolution',200);

% ---- NLOS risk map (kept for the paper's motivation figure) -----------
h=figure('Color','w','Position',[100 100 760 560]);
imagesc(gx, gy, Rmap); set(gca,'YDir','normal'); axis image; hold on;
colormap(hot); cb=colorbar; cb.Label.String='predicted P(NLOS)';
plot(paths{1}(:,1),paths{1}(:,2),'--','Color',[0.4 0.4 0.4],'LineWidth',1.6);
plot(paths{2}(:,1),paths{2}(:,2),'-','Color',[0.2 0.9 0.4],'LineWidth',2.0);
plot(cfg.anchors(:,1),cfg.anchors(:,2),'c^','MarkerFaceColor','c','MarkerSize',9);
legend({'lawnmower','NLOS-avoiding A*','anchors'},'Location','northoutside','Orientation','horizontal');
xlabel('X (m)'); ylabel('Y (m)'); title('Predicted NLOS risk map with planned paths');
exportgraphics(h,fullfile(cfg.figdir,'fig_risk_map.png'),'Resolution',200);

results=struct('strategies',{strategies},'RMSE',RMSE,'EXPO',EXPO, ...
    'plen',plen,'pWilcoxon',pW,'gain',gain,'gainMedian',gainMed, ...
    'Rmap',Rmap,'Gmap',Gmap);
save(fullfile(pwd,'m4_planner_results.mat'),'results');
fprintf('Saved m4_planner_results.mat and planner figures.\n');
end