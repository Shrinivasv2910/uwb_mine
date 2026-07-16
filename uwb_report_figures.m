clear; clc; close all;

T  = readtable('uwb_ddpg_results.csv');
t  = T.Time_s;
W  = 20; H = 15;

Ctr=[0.30 0.30 0.30]; Craw=[0.90 0.20 0.08];
Cfix=[0.42 0.08 0.75]; Crl=[0.05 0.62 0.28];
Cq=[0.90 0.52 0.05]; Cr=[0.08 0.42 0.85];
Cpnk=[0.95 0.35 0.65]; GS=[0.96 0.96 0.96];
R=300;

rmse_raw=sqrt(mean(T.Error_Raw_cm.^2));
rmse_fix=T.RMSE_FixKF_cm(end);
rmse_ddpg=T.RMSE_DDPG_cm(end);
anc=[0 0;W 0;W H;0 H];
nl2=T.NLOS_Flag==1;

%% FIG 2 — TRAJECTORY
figure('Color','w','Position',[50 50 580 520],'NumberTitle','off');
axes('Position',[0.12 0.10 0.84 0.83]); hold on;
rectangle('Position',[0 0 W H],'EdgeColor',[0.60 0.60 0.60],'LineWidth',1.0,'LineStyle','--');
plot(T.TrueX_m,T.TrueY_m,'Color',Crl,'LineWidth',0.9);
cx=mean(T.TrueX_m); cy=mean(T.TrueY_m);
for a=1:4
    plot([anc(a,1) cx],[anc(a,2) cy],'Color',Cpnk,'LineWidth',0.8,'LineStyle','--');
end
for a=1:4
    scatter(anc(a,1),anc(a,2),120,'r','s','filled');
end
off=[0.6 0.8; W-2.5 0.8; W-2.5 H-1.6; 0.6 H-1.6];
for a=1:4
    text(off(a,1),off(a,2),sprintf('A%d',a),'FontSize',9,'FontWeight','bold',...
        'Color',[0.70 0.05 0.05]);
end
scatter(T.TrueX_m(1),T.TrueY_m(1),80,[0.1 0.6 0.1],'^','filled');
scatter(T.TrueX_m(end),T.TrueY_m(end),80,[0.1 0.1 0.8],'o','filled');
text(T.TrueX_m(1)+0.5,T.TrueY_m(1)+0.5,'Start','FontSize',8,'Color',[0.1 0.6 0.1]);
text(T.TrueX_m(end)+0.5,T.TrueY_m(end)+0.5,'End','FontSize',8,'Color',[0.1 0.1 0.8]);
xlim([-2 W+3]); ylim([-2 H+3]); axis equal;
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on');
xlabel('X (m)'); ylabel('Y (m)');
title('Boustrophedon Scan Path with UWB Anchor Layout','FontSize',10,'FontWeight','bold');
exportgraphics(gcf,'fig2_trajectory.png','Resolution',R,'BackgroundColor','white');
fprintf('Fig 2 saved\n'); close;

%% FIG 3 — RANGE
figure('Color','w','Position',[50 50 1000 360],'NumberTitle','off');
axes('Position',[0.07 0.13 0.90 0.79]); hold on;
plot(t,T.TrueRange_m,'Color',Ctr,'LineWidth',1.6,'DisplayName','Ground Truth');
plot(t,T.RawUWB_m,'Color',Craw,'LineWidth',0.6,'DisplayName','Raw UWB');
plot(t,T.FixedKF_m,'Color',Cfix,'LineWidth',1.1,'DisplayName','Fixed Kalman');
plot(t,T.DDPG_KF_m,'Color',Crl,'LineWidth',1.5,'DisplayName','DDPG Kalman');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60,'YTick',0:2:22);
xlabel('Time (s)'); ylabel('Range (m)');
title('UWB Range: Ground Truth vs Raw vs Fixed Kalman vs DDPG Kalman','FontSize',10,'FontWeight','bold');
legend('Location','best','FontSize',9,'Box','off');
xlim([0 60]); ylim([0 22]);
exportgraphics(gcf,'fig3_range.png','Resolution',R,'BackgroundColor','white');
fprintf('Fig 3 saved\n'); close;

%% FIG 4 — ERROR
figure('Color','w','Position',[50 50 1000 380],'NumberTitle','off');
axes('Position',[0.07 0.13 0.90 0.79]); hold on;
idx=t<=30;
plot(t(idx),T.Error_Raw_cm(idx),'Color',Craw,'LineWidth',0.6,'DisplayName','Raw UWB');
plot(t(idx),T.Error_FixKF_cm(idx),'Color',Cfix,'LineWidth',1.1,'DisplayName','Fixed Kalman');
plot(t(idx),T.Error_DDPG_cm(idx),'Color',Crl,'LineWidth',1.5,'DisplayName','DDPG Kalman');
yline(0,'Color',[0.50 0.50 0.50],'LineWidth',0.7,'LineStyle','--','HandleVisibility','off');
nl=find(T.NLOS_Flag(idx)==1);
for i=1:length(nl)
    xline(t(nl(i)),'Color',[0.92 0.55 0.05],'LineWidth',0.5,'Alpha',0.35,'HandleVisibility','off');
end
scatter(-999,0,20,[0.92 0.55 0.05],'v','filled','DisplayName',sprintf('NLOS (n=%d)',length(nl)));
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:30,'YTick',-30:10:90);
xlabel('Time (s)'); ylabel('Error (cm)');
title('Ranging Error Comparison with NLOS Events Highlighted  (0–30 s)','FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',9,'Box','off');
xlim([0 30]); ylim([-30 90]);
exportgraphics(gcf,'fig4_error.png','Resolution',R,'BackgroundColor','white');
fprintf('Fig 4 saved\n'); close;

%% FIG 5 — RMSE BAR
figure('Color','w','Position',[50 50 520 440],'NumberTitle','off');
axes('Position',[0.16 0.15 0.80 0.70]); hold on;
vals=[rmse_raw rmse_fix rmse_ddpg];
bh=bar(vals,'FaceColor','flat','EdgeColor','none','BarWidth',0.48);
bh.CData=[Craw;Cfix;Crl];
set(gca,'XTick',1:3,'XTickLabel',{'Raw UWB','Fixed Kalman','DDPG Kalman'},...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:55);
grid on; ylabel('RMSE (cm)');
title('RMSE Comparison','FontSize',10,'FontWeight','bold');
ylim([0 max(vals)*1.42]);
for i=1:3
    text(i,vals(i)+1.0,sprintf('%.2f',vals(i)),'HorizontalAlignment','center',...
        'FontSize',9.5,'FontWeight','bold','Color',[0.15 0.15 0.15]);
end
text(2.5,max(vals)*1.30,sprintf('\\downarrow %.1f%% improvement',27.82),...
    'HorizontalAlignment','center','FontSize',9,'Color',[0.20 0.20 0.20]);
exportgraphics(gcf,'fig5_rmse_bar.png','Resolution',R,'BackgroundColor','white');
fprintf('Fig 5 saved\n'); close;

%% FIG 6 — Q & R
figure('Color','w','Position',[50 50 1000 400],'NumberTitle','off');
subplot(2,1,1); hold on;
plot(t,T.Qpos,'Color',Cq,'LineWidth',1.1,'DisplayName','Q_{pos}');
scatter(t(nl2),T.Qpos(nl2),10,[0.75 0.10 0.10],'o','filled',...
    'MarkerFaceAlpha',0.5,'DisplayName','NLOS event');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
ylabel('Q_{pos}');
title('DDPG Agent: Adaptive Kalman Parameters Over Time','FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',8,'Box','off'); xlim([0 60]);

subplot(2,1,2); hold on;
plot(t,T.R,'Color',Cr,'LineWidth',1.1,'DisplayName','R');
scatter(t(nl2),T.R(nl2),10,[0.75 0.10 0.10],'o','filled',...
    'MarkerFaceAlpha',0.5,'DisplayName','NLOS event');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60);
ylabel('R'); xlabel('Time (s)');
legend('Location','northeast','FontSize',8,'Box','off'); xlim([0 60]);
exportgraphics(gcf,'fig6_qr_adaptation.png','Resolution',R,'BackgroundColor','white');
fprintf('Fig 6 saved\n'); close;

%% FIG 7 — RUNNING RMSE
figure('Color','w','Position',[50 50 1000 360],'NumberTitle','off');
axes('Position',[0.07 0.13 0.88 0.79]); hold on;
patch([t;flipud(t)],[T.RMSE_FixKF_cm;flipud(T.RMSE_DDPG_cm)],...
    [0.82 0.82 0.82],'FaceAlpha',0.20,'EdgeColor','none');
plot(t,T.RMSE_FixKF_cm,'Color',Cfix,'LineWidth',1.2,'DisplayName','Fixed Kalman');
plot(t,T.RMSE_DDPG_cm,'Color',Crl,'LineWidth',1.5,'DisplayName','DDPG Kalman');
yline(rmse_fix,'--','Color',Cfix,'LineWidth',0.7,'Alpha',0.6,'HandleVisibility','off');
yline(rmse_ddpg,'--','Color',Crl,'LineWidth',0.7,'Alpha',0.6,'HandleVisibility','off');
text(61,rmse_fix,sprintf('%.2f cm',rmse_fix),'FontSize',8.5,'Color',Cfix,...
    'FontWeight','bold','VerticalAlignment','middle');
text(61,rmse_ddpg,sprintf('%.2f cm',rmse_ddpg),'FontSize',8.5,'Color',Crl,...
    'FontWeight','bold','VerticalAlignment','middle');
grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','XTick',0:5:60,'YTick',0:2:46);
xlabel('Time (s)'); ylabel('Running RMSE (cm)');
title('Running RMSE: Fixed Kalman vs DDPG Adaptive Kalman  (27.82% Improvement)',...
    'FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',9,'Box','off');
xlim([0 63]); ylim([0 max(T.RMSE_FixKF_cm)*1.12]);
exportgraphics(gcf,'fig7_running_rmse.png','Resolution',R,'BackgroundColor','white');
fprintf('Fig 7 saved\n'); close;

%% FIG 8 — NLOS + DISTRIBUTION
figure('Color','w','Position',[50 50 1020 400],'NumberTitle','off');
nlos_raw =abs(T.Error_Raw_cm(T.NLOS_Flag==1));
nlos_fix =abs(T.Error_FixKF_cm(T.NLOS_Flag==1));
nlos_ddpg=abs(T.Error_DDPG_cm(T.NLOS_Flag==1));
mv=[median(nlos_raw) median(nlos_fix) median(nlos_ddpg)];

subplot(1,2,1);
bh2=bar(mv,'FaceColor','flat','EdgeColor','none','BarWidth',0.50);
bh2.CData=[Craw;Cfix;Crl];
set(gca,'XTick',1:3,'XTickLabel',{'Raw','Fixed KF','DDPG KF'},...
    'Color',GS,'GridAlpha',0.12,'Box','on','YTick',0:5:55);
grid on; ylabel('Median |Error| at NLOS Events (cm)');
title('NLOS Event Error','FontSize',10,'FontWeight','bold');
ylim([0 max(mv)*1.42]);
for i=1:3
    text(i,mv(i)+0.8,sprintf('%.1f cm',mv(i)),'HorizontalAlignment','center',...
        'FontSize',9.5,'FontWeight','bold','Color',[0.15 0.15 0.15]);
end

subplot(1,2,2); hold on;
edges=-50:4:110; ctr=edges(1:end-1)+2;
hr =histcounts(T.Error_Raw_cm, edges,'Normalization','probability');
hf =histcounts(T.Error_FixKF_cm,edges,'Normalization','probability');
hd =histcounts(T.Error_DDPG_cm, edges,'Normalization','probability');
plot(ctr,hr,'Color',Craw,'LineWidth',1.0,'DisplayName','Raw');
plot(ctr,hf,'Color',Cfix,'LineWidth',1.2,'DisplayName','Fixed KF');
plot(ctr,hd,'Color',Crl,'LineWidth',1.5,'DisplayName','DDPG KF');
xline(0,'--','Color',[0.45 0.45 0.45],'LineWidth',0.8,'HandleVisibility','off');
grid on; set(gca,'Color',GS,'GridAlpha',0.12,'Box','on',...
    'XTick',-50:25:100,'YTick',0:0.02:0.16);
xlabel('Error (cm)'); ylabel('Probability');
title('Error Distribution','FontSize',10,'FontWeight','bold');
legend('Location','northeast','FontSize',8,'Box','off');
xlim([-50 110]); ylim([0 0.16]);
sgtitle('NLOS Event Analysis and Error Distribution','FontSize',11,'FontWeight','bold');
exportgraphics(gcf,'fig8_nlos_analysis.png','Resolution',R,'BackgroundColor','white');
fprintf('Fig 8 saved\n'); close;

fprintf('\nDone. 7 figures saved at 300 dpi.\n');