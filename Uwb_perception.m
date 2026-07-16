clear; clc; close all;

%% =========================================================
%  uwb_perception.m
%  UWB-Anchored GPS-Denied Drone Positioning — Mine Monitoring
%  Month 2 | Task 6: Perception Geometry and Localisation Uncertainty
%
%  Camera: Waveshare IMX219-83 Stereo
%    Sensor  : Sony IMX219, 8MP
%    Resolution: 3280 x 2464 pixels (horizontal used)
%    FOV     : 83 degrees (horizontal field of view)
%
%  Geometry:
%    GSD = 2*h*tan(fov/2) / px_width        [m/pixel]
%    sigma_g = GSD * sigma_px                [m, per axis]
%    sqrt(trace(C)) = sigma_g * sqrt(2)      [DST metric, 2D]
%
%  DST acceptance: sqrt(trace(C)) <= 0.5 m
%
%  NIT Patna | Shrinivas V (2350011) | Dr. Golak Bihari Mahanta
%% =========================================================

%% PARALLEL POOL
nCores=feature('numcores'); cl=parcluster('local');
maxW=cl.NumWorkers; nUse=min(maxW,max(1,floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n',nCores,nUse);
if isempty(gcp('nocreate')), parpool('local',nUse); end

%% CAMERA SPECS — Waveshare IMX219-83 Stereo
cam.name = 'Waveshare IMX219-83 Stereo';
cam.px   = 3280;     % horizontal pixels
cam.fov  = 83;       % horizontal FOV (degrees)
cam.desc = '8MP Sony IMX219 | 3280x2464 | 83 deg HFOV';

sigma_px_vals = [3 5 8 12 20];   % detection bounding box uncertainty (pixels)
sigma_px_ref  = 5;               % reference case
h_vals        = 0.5:0.05:15.0;  % altitude range (m)
DST_target    = 0.5;             % metres

%% HELPER FUNCTIONS
function sigma_g = ground_sigma(h, fov_deg, px_width, sigma_px)
    fov_rad = fov_deg * pi/180;
    GSD     = 2 .* h .* tan(fov_rad/2) ./ px_width;  % m/pixel
    sigma_g = GSD .* sigma_px;                         % elementwise
end
function d = dst_metric(sigma_g)
    d = sigma_g .* sqrt(2);
end
function s = pass_fail(c)
    if c, s='PASSED'; else, s='FAILED'; end
end

%% =========================================================
%  GSD TABLE at key altitudes
%% =========================================================
fprintf('%s\n  Camera: %s\n  %s\n%s\n\n',...
    repmat('=',1,65),cam.name,cam.desc,repmat('=',1,65));

h_key = [2 3 4 5 7 10 15];
fprintf('  GSD (cm/pixel) at key altitudes:\n');
fprintf('  %-8s','Alt (m)');
for hk=h_key, fprintf('  %5dm',hk); end, fprintf('\n');
fprintf('  %-8s','GSD');
for hk=h_key
    gsd=ground_sigma(hk,cam.fov,cam.px,1)*100;
    fprintf('  %4.2f ',gsd);
end
fprintf('\n\n');

%% =========================================================
%  MAX ALTITUDE TABLE — per sigma_px
%% =========================================================
fprintf('  Max altitude to meet DST 0.5m criterion:\n');
fprintf('  %-10s  %-10s  %-12s  %-10s\n',...
    'sigma_px','Max alt(m)','sqrt(C) @alt','GSD @alt');
fprintf('  %s\n',repmat('-',1,50));
h_max_vals = zeros(1,length(sigma_px_vals));
for si=1:length(sigma_px_vals)
    sp=sigma_px_vals(si);
    dv=dst_metric(ground_sigma(h_vals,cam.fov,cam.px,sp));
    idx=find(dv<=DST_target,1,'last');
    if ~isempty(idx)
        h_max_vals(si)=h_vals(idx);
        gsd_at=ground_sigma(h_vals(idx),cam.fov,cam.px,1)*100;
        fprintf('  %-10d  %-10.1f  %-12.2f  %.2f cm/px\n',...
            sp,h_vals(idx),dv(idx)*100,gsd_at);
    else
        fprintf('  %-10d  not met\n',sp);
    end
end
fprintf('\n');

%% =========================================================
%  MULTI-VIEW FUSION
%% =========================================================
sg_single      = ground_sigma(h_vals, cam.fov, cam.px, sigma_px_ref);
dst_single     = dst_metric(sg_single);
sg_fused_same  = sg_single / sqrt(2);
dst_fused_same = dst_metric(sg_fused_same);

h1_fix = 4.0;
sg1 = ground_sigma(h1_fix, cam.fov, cam.px, sigma_px_ref);
sg_fused_diff = zeros(1,length(h_vals));
for i=1:length(h_vals)
    sg2 = ground_sigma(h_vals(i), cam.fov, cam.px, sigma_px_ref);
    sg_fused_diff(i) = 1/sqrt(1/sg1^2 + 1/sg2^2);
end
dst_fused_diff = dst_metric(sg_fused_diff);

idx_s = find(dst_single    <=DST_target,1,'last');
idx_f = find(dst_fused_same<=DST_target,1,'last');
idx_d = find(dst_fused_diff<=DST_target,1,'last');
h_single_min = h_vals(idx_s);
h_same_min   = h_vals(idx_f);
h_diff_min   = h_vals(idx_d);

fprintf('  Multi-View Fusion  |  sigma_px=%d\n\n',sigma_px_ref);
fprintf('  Single drone max altitude        : %.1f m\n',h_single_min);
fprintf('  2 drones, same altitude          : %.1f m  (%.2fx headroom)\n',...
    h_same_min, h_same_min/h_single_min);
fprintf('  2 drones, D1@%.1fm + D2 at h    : %.1f m  (%.2fx headroom)\n\n',...
    h1_fix, h_diff_min, h_diff_min/h_single_min);

fprintf('  At 5m altitude, sigma_px=%d:\n',sigma_px_ref);
dv5  = dst_metric(ground_sigma(5,cam.fov,cam.px,sigma_px_ref));
dv5f = dst_metric(ground_sigma(5,cam.fov,cam.px,sigma_px_ref)/sqrt(2));
fprintf('    Single drone   : %.2f cm  --> %s\n',dv5*100, pass_fail(dv5<=DST_target));
fprintf('    2-drone fusion : %.2f cm  --> %s\n\n',dv5f*100,pass_fail(dv5f<=DST_target));

%% COLOURS
C_single=[0.92 0.22 0.08];
C_fused =[0.05 0.72 0.32];
C_fused2=[0.15 0.45 0.80];
GS=[0.97 0.97 0.97]; RES=300;
sp_cols={[0.05 0.72 0.32],[0.15 0.45 0.80],...
         [0.92 0.55 0.05],[0.85 0.20 0.20],[0.50 0.10 0.80]};

%% FIG 1 — DST METRIC vs ALTITUDE (all sigma_px values)
figure('Color','w','Position',[50 50 820 460],'NumberTitle','off');
axes('Position',[0.10 0.14 0.86 0.76]); hold on;

patch([0 max(h_vals) max(h_vals) 0],[0 0 50 50],...
    [0.80 0.95 0.80],'FaceAlpha',0.18,'EdgeColor','none','HandleVisibility','off');
text(1.5,8,'Acceptable','FontSize',8,'Color',[0.15 0.55 0.15],'HorizontalAlignment','center');

for si=1:length(sigma_px_vals)
    sp=sigma_px_vals(si);
    dv=dst_metric(ground_sigma(h_vals,cam.fov,cam.px,sp));
    plot(h_vals,dv*100,'Color',sp_cols{si},'LineWidth',1.4,...
        'DisplayName',sprintf('sigma_{px}=%d px',sp));
    idx=find(dv<=DST_target,1,'last');
    if ~isempty(idx)
        plot(h_vals(idx),dv(idx)*100,'o','MarkerSize',7,...
            'Color',sp_cols{si},'MarkerFaceColor',sp_cols{si},...
            'HandleVisibility','off');
        text(h_vals(idx)+0.2,dv(idx)*100+2,...
            sprintf('%.1f m',h_vals(idx)),...
            'FontSize',8,'Color',sp_cols{si},'FontWeight','bold');
    end
end
yline(DST_target*100,'--','Color',[0.85 0.20 0.20],'LineWidth',1.8,...
    'Label','DST criterion: 50 cm',...
    'LabelHorizontalAlignment','left','HandleVisibility','off');

grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8,...
    'XTick',0:1:15,'YTick',0:10:200);
xlabel('Drone Altitude (m)'); ylabel('sqrt(trace(C))  (cm)');
title(sprintf('Detection Localisation Uncertainty vs Altitude  |  %s',cam.name),...
    'FontSize',10,'FontWeight','bold');
legend('Location','northwest','FontSize',9,'Box','off');
xlim([0.5 15]); ylim([0 200]);
exportgraphics(gcf,'perc_dst_vs_altitude.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 1 saved: perc_dst_vs_altitude.png\n'); close;

%% FIG 2 — MULTI-VIEW FUSION BENEFIT
figure('Color','w','Position',[50 50 820 440],'NumberTitle','off');
axes('Position',[0.10 0.14 0.86 0.76]); hold on;

patch([0 max(h_vals) max(h_vals) 0],[0 0 50 50],...
    [0.80 0.95 0.80],'FaceAlpha',0.18,'EdgeColor','none','HandleVisibility','off');
text(2.0,8,'Acceptable','FontSize',8,'Color',[0.15 0.55 0.15],'HorizontalAlignment','center');

plot(h_vals,dst_single*100,    'Color',C_single,'LineWidth',1.5,'DisplayName','Single drone');
plot(h_vals,dst_fused_same*100,'Color',C_fused, 'LineWidth',1.5,'DisplayName','2 drones, same altitude');
plot(h_vals,dst_fused_diff*100,'Color',C_fused2,'LineWidth',1.5,...
    'DisplayName',sprintf('2 drones, D1@%.1fm + D2 at h',h1_fix));

h_marks  = [h_single_min  h_same_min   h_diff_min];
col_marks = {C_single,    C_fused,     C_fused2};
for mi=1:3
    hm=h_marks(mi); col=col_marks{mi};
    dv=dst_metric(ground_sigma(hm,cam.fov,cam.px,sigma_px_ref))*100;
    plot(hm,dv,'o','MarkerSize',8,'Color',col,'MarkerFaceColor',col,...
        'HandleVisibility','off');
    text(hm+0.2,dv+2.5,sprintf('%.1f m',hm),...
        'FontSize',8,'Color',col,'FontWeight','bold');
end

yline(DST_target*100,'--','Color',[0.85 0.20 0.20],'LineWidth',1.8,...
    'Label','DST 50 cm','LabelHorizontalAlignment','left','HandleVisibility','off');

grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8,...
    'XTick',0:1:15,'YTick',0:10:200);
xlabel('Drone Altitude (m)'); ylabel('sqrt(trace(C))  (cm)');
title(sprintf('Multi-View Fusion Benefit  |  %s  |  sigma_{px}=%d px',...
    cam.name,sigma_px_ref),'FontSize',10,'FontWeight','bold');
legend('Location','northwest','FontSize',9,'Box','off');
xlim([0.5 15]); ylim([0 200]);
exportgraphics(gcf,'perc_fusion_benefit.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 2 saved: perc_fusion_benefit.png\n'); close;

%% FIG 3 — GSD vs ALTITUDE
figure('Color','w','Position',[50 50 820 400],'NumberTitle','off');
axes('Position',[0.10 0.14 0.86 0.76]); hold on;

gsd_vals = ground_sigma(h_vals,cam.fov,cam.px,1)*100;  % cm/pixel
plot(h_vals,gsd_vals,'Color',[0.15 0.45 0.80],'LineWidth',1.8,...
    'DisplayName',cam.name);

% Mark key altitudes
h_mark=[3 5 7 10];
for hm=h_mark
    gsd_m=ground_sigma(hm,cam.fov,cam.px,1)*100;
    plot(hm,gsd_m,'o','MarkerSize',7,'Color',[0.15 0.45 0.80],...
        'MarkerFaceColor',[0.15 0.45 0.80],'HandleVisibility','off');
    text(hm+0.2,gsd_m+0.02,sprintf('%.2f cm/px\n@%dm',gsd_m,hm),...
        'FontSize',7.5,'Color',[0.15 0.45 0.80],'FontWeight','bold');
end

grid on; set(gca,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',8,...
    'XTick',0:1:15,'YTick',0:0.5:5);
xlabel('Drone Altitude (m)'); ylabel('GSD (cm/pixel)');
title(sprintf('Ground Sampling Distance vs Altitude  |  %s',cam.name),...
    'FontSize',10,'FontWeight','bold');
legend('Location','northwest','FontSize',9,'Box','off');
xlim([0.5 15]);
exportgraphics(gcf,'perc_gsd_vs_altitude.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 3 saved: perc_gsd_vs_altitude.png\n'); close;

%% FIG 4 — OPERATING ENVELOPE HEATMAP (single camera, fixed)
figure('Color','w','Position',[50 50 700 460],'NumberTitle','off');
axes('Position',[0.11 0.13 0.75 0.77]); hold on;

h_fine  = 0.5:0.1:15.0;
sp_fine = 1:1:25;
% Scalar loop to avoid matrix multiply issue
DST_grid = zeros(length(sp_fine), length(h_fine));
for si=1:length(sp_fine)
    for hi=1:length(h_fine)
        DST_grid(si,hi) = dst_metric(ground_sigma(...
            h_fine(hi), cam.fov, cam.px, sp_fine(si))) * 100;
    end
end

imagesc(h_fine, sp_fine, DST_grid);
colormap(flipud(summer));
cb=colorbar; cb.Label.String='sqrt(trace(C))  (cm)'; cb.FontSize=8;
clim([0 200]);
contour(h_fine, sp_fine, DST_grid, [50 50], 'r-', 'LineWidth',2.5);
text(7.5, 22, 'Red line: DST=50cm', 'FontSize',8,...
    'Color',[0.85 0.10 0.10],'HorizontalAlignment','center');

set(gca,'FontSize',8,'YDir','normal');
xlabel('Drone Altitude (m)','FontSize',9);
ylabel('sigma_{px} (pixels)','FontSize',9);
title(sprintf('Operating Envelope  |  %s\nsqrt(trace(C)) (cm)',cam.name),...
    'FontSize',9,'FontWeight','bold');
xlim([0.5 15]); ylim([1 25]);
exportgraphics(gcf,'perc_operating_envelope.png','Resolution',RES,'BackgroundColor','white');
fprintf('Fig 4 saved: perc_operating_envelope.png\n'); close;

%% EXPORT CSV
csv_file='uwb_perception_results.csv';
fid=fopen(csv_file,'w');
fprintf(fid,'Altitude_m,GSD_cmperpx,DST_Single_cm,DST_FusedSame_cm,DST_FusedDiff_cm\n');
for i=1:length(h_vals)
    gsd=ground_sigma(h_vals(i),cam.fov,cam.px,1)*100;
    fprintf(fid,'%.2f,%.4f,%.4f,%.4f,%.4f\n',...
        h_vals(i),gsd,...
        dst_single(i)*100,dst_fused_same(i)*100,dst_fused_diff(i)*100);
end
fclose(fid);
fprintf('Saved: %s  [%d rows]\n\n',csv_file,length(h_vals));

save('perception_results.mat','h_vals','sigma_px_vals','h_max_vals',...
    'dst_single','dst_fused_same','dst_fused_diff',...
    'h_single_min','h_same_min','h_diff_min',...
    'cam','DST_target','sigma_px_ref');
fprintf('Saved: perception_results.mat\n\n');

fprintf('%s\n  Task 6 complete.\n\n',repmat('=',1,65));
fprintf('  %s\n\n',cam.desc);
fprintf('  Single drone max altitude for DST 0.5m : %.1f m\n',h_single_min);
fprintf('  2-drone fusion (same alt) max altitude  : %.1f m  (%.2fx)\n',...
    h_same_min,h_same_min/h_single_min);
fprintf('  At 5m, sigma_px=%d: %.2f cm --> %s\n',...
    sigma_px_ref,dv5*100,pass_fail(dv5<=DST_target));
fprintf('  At 5m, 2-drone fusion: %.2f cm --> %s\n',...
    dv5f*100,pass_fail(dv5f<=DST_target));
fprintf('%s\n',repmat('=',1,65));