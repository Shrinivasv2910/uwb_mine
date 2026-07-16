clear; clc; close all;

%% =========================================================
%  uwb_swarm_animation.m
%  UWB Swarm Animation — White Background, Clean Layout
%  NIT Patna | Shrinivas V | Dr. Golak Bihari Mahanta
%% =========================================================

if ~exist('ddpg_4anchor_trained.mat','file')
    error('ddpg_4anchor_trained.mat not found.');
end
d2=load('ddpg_4anchor_trained.mat');
fs=d2.fs; dt=d2.dt; W=d2.W; H=d2.H;
anchors=d2.anchors; nAnc=4;

%% PATHS
T_sw=30; t_sw=0:1/fs:T_sw; N_sw=length(t_sw);
ly=linspace(1,H-1,8);

ly_odd=ly(1:2:end);
px1=[]; py1=[];
for i=1:length(ly_odd)
    if mod(i,2)==1, px1=[px1 1 W-1]; else, px1=[px1 W-1 1]; end
    py1=[py1 ly_odd(i) ly_odd(i)];
end
dc1=[0 cumsum(sqrt(diff(px1).^2+diff(py1).^2))];
dq1=linspace(0,dc1(end),N_sw);
d1x=interp1(dc1,px1,dq1); d1y=interp1(dc1,py1,dq1);

ly_even=ly(2:2:end);
px2=[]; py2=[];
for i=1:length(ly_even)
    if mod(i,2)==1, px2=[px2 1 W-1]; else, px2=[px2 W-1 1]; end
    py2=[py2 ly_even(i) ly_even(i)];
end
dc2=[0 cumsum(sqrt(diff(px2).^2+diff(py2).^2))];
dq2=linspace(0,dc2(end),N_sw);
d2x=interp1(dc2,px2,dq2); d2y=interp1(dc2,py2,dq2);

%% HELPERS
function a=act_fwd(n,s)
    a=tanh(n.W3*max(0,n.W2*max(0,n.W1*s+n.b1)+n.b2)+n.b3);
end
function [qp,qv,rv]=decode_action(a,qpn,qpx,qvn,qvx,rn,rx)
    qp=max(qpn,min(qpx,qpn+(a(1)+1)/2*(qpx-qpn)));
    qv=max(qvn,min(qvx,qvn+(a(2)+1)/2*(qvx-qvn)));
    rv=max(rn, min(rx, rn +(a(3)+1)/2*(rx -rn )));
end
function s=build_state(r4,e4,vel2,nm,W,H)
    s=[r4/sqrt(W^2+H^2); e4/100; vel2/5; double(nm)];
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

%% PRE-COMPUTE
function [est_pos,qp_h,rv_h,raw_r,nlos_f]=precompute(tx,ty,anchors,d2,W,H,N,seed)
    nAnc=4; rng(seed);
    true_r=zeros(nAnc,N);
    for a=1:nAnc
        true_r(a,:)=sqrt((tx-anchors(a,1)).^2+(ty-anchors(a,2)).^2);
    end
    raw_r=zeros(nAnc,N); nlos_f=false(nAnc,N);
    for a=1:nAnc
        noise=0.20*randn(1,N)+(rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
        raw_r(a,:)=max(true_r(a,:)+noise,0.01);
        nlos_f(a,:)=(raw_r(a,:)-true_r(a,:))>0.20;
    end
    est_gn=zeros(2,N); p0=[W/2 H/2];
    for k=1:N
        try, est_gn(:,k)=gauss_newton(raw_r(:,k),anchors,p0)';
        catch, est_gn(:,k)=p0'; end
        p0=est_gn(:,k)';
    end
    est_gn(1,:)=max(min(est_gn(1,:),W+2),-2);
    est_gn(2,:)=max(min(est_gn(2,:),H+2),-2);
    xk=[est_gn(1,1);est_gn(2,1);0;0]; Pk=eye(4);
    est_pos=zeros(2,N); qp_h=zeros(1,N); rv_h=zeros(1,N);
    st=build_state(raw_r(:,1),zeros(4,1),[0;0],...
        mean(double(nlos_f(:,1))),W,H);
    pp=est_gn(:,1)';
    for k=1:N
        ao=act_fwd(d2.actor,st); ao(isnan(ao)|isinf(ao))=0;
        [qp,qv,rv]=decode_action(ao,...
            d2.Q_pos_min,d2.Q_pos_max,...
            d2.Q_vel_min,d2.Q_vel_max,...
            d2.R_min,d2.R_max);
        qp_h(k)=qp; rv_h(k)=rv;
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
        er=sqrt((pe(1)-d2.anchors(:,1)).^2+(pe(2)-d2.anchors(:,2)).^2);
        e4=(er-raw_r(:,k))*100; v2=xk(3:4);
        st=build_state(raw_r(:,k),e4,v2,mean(double(nlos_f(:,k))),W,H);
    end
end

fprintf('Pre-computing Drone 1...\n');
[d1_est,d1_qp,d1_rv,d1_raw,d1_nlos]=precompute(d1x,d1y,anchors,d2,W,H,N_sw,7);
fprintf('Pre-computing Drone 2...\n');
[d2_est,d2_qp,d2_rv,d2_raw,d2_nlos]=precompute(d2x,d2y,anchors,d2,W,H,N_sw,13);
fprintf('Done. Starting animation...\n\n');

%% ERRORS & RUNNING RMSE
err1=sqrt(sum((d1_est-[d1x;d1y]).^2,1))*100;
err2=sqrt(sum((d2_est-[d2x;d2y]).^2,1))*100;
rmse1=zeros(1,N_sw); rmse2=zeros(1,N_sw); a1c=0; a2c=0;
for k=1:N_sw
    a1c=a1c+err1(k)^2; rmse1(k)=sqrt(a1c/k);
    a2c=a2c+err2(k)^2; rmse2(k)=sqrt(a2c/k);
end

%% COVERAGE
grid_res=0.5; scan_r=3.0;
xg=0:grid_res:W; yg=0:grid_res:H;
[Xg,Yg]=meshgrid(xg,yg);
n_cells=numel(Xg);
cov_t=zeros(1,N_sw);
covered_pre=false(size(Xg));
for k=1:N_sw
    covered_pre=covered_pre|...
        (sqrt((Xg-d1x(k)).^2+(Yg-d1y(k)).^2)<=scan_r)|...
        (sqrt((Xg-d2x(k)).^2+(Yg-d2y(k)).^2)<=scan_r);
    cov_t(k)=100*sum(covered_pre(:))/n_cells;
end

%% SEPARATION
sep=sqrt((d1x-d2x).^2+(d1y-d2y).^2);

%% DRONE ICON PARAMS
arm_ang=[45 135 225 315]*pi/180;
drone_sc=0.50; prop_r=0.20;
th_p=linspace(0,2*pi,16); th_b=linspace(0,2*pi,12);

%% COLOURS — for white background
C1=[0.05 0.65 0.25];   % drone 1 green
C2=[0.85 0.15 0.08];   % drone 2 red
Ca=[0.80 0.10 0.10];   % anchor red
Cn=[0.95 0.50 0.00];   % NLOS orange
GS=[0.97 0.97 0.97];   % grid background

%% =========================================================
%  FIGURE — white background, 3 right panels, no dashboard
%% =========================================================
fig=figure('Color','w',...
    'Position',[20 30 1520 860],...
    'Name','UWB Swarm Animation | NIT Patna | Month 2',...
    'NumberTitle','off');

%% --- Mine Map (left, large) ---
ax_map=axes(fig,'Position',[0.03 0.08 0.54 0.86]);
hold(ax_map,'on'); axis(ax_map,'equal');
set(ax_map,'Color',GS,'XColor',[0.2 0.2 0.2],'YColor',[0.2 0.2 0.2],...
    'GridColor',[0.7 0.7 0.7],'GridAlpha',0.5,'Box','on',...
    'XTick',0:5:W,'YTick',0:5:H,'FontSize',9,'FontName','Arial');
grid(ax_map,'on');
xlim(ax_map,[-0.5 W+0.5]); ylim(ax_map,[-0.5 H+0.5]);
xlabel(ax_map,'X (m)','FontSize',10,'Color',[0.2 0.2 0.2]);
ylabel(ax_map,'Y (m)','FontSize',10,'Color',[0.2 0.2 0.2]);
title(ax_map,'2-Drone Swarm  |  UWB Mine Monitoring  |  NIT Patna',...
    'FontSize',12,'FontWeight','bold','Color',[0.1 0.1 0.1]);

% Mine boundary
rectangle(ax_map,'Position',[0 0 W H],'EdgeColor',[0.5 0.5 0.5],...
    'LineWidth',1.5,'LineStyle','--');

% Coverage image
cov_rgb=zeros(length(yg),length(xg),3);
cov_alpha=zeros(length(yg),length(xg));
h_cov_img=imagesc(ax_map,xg,yg,cov_rgb);
set(h_cov_img,'AlphaData',cov_alpha);

% Planned paths (very faint)
plot(ax_map,d1x,d1y,'Color',[C1 0.15],'LineWidth',1.0);
plot(ax_map,d2x,d2y,'Color',[C2 0.15],'LineWidth',1.0);

% Estimated trails
h_trail1=plot(ax_map,NaN,NaN,'Color',[C1 0.55],'LineWidth',1.3);
h_trail2=plot(ax_map,NaN,NaN,'Color',[C2 0.55],'LineWidth',1.3);

% Anchors
h_anc=gobjects(nAnc,1);
anc_lbl_off=[0.5 0.7; W-2.5 0.7; W-2.5 H-1.2; 0.5 H-1.2];
for a=1:nAnc
    h_anc(a)=scatter(ax_map,anchors(a,1),anchors(a,2),200,...
        'rs','filled','MarkerFaceColor',Ca,'MarkerEdgeColor','k','LineWidth',1.0);
    text(ax_map,anc_lbl_off(a,1),anc_lbl_off(a,2),sprintf('A%d',a),...
        'Color',[0.65 0.05 0.05],'FontSize',9,'FontWeight','bold');
end

% UWB ranging lines
h_uwb1=gobjects(nAnc,1); h_uwb2=gobjects(nAnc,1);
for a=1:nAnc
    h_uwb1(a)=plot(ax_map,[anchors(a,1) NaN],[anchors(a,2) NaN],...
        'Color',[C1 0.65],'LineWidth',0.9);
    h_uwb2(a)=plot(ax_map,[anchors(a,1) NaN],[anchors(a,2) NaN],...
        'Color',[C2 0.65],'LineWidth',0.9,'LineStyle','--');
end

% Drone 1 icon
h_arm1=gobjects(4,1); h_prop1=gobjects(4,1);
for ii=1:4
    h_arm1(ii)=plot(ax_map,NaN,NaN,'Color',[0.15 0.15 0.15],'LineWidth',2.2);
    h_prop1(ii)=patch(ax_map,NaN,NaN,C1,'EdgeColor',[0.02 0.40 0.15],...
        'LineWidth',1.0,'FaceAlpha',0.85);
end
h_body1=patch(ax_map,NaN,NaN,[0.25 0.25 0.25],'EdgeColor',[0.4 0.4 0.4],'LineWidth',0.8);
h_dir1=plot(ax_map,NaN,NaN,'Color',[0.95 0.80 0.00],'LineWidth',2.2);

% Drone 2 icon
h_arm2=gobjects(4,1); h_prop2=gobjects(4,1);
for ii=1:4
    h_arm2(ii)=plot(ax_map,NaN,NaN,'Color',[0.15 0.15 0.15],'LineWidth',2.2);
    h_prop2(ii)=patch(ax_map,NaN,NaN,C2,'EdgeColor',[0.55 0.05 0.03],...
        'LineWidth',1.0,'FaceAlpha',0.85);
end
h_body2=patch(ax_map,NaN,NaN,[0.25 0.25 0.25],'EdgeColor',[0.4 0.4 0.4],'LineWidth',0.8);
h_dir2=plot(ax_map,NaN,NaN,'Color',[0.10 0.50 0.90],'LineWidth',2.2);

% Drone labels — track with drone position
h_lbl1=text(ax_map,NaN,NaN,'D1','Color',C1,'FontSize',8,...
    'FontWeight','bold','HorizontalAlignment','center');
h_lbl2=text(ax_map,NaN,NaN,'D2','Color',C2,'FontSize',8,...
    'FontWeight','bold','HorizontalAlignment','center');

% Time + coverage overlay on map (top-left, clean)
h_time_txt=text(ax_map,0.5,H-0.6,'t = 0.0 s',...
    'FontSize',11,'FontWeight','bold','Color',[0.15 0.15 0.15]);
h_cov_txt=text(ax_map,0.5,H-1.5,'Coverage: 0.0%',...
    'FontSize',11,'FontWeight','bold','Color',[0.05 0.55 0.15]);

%% --- RMSE plot (right top) ---
ax_rmse=axes(fig,'Position',[0.61 0.68 0.37 0.25]);
hold(ax_rmse,'on');
set(ax_rmse,'Color',GS,'GridColor',[0.7 0.7 0.7],'GridAlpha',0.5,...
    'Box','on','FontSize',9,'FontName','Arial');
grid(ax_rmse,'on');
h_r1=animatedline(ax_rmse,'Color',C1,'LineWidth',2.0,'DisplayName','Drone 1 (odd lanes)');
h_r2=animatedline(ax_rmse,'Color',C2,'LineWidth',2.0,'DisplayName','Drone 2 (even lanes)');
xlim(ax_rmse,[0 T_sw]); ylim(ax_rmse,[0 55]);
xlabel(ax_rmse,'Time (s)','FontSize',9);
ylabel(ax_rmse,'RMSE (cm)','FontSize',9);
title(ax_rmse,'Running RMSE per Drone','FontSize',10,'FontWeight','bold');
legend(ax_rmse,'Location','northeast','FontSize',8,'Box','off');
yline(ax_rmse,50,'--','Color',[0.80 0.15 0.15],'LineWidth',1.2,...
    'Label','DST 50 cm','LabelHorizontalAlignment','left',...
    'LabelVerticalAlignment','bottom','FontSize',8);
% Live RMSE values — placed inside axes, separated
h_rmse1_txt=text(ax_rmse,T_sw*0.75,48,'','FontSize',9,'Color',C1,...
    'FontWeight','bold','HorizontalAlignment','center');
h_rmse2_txt=text(ax_rmse,T_sw*0.75,43,'','FontSize',9,'Color',C2,...
    'FontWeight','bold','HorizontalAlignment','center');

%% --- Coverage plot (right middle) ---
ax_cov=axes(fig,'Position',[0.61 0.37 0.37 0.25]);
hold(ax_cov,'on');
set(ax_cov,'Color',GS,'GridColor',[0.7 0.7 0.7],'GridAlpha',0.5,...
    'Box','on','FontSize',9,'FontName','Arial');
grid(ax_cov,'on');
h_cline=animatedline(ax_cov,'Color',[0.05 0.55 0.15],'LineWidth',2.0);
xlim(ax_cov,[0 T_sw]); ylim(ax_cov,[0 105]);
xlabel(ax_cov,'Time (s)','FontSize',9);
ylabel(ax_cov,'Coverage (%)','FontSize',9);
title(ax_cov,'Mine Area Coverage (scan radius = 3 m)','FontSize',10,'FontWeight','bold');
yline(ax_cov,100,'--','Color',[0.05 0.55 0.15],'LineWidth',1.0,...
    'HandleVisibility','off');
% Coverage % label inside plot
h_cov_pct_ax=text(ax_cov,2,90,'0.0%',...
    'FontSize',14,'FontWeight','bold','Color',[0.05 0.55 0.15]);

%% --- Separation plot (right bottom) ---
ax_sep=axes(fig,'Position',[0.61 0.08 0.37 0.22]);
hold(ax_sep,'on');
set(ax_sep,'Color',GS,'GridColor',[0.7 0.7 0.7],'GridAlpha',0.5,...
    'Box','on','FontSize',9,'FontName','Arial');
grid(ax_sep,'on');
h_sepline=animatedline(ax_sep,'Color',[0.15 0.45 0.80],'LineWidth',2.0);
xlim(ax_sep,[0 T_sw]); ylim(ax_sep,[0 4]);
xlabel(ax_sep,'Time (s)','FontSize',9);
ylabel(ax_sep,'Separation (m)','FontSize',9);
title(ax_sep,'Inter-Drone Separation','FontSize',10,'FontWeight','bold');
yline(ax_sep,2.0,'--','Color',[0.85 0.40 0.05],'LineWidth',1.5,...
    'Label','2 m min safe','LabelHorizontalAlignment','left',...
    'LabelVerticalAlignment','bottom','FontSize',8);
% NLOS counter label — bottom right of sep plot
h_nlos_txt=text(ax_sep,T_sw*0.75,3.5,'NLOS: 0',...
    'FontSize',10,'FontWeight','bold','Color',[0.85 0.40 0.05],...
    'HorizontalAlignment','center');

%% =========================================================
%  ANIMATION LOOP
%% =========================================================
nlos_count=0;
covered_anim=false(size(Xg));

fprintf('%-6s %-8s %-10s %-10s %-8s\n','t(s)','Cov%','D1_RMSE','D2_RMSE','Sep(m)');
fprintf('%s\n',repmat('-',1,46));

for k=1:N_sw
    if ~ishandle(fig), break; end

    %% Coverage grid update
    dist1=sqrt((Xg-d1x(k)).^2+(Yg-d1y(k)).^2);
    dist2=sqrt((Xg-d2x(k)).^2+(Yg-d2y(k)).^2);
    covered_anim=covered_anim|(dist1<=scan_r)|(dist2<=scan_r);
    cov_rgb(:,:,1)=0.55*double(covered_anim);
    cov_rgb(:,:,2)=0.90*double(covered_anim);
    cov_rgb(:,:,3)=0.55*double(covered_anim);
    cov_alpha=0.30*double(covered_anim);
    set(h_cov_img,'CData',cov_rgb,'AlphaData',cov_alpha);

    %% Trails
    set(h_trail1,'XData',d1_est(1,1:k),'YData',d1_est(2,1:k));
    set(h_trail2,'XData',d2_est(1,1:k),'YData',d2_est(2,1:k));

    %% Headings
    if k>1
        hdg1=atan2(d1y(k)-d1y(k-1),d1x(k)-d1x(k-1));
        hdg2=atan2(d2y(k)-d2y(k-1),d2x(k)-d2x(k-1));
    else
        hdg1=pi/2; hdg2=pi/2;
    end

    %% Drone 1 icon
    rot1=arm_ang+hdg1;
    for ii=1:4
        tx=d1x(k)+drone_sc*cos(rot1(ii));
        ty=d1y(k)+drone_sc*sin(rot1(ii));
        set(h_arm1(ii),'XData',[d1x(k) tx],'YData',[d1y(k) ty]);
        set(h_prop1(ii),'XData',tx+prop_r*cos(th_p),'YData',ty+prop_r*sin(th_p));
    end
    set(h_body1,'XData',d1x(k)+0.14*cos(th_b),'YData',d1y(k)+0.14*sin(th_b));
    set(h_dir1,'XData',[d1x(k) d1x(k)+drone_sc*0.7*cos(hdg1)],...
               'YData',[d1y(k) d1y(k)+drone_sc*0.7*sin(hdg1)]);
    set(h_lbl1,'Position',[d1x(k) d1y(k)+drone_sc+0.35]);

    %% Drone 2 icon
    rot2=arm_ang+hdg2;
    for ii=1:4
        tx=d2x(k)+drone_sc*cos(rot2(ii));
        ty=d2y(k)+drone_sc*sin(rot2(ii));
        set(h_arm2(ii),'XData',[d2x(k) tx],'YData',[d2y(k) ty]);
        set(h_prop2(ii),'XData',tx+prop_r*cos(th_p),'YData',ty+prop_r*sin(th_p));
    end
    set(h_body2,'XData',d2x(k)+0.14*cos(th_b),'YData',d2y(k)+0.14*sin(th_b));
    set(h_dir2,'XData',[d2x(k) d2x(k)+drone_sc*0.7*cos(hdg2)],...
               'YData',[d2y(k) d2y(k)+drone_sc*0.7*sin(hdg2)]);
    set(h_lbl2,'Position',[d2x(k) d2y(k)+drone_sc+0.35]);

    %% UWB lines + anchor flash
    for a=1:nAnc
        nl1=d1_nlos(a,k); nl2=d2_nlos(a,k);

        c1a=C1*(1-0.5*nl1)+[0.95 0.50 0.00]*0.5*nl1;
        set(h_uwb1(a),'XData',[anchors(a,1) d1x(k)],...
                       'YData',[anchors(a,2) d1y(k)],...
                       'Color',[c1a 0.65+0.25*nl1],...
                       'LineWidth',0.9+1.2*nl1,...
                       'LineStyle',iif(nl1,'--','-'));

        c2a=C2*(1-0.5*nl2)+[0.95 0.50 0.00]*0.5*nl2;
        set(h_uwb2(a),'XData',[anchors(a,1) d2x(k)],...
                       'YData',[anchors(a,2) d2y(k)],...
                       'Color',[c2a 0.65+0.25*nl2],...
                       'LineWidth',0.9+1.2*nl2,...
                       'LineStyle',iif(nl2,':','-'));

        if nl1||nl2
            set(h_anc(a),'MarkerFaceColor',Cn);
        else
            set(h_anc(a),'MarkerFaceColor',Ca);
        end
    end
    nlos_count=nlos_count+sum(d1_nlos(:,k))+sum(d2_nlos(:,k));

    %% Map overlays
    set(h_time_txt,'String',sprintf('t = %.1f s',t_sw(k)));
    set(h_cov_txt, 'String',sprintf('Coverage: %.1f%%',cov_t(k)));

    %% Right panels
    addpoints(h_r1,t_sw(k),rmse1(k));
    addpoints(h_r2,t_sw(k),rmse2(k));
    set(h_rmse1_txt,'String',sprintf('D1: %.1f cm',rmse1(k)));
    set(h_rmse2_txt,'String',sprintf('D2: %.1f cm',rmse2(k)));

    addpoints(h_cline,t_sw(k),cov_t(k));
    set(h_cov_pct_ax,'String',sprintf('%.1f%%',cov_t(k)));

    addpoints(h_sepline,t_sw(k),sep(k));
    sep_col=iif(sep(k)<2.0,[0.85 0.40 0.05],[0.15 0.45 0.80]);
    set(h_nlos_txt,'String',sprintf('NLOS events: %d',nlos_count),...
        'Color',[0.85 0.40 0.05]);

    if mod(k,10)==0
        fprintf('%-6.1f %-8.1f %-10.2f %-10.2f %-8.2f\n',...
            t_sw(k),cov_t(k),rmse1(k),rmse2(k),sep(k));
    end

    pause(0.055);
    drawnow limitrate;
end

fprintf('%s\n',repmat('-',1,46));
fprintf('Done. Coverage: %.1f%% | D1: %.2f cm | D2: %.2f cm\n',...
    cov_t(end),rmse1(end),rmse2(end));

function out=iif(cond,a,b)
    if cond, out=a; else, out=b; end
end