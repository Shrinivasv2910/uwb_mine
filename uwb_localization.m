clear; clc; close all;

fs=10; T=60; t=0:1/fs:T; N=length(t); dt=1/fs;

W=20; H=15;
anchors=[0 0; W 0; W H; 0 H];
nA=4;

% Boustrophedon path
lanes=8; lane_y=linspace(1,H-1,lanes);
pts_x=[]; pts_y=[];
for L=1:lanes
    if mod(L,2)==1, pts_x=[pts_x 1 W-1]; else, pts_x=[pts_x W-1 1]; end
    pts_y=[pts_y lane_y(L) lane_y(L)];
end
d_cum=[0 cumsum(sqrt(diff(pts_x).^2+diff(pts_y).^2))];
d_query=linspace(0,d_cum(end),N);
true_x=interp1(d_cum,pts_x,d_query,'linear');
true_y=interp1(d_cum,pts_y,d_query,'linear');

true_ranges=zeros(nA,N);
for a=1:nA
    true_ranges(a,:)=sqrt((true_x-anchors(a,1)).^2+(true_y-anchors(a,2)).^2);
end

raw_r=zeros(nA,N); fir_r=zeros(nA,N);
b_fir=fir1(8,0.3);
for a=1:nA
    n=0.20*randn(1,N)+(rand(1,N)<0.10).*(0.30+0.70*rand(1,N));
    raw_r(a,:)=max(true_ranges(a,:)+n,0.01);
    fir_r(a,:)=filtfilt(b_fir,1,medfilt1(raw_r(a,:),5));
end

function pos=gn(ranges,anch,p0)
    p=p0;
    for i=1:15
        d=sqrt(sum((anch-p).^2,2));
        J=-(anch-p)./d;
        dp=(J'*J)\(J'*(ranges-d));
        p=p+dp';
        if norm(dp)<1e-6, break; end
    end
    pos=p;
end

est_raw=zeros(2,N); est_fir=zeros(2,N);
p0=[W/2 H/2];
for k=1:N
    try, est_raw(:,k)=gn(raw_r(:,k),anchors,p0)'; catch, est_raw(:,k)=p0'; end
    try, est_fir(:,k)=gn(fir_r(:,k),anchors,p0)'; catch, est_fir(:,k)=p0'; end
    p0=est_fir(:,k)';
end
est_raw(1,:)=max(min(est_raw(1,:),W+2),-2);
est_raw(2,:)=max(min(est_raw(2,:),H+2),-2);
est_fir(1,:)=max(min(est_fir(1,:),W+2),-2);
est_fir(2,:)=max(min(est_fir(2,:),H+2),-2);

Fk=[1 0 dt 0;0 1 0 dt;0 0 1 0;0 0 0 1];
Hk=[1 0 0 0;0 1 0 0];
Qk=diag([0.001 0.001 0.01 0.01]);
Rk=diag([0.12^2 0.12^2]);
xk=[est_fir(1,1);est_fir(2,1);0;0]; Pk=eye(4);
kf2d=zeros(2,N);
for k=1:N
    xk=Fk*xk; Pk=Fk*Pk*Fk'+Qk;
    Kk=Pk*Hk'/(Hk*Pk*Hk'+Rk);
    xk=xk+Kk*(est_fir(:,k)-Hk*xk);
    Pk=(eye(4)-Kk*Hk)*Pk;
    kf2d(:,k)=xk(1:2);
end

err_raw=sqrt(sum((est_raw-[true_x;true_y]).^2,1))*100;
err_fir=sqrt(sum((est_fir-[true_x;true_y]).^2,1))*100;
err_kf =sqrt(sum((kf2d  -[true_x;true_y]).^2,1))*100;
fprintf('RMSE  Raw=%.1fcm  FIR=%.1fcm  Kalman=%.1fcm\n',...
    mean(err_raw),mean(err_fir),mean(err_kf));

Ctr=[0.55 0.55 0.55]; Craw=[0.92 0.22 0.08];
Cfir=[0.60 0.15 0.85]; Ckf=[0.05 0.48 0.90];

fig=figure('Color','w','Position',[40 40 1400 720],...
    'Name','UWB 2D Localization  |  4 Anchors  |  NIT Patna','NumberTitle','off');
tiledlayout(2,3,'TileSpacing','compact','Padding','tight');

%% MAP - no legend, only yellow/red/pink lines
ax1=nexttile([2 1]); hold on;
rectangle('Position',[0 0 W H],'EdgeColor',[0.80 0.80 0.80],'LineWidth',1.5,'LineStyle','--');
for a=1:nA
    scatter(ax1,anchors(a,1),anchors(a,2),160,'r','s','filled');
    text(ax1,anchors(a,1)+0.5,anchors(a,2)+0.7,sprintf('A%d',a),...
        'FontSize',9,'FontWeight','bold','Color',[0.75 0.08 0.08]);
end
% Yellow = planned path
plot(ax1,true_x,true_y,'Color',[0.95 0.82 0.05],'LineWidth',1.4);
% Red = true covered path
h_cov=plot(ax1,NaN,NaN,'Color',[0.88 0.12 0.08],'LineWidth',1.8);
% Pink lines from ALL 4 anchors to drone
h_pnk=gobjects(nA,1);
for a=1:nA
    h_pnk(a)=plot(ax1,[anchors(a,1) NaN],[anchors(a,2) NaN],...
        'Color',[1.0 0.42 0.70],'LineWidth',1.0);
end
% Drone dot
h_dot=scatter(ax1,NaN,NaN,110,Ckf,'o','filled');
xlim(ax1,[-2 W+3]); ylim(ax1,[-2 H+3]);
legend(ax1,'off');
set(ax1,'Color','w','GridAlpha',0.12,'FontSize',8,'Box','on'); grid(ax1,'on');
xlabel(ax1,'X (m)','FontSize',8); ylabel(ax1,'Y (m)','FontSize',8);
title(ax1,'Mine Map  |  4 UWB Anchors  |  Boustrophedon','FontSize',9,'FontWeight','bold');

%% X POSITION
ax2=nexttile; hold on;
h_tx=animatedline(ax2,'Color',Ctr, 'LineWidth',2.0,'DisplayName','True X');
h_rx=animatedline(ax2,'Color',Craw,'LineWidth',0.7,'DisplayName','Raw');
h_fx=animatedline(ax2,'Color',Cfir,'LineWidth',1.2,'DisplayName','FIR');
h_kx=animatedline(ax2,'Color',Ckf, 'LineWidth',2.0,'DisplayName','Kalman');
ylim(ax2,[-2 W+2]); xlim(ax2,[0 10]);
set(ax2,'Color','w','GridAlpha',0.12,'FontSize',8,'Box','on'); grid(ax2,'on');
ylabel(ax2,'X (m)','FontSize',8);
title(ax2,'X Position Estimate','FontSize',9,'FontWeight','bold');
legend(ax2,'True','Raw','FIR','Kalman','Location','best','FontSize',7,'Box','off');

%% POSITION ERROR
ax3=nexttile; hold on;
h_perr=animatedline(ax3,'Color',Craw,'LineWidth',0.7,'DisplayName','Raw');
h_peff=animatedline(ax3,'Color',Cfir,'LineWidth',1.2,'DisplayName','FIR');
h_pekf=animatedline(ax3,'Color',Ckf, 'LineWidth',2.0,'DisplayName','Kalman');
ylim(ax3,[0 80]); xlim(ax3,[0 10]);
set(ax3,'Color','w','GridAlpha',0.12,'FontSize',8,'Box','on'); grid(ax3,'on');
ylabel(ax3,'Position Error (cm)','FontSize',8);
title(ax3,'2D Position Error','FontSize',9,'FontWeight','bold');
legend(ax3,'Raw','FIR','Kalman','Location','best','FontSize',7,'Box','off');

%% SERIAL MONITOR
ax4=nexttile([1 2]); axis off;
set(ax4,'Color',[0.06 0.06 0.06]);
title(ax4,'Serial Monitor  |  UWB 2D Localization','FontSize',9,'FontWeight','bold','Color','w');
cx=[0.01 0.10 0.20 0.31 0.42 0.53 0.64 0.75 0.87];
hdrs={'#','t(s)','True-X','True-Y','Raw-X','Raw-Y','KF-X','KF-Y','Err(cm)'};
hc={[0.4 1 0.4],[0.4 1 0.4],Ctr+0.2,Ctr+0.2,Craw,Craw,Ckf,Ckf,[0.95 0.80 0.15]};
for c=1:9
    text(ax4,cx(c),0.94,hdrs{c},'Units','normalized','Color',hc{c},...
        'FontName','Courier New','FontSize',8,'FontWeight','bold');
end
nrows=10; ry=fliplr(linspace(0.03,0.86,nrows));
hT=gobjects(nrows,9);
for r=1:nrows
    for c=1:9
        hT(r,c)=text(ax4,cx(c),ry(r),'','Units','normalized',...
            'Color',hc{c},'FontName','Courier New','FontSize',8);
    end
end
buf=repmat({'','','','','','','','',''},nrows,1);

%% ANIMATION
for k=1:N
    if ~ishandle(fig); break; end

    set(h_cov,'XData',true_x(1:k),'YData',true_y(1:k));
    set(h_dot,'XData',kf2d(1,k),  'YData',kf2d(2,k));

    % Pink lines from ALL 4 anchors to drone
    for a=1:nA
        set(h_pnk(a),'XData',[anchors(a,1) kf2d(1,k)],...
                      'YData',[anchors(a,2) kf2d(2,k)]);
    end

    addpoints(h_tx,t(k),true_x(k));
    addpoints(h_rx,t(k),est_raw(1,k));
    addpoints(h_fx,t(k),est_fir(1,k));
    addpoints(h_kx,t(k),kf2d(1,k));
    addpoints(h_perr,t(k),err_raw(k));
    addpoints(h_peff,t(k),err_fir(k));
    addpoints(h_pekf,t(k),err_kf(k));

    if t(k)>10
        xlim(ax2,[t(k)-10 t(k)]);
        xlim(ax3,[t(k)-10 t(k)]);
    end

    nr={sprintf('%d',k),sprintf('%.1f',t(k)),...
        sprintf('%.2f',true_x(k)),sprintf('%.2f',true_y(k)),...
        sprintf('%.2f',est_raw(1,k)),sprintf('%.2f',est_raw(2,k)),...
        sprintf('%.2f',kf2d(1,k)),sprintf('%.2f',kf2d(2,k)),...
        sprintf('%.1f',err_kf(k))};
    buf=[nr;buf(1:end-1,:)];
    for r=1:nrows
        for c=1:9, set(hT(r,c),'String',buf{r,c}); end
    end

    pause(0.08); drawnow;
end