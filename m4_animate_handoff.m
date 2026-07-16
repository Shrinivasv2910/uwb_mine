function m4_animate_handoff(cfg, net)
% M4_ANIMATE_HANDOFF  Two-drone, 8-anchor 3D swarm animation with live anchor
% HANDOFF. Drones start at opposite corners and split the mine. Each keeps an
% active set of cfg.activeK anchors; when the learned detector flags an active
% anchor as NLOS, the drone drops it and switches to the best clean anchor.
% Anti-thrash: confirmation + cooldown. Estimate trail smoothed FOR DISPLAY;
% RMSE text is computed from the raw estimate.

if nargin<2, S=load('nlos_detector.mat'); net=S.net; end
if nargin<1, cfg=m4_config_8anchor(); end

A=cfg.anchors; nA=cfg.nAnchors; D=2;
dcol=[0.1 0.5 0.9; 0.9 0.4 0.1];
seeds=[cfg.seedOffset+500+7, cfg.seedOffset+500+11];

truth=cell(D,1); est=cell(D,1); activeH=cell(D,1); switchH=cell(D,1);
for d=1:D
    path=half_mine_path(cfg, d, D);
    [truth{d},est{d},activeH{d},switchH{d}]=fly_handoff(cfg,net,path,seeds(d));
end
for d=1:D
    fprintf('Drone %d: handoffs = %d (%.1f/sec)\n', d, ...
        sum(any(switchH{d},2)), sum(any(switchH{d},2))/cfg.T);
end
N=size(truth{1},1);

% --- display-only smoothing of the estimate trail (RMSE text uses raw est) ---
estS=cell(D,1);
for d=1:D
    estS{d}=est{d}; w=11;
    for c=1:3, estS{d}(:,c)=movmean(est{d}(:,c),w); end
end

TAIL = N;  % show only the last TAIL samples of each trail (keeps it clean)

f=figure('Color','w','Position',[80 80 1000 740]);
ax=axes('Parent',f); hold(ax,'on'); grid(ax,'on'); view(ax,135,30);
axis(ax,[-1 cfg.W+1 -1 cfg.H+1 0 6.5]);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Two-Drone, 8-Anchor Swarm - Live NLOS Anchor Handoff');

% static scene first
patch('XData',[0 cfg.W cfg.W 0],'YData',[0 0 cfg.H cfg.H],'ZData',[0 0 0 0], ...
    'FaceColor',[0.93 0.93 0.9],'FaceAlpha',0.5,'EdgeColor',[0.6 0.6 0.6]);
for i=1:nA
    plot3(A(i,1),A(i,2),A(i,3),'k^','MarkerFaceColor',[0.2 0.2 0.7],'MarkerSize',9);
    text(A(i,1),A(i,2),A(i,3)+0.25,sprintf('A%d',i),'FontWeight','bold','FontSize',8);
end

% link handles (under drones)
hLink=gobjects(D,nA);
for d=1:D
    for i=1:nA, hLink(d,i)=plot3(ax,nan,nan,nan,'-','LineWidth',1.2); end
end

% trail + estimate + drone handles last (on top)
hTrail=gobjects(D,1); hEst=gobjects(D,1); hDrone=gobjects(D,1);
for d=1:D
    hTrail(d)=plot3(ax,nan,nan,nan,'-','Color',dcol(d,:),'LineWidth',2.2);
    hEst(d)  =plot3(ax,nan,nan,nan,':','Color',min(dcol(d,:)+0.25,1),'LineWidth',1.2);
    hDrone(d)=plot3(ax,nan,nan,nan,'o','MarkerSize',16, ...
        'MarkerFaceColor',dcol(d,:),'MarkerEdgeColor','k','LineWidth',1.5);
end
hTxt=text(ax,0.5,cfg.H-0.5,6.0,'','FontWeight','bold','BackgroundColor','w','EdgeColor','k','FontSize',9);

vname=fullfile(cfg.figdir,'anim_swarm_handoff.mp4');
vw=VideoWriter(vname,'MPEG-4'); vw.FrameRate=20; open(vw);

step=3; rmseNow=zeros(D,1); swCount=zeros(D,1);
for k=cfg.win:step:N
    lo=max(1,k-TAIL);                         % trailing window start
    for d=1:D
        tr=truth{d}; act=activeH{d}(k,:); sw=switchH{d}(k,:);
        % links
        for i=1:nA
            if act(i)
                if sw(i), col=[0.95 0.85 0.1]; lw=2.6;
                else      col=[0.2 0.7 0.3];  lw=1.6; end
                set(hLink(d,i),'XData',[tr(k,1) A(i,1)],'YData',[tr(k,2) A(i,2)], ...
                    'ZData',[tr(k,3) A(i,3)],'Color',col,'LineWidth',lw);
            else
                set(hLink(d,i),'XData',nan,'YData',nan,'ZData',nan);
            end
        end
        % trail (clean truth, last TAIL samples), smoothed estimate, drone
        set(hTrail(d),'XData',tr(lo:k,1),'YData',tr(lo:k,2),'ZData',tr(lo:k,3));
        set(hEst(d),  'XData',estS{d}(lo:k,1),'YData',estS{d}(lo:k,2),'ZData',estS{d}(lo:k,3));
        set(hDrone(d),'XData',tr(k,1),'YData',tr(k,2),'ZData',tr(k,3));
        rmseNow(d)=sqrt(mean(sum((est{d}(cfg.win:k,:)-tr(cfg.win:k,:)).^2,2)))*100;
        swCount(d)=sum(any(switchH{d}(1:k,:),2));
    end
    set(hTxt,'String',sprintf(['t = %4.1f s\n' ...
        'D1: RMSE %.1f cm | %d handoffs\n' ...
        'D2: RMSE %.1f cm | %d handoffs'], ...
        (k-1)*cfg.dt, rmseNow(1),swCount(1), rmseNow(2),swCount(2)));
    drawnow;
    fr=getframe(f); img=fr.cdata; img=img(1:floor(end/2)*2,1:floor(end/2)*2,:);
    writeVideo(vw,img);
end
close(vw);
fprintf('Saved %s\n',vname);
end

% ===================================================================
function path = half_mine_path(cfg, d, D)
N=cfg.N; margin=1.5;
laneY=linspace(margin, cfg.H-margin, cfg.nLanes);
if d==1
    xLo=margin; xHi=cfg.W/2;        order=1:cfg.nLanes;       startFlip=false;
else
    xLo=cfg.W/2; xHi=cfg.W-margin;  order=cfg.nLanes:-1:1;    startFlip=true;
end
wp=[]; flip=startFlip;
for li=1:numel(order)
    k=order(li);
    if ~flip, wp=[wp; xLo laneY(k); xHi laneY(k)]; %#ok<AGROW>
    else,     wp=[wp; xHi laneY(k); xLo laneY(k)]; end %#ok<AGROW>
    flip=~flip;
end
dd=[0; cumsum(sqrt(sum(diff(wp).^2,2)))];
s=linspace(0,dd(end),N).';
x=interp1(dd,wp(:,1),s); y=interp1(dd,wp(:,2),s);
t=(0:N-1).'*cfg.dt;
zlo=cfg.alt_min+0.4*(d-1); zhi=cfg.alt_max+0.4*(d-1);
z=zlo+(zhi-zlo)*0.5*(1-cos(2*pi*t/cfg.T*2));
path=[x y z];
end

% ===================================================================
function [truth, est, activeH, switchH] = fly_handoff(cfg, net, path, seed)
rng(seed,'twister');
N=size(path,1); nA=cfg.nAnchors; A=cfg.anchors; dt=cfg.dt; K=cfg.activeK;
t=(0:N-1).'*dt; truth=path;

rng_true=zeros(N,nA);
for i=1:nA, rng_true(:,i)=vecnorm(path-A(i,:),2,2); end
pf=m4_nlos_field(cfg,path); nlos=rand(N,nA)<pf;
los=cfg.sigma_los*randn(N,nA);
bias=(cfg.nlos_amp(1)+diff(cfg.nlos_amp)*rand(N,nA)).*nlos;
phase=2*pi*rand(1,nA); clk=cfg.clk_amp*sin(2*pi*cfg.clk_freq*t+phase);
rng_raw=max(round((rng_true+los+bias+clk)/cfg.quant)*cfg.quant,0.1);
resid=baseline_resid(cfg,path,rng_raw);

diagL=sqrt(cfg.W^2+cfg.H^2+cfg.ceil_z^2);
rawN=rng_raw/diagL; drw=[zeros(1,nA); diff(rng_raw)];
pHat=zeros(N,nA); ks=cfg.win:N;
for i=1:nA
    seqs={};
    for k=ks
        idx=(k-cfg.win+1):k;
        seqs{end+1,1}=[rawN(idx,i).'; resid(idx,i).'; drw(idx,i).']; %#ok<AGROW>
    end
    [~,scr]=classify(net,seqs); pHat(ks,i)=scr(:,2);
end

d0=vecnorm(A-path(1,:),2,2); [~,ord]=sort(d0,'ascend');
active=false(1,nA); active(ord(1:K))=true;

nlosStreak=zeros(1,nA); cooldown=zeros(1,nA);
CONFIRM=3; COOL=15;

F=[eye(3) dt*eye(3); zeros(3) eye(3)];
G=[0.5*dt^2*eye(3); dt*eye(3)]; Q=G*G.'*0.6; Rmeas=0.04; pThr=0.5;
x=[path(1,:).';0;0;0]; P=eye(6); est=zeros(N,3);
activeH=false(N,nA); switchH=false(N,nA);

for k=1:N
    x=F*x; P=F*P*F.'+Q; p=x(1:3).';

    cooldown=max(cooldown-1,0);
    flagged=pHat(k,:)>=pThr;
    nlosStreak(flagged)=nlosStreak(flagged)+1;
    nlosStreak(~flagged)=0;

    swapped=false(1,nA);
    badActive=active & (nlosStreak>=CONFIRM) & (cooldown==0);
    if any(badActive)
        active(badActive)=false; cooldown(badActive)=COOL;
        need=K-nnz(active);
        if need>0
            cand=find(~active & ~flagged & cooldown==0);
            if ~isempty(cand)
                dc=vecnorm(A(cand,:)-p,2,2); [~,oc]=sort(dc,'ascend');
                take=cand(oc(1:min(need,numel(cand))));
                active(take)=true; swapped(take)=true; cooldown(take)=COOL;
            end
        end
    end
    if nnz(active)<3
        [~,o2]=sort(pHat(k,:),'ascend');
        for jj=o2
            if ~active(jj), active(jj)=true; swapped(jj)=true; end
            if nnz(active)>=3, break; end
        end
    end

    activeH(k,:)=active; switchH(k,:)=swapped;

    for i=find(active)
        dpred=norm(p-A(i,:)); innov=rng_raw(k,i)-dpred;
        if abs(innov)>3.0, continue; end
        Hi=zeros(1,6); Hi(1:3)=(p-A(i,:))/max(dpred,1e-6);
        S=Hi*P*Hi.'+Rmeas; K2=(P*Hi.')/S;
        x=x+K2*innov; P=(eye(6)-K2*Hi)*P; p=x(1:3).';
    end
    P=(P+P.')/2; P=min(P,1e4); est(k,:)=x(1:3).';
end
end

% ===================================================================
function resid = baseline_resid(cfg, truth, rng_raw)
N=size(truth,1); nA=cfg.nAnchors; A=cfg.anchors; dt=cfg.dt;
F=[eye(3) dt*eye(3); zeros(3) eye(3)];
G=[0.5*dt^2*eye(3); dt*eye(3)]; Q=G*G.'*0.05; Rfix=0.04;
x=[truth(1,:).';0;0;0]; P=eye(6); resid=zeros(N,nA);
for k=1:N
    x=F*x; P=F*P*F.'+Q; p=x(1:3).';
    for i=1:nA
        d=norm(p-A(i,:)); innov=rng_raw(k,i)-d; resid(k,i)=innov;
        Hi=zeros(1,6); Hi(1:3)=(p-A(i,:))/max(d,1e-6);
        S=Hi*P*Hi.'+Rfix; K=(P*Hi.')/S;
        x=x+K*innov; P=(eye(6)-K*Hi)*P; p=x(1:3).';
    end
end
end