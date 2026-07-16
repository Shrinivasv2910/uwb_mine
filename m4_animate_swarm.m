function m4_animate_swarm(cfg, net)
% M4_ANIMATE_SWARM  Two-drone 3D swarm animation. Each drone flies a subset of
% lanes and runs the learned detector + hard-reject INDEPENDENTLY. Both drones'
% UWB links shown live, coloured by each drone's own NLOS belief. Live per-drone
% RMSE. Saves an MP4. Demonstrates the proven detector in a swarm setting.

if nargin<2, S=load('nlos_detector.mat'); net=S.net; end
if nargin<1, cfg=m4_config_plan(); end

A=cfg.anchors; nA=cfg.nAnchors;
seeds=[cfg.seedOffset+500+7, cfg.seedOffset+500+11];   % one per drone
D=2; dcol=[0.1 0.5 0.9; 0.9 0.4 0.1];                  % drone colours

truth=cell(D,1); est=cell(D,1); pHat=cell(D,1); useH=cell(D,1);
for d=1:D
    path=lane_subset_path(cfg, d, D);
    [truth{d},est{d},pHat{d},useH{d}]=fly_hardreject(cfg,net,path,seeds(d));
end
N=size(truth{1},1);

f=figure('Color','w','Position',[80 80 960 720]);
ax=axes('Parent',f); hold(ax,'on'); grid(ax,'on'); view(ax,135,28);
axis(ax,[-1 cfg.W+1 -1 cfg.H+1 0 cfg.alt_max+1.5]);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Two-Drone Swarm - Independent Learned NLOS Detection + Hard Reject');

patch('XData',[0 cfg.W cfg.W 0],'YData',[0 0 cfg.H cfg.H], ...
    'ZData',[0 0 0 0],'FaceColor',[0.93 0.93 0.9],'FaceAlpha',0.6, ...
    'EdgeColor',[0.6 0.6 0.6]);
for i=1:nA
    plot3(A(i,1),A(i,2),A(i,3),'k^','MarkerFaceColor',[0.2 0.2 0.7],'MarkerSize',10);
    text(A(i,1),A(i,2),A(i,3)+0.4,sprintf('A%d',i),'FontWeight','bold');
end

hTrail=gobjects(D,1); hEst=gobjects(D,1); hDrone=gobjects(D,1);
hLink=gobjects(D,nA);
for d=1:D
    hTrail(d)=plot3(ax,nan,nan,nan,'-','Color',[0.6 0.6 0.6],'LineWidth',0.8);
    hEst(d)  =plot3(ax,nan,nan,nan,'-','Color',dcol(d,:),'LineWidth',1.2);
    hDrone(d)=plot3(ax,nan,nan,nan,'o','MarkerSize',12, ...
        'MarkerFaceColor',dcol(d,:),'MarkerEdgeColor','k');
    for i=1:nA
        hLink(d,i)=plot3(ax,nan,nan,nan,'-','LineWidth',1.2);
    end
end
hTxt=text(ax,1,cfg.H-1,cfg.alt_max+1,'','FontWeight','bold', ...
    'BackgroundColor','w','EdgeColor','k');

vname=fullfile(cfg.figdir,'anim_swarm_hardreject.mp4');
vw=VideoWriter(vname,'MPEG-4'); vw.FrameRate=20; open(vw);

step=3; rmseNow=zeros(D,1);
for k=cfg.win:step:N
    for d=1:D
        tr=truth{d}; es=est{d};
        set(hTrail(d),'XData',tr(1:k,1),'YData',tr(1:k,2),'ZData',tr(1:k,3));
        set(hEst(d),  'XData',es(1:k,1),'YData',es(1:k,2),'ZData',es(1:k,3));
        set(hDrone(d),'XData',tr(k,1),'YData',tr(k,2),'ZData',tr(k,3));
        for i=1:nA
            a=pHat{d}(k,i);
            set(hLink(d,i),'XData',[tr(k,1) A(i,1)], ...
                'YData',[tr(k,2) A(i,2)],'ZData',[tr(k,3) A(i,3)], ...
                'Color',[a 0.6*(1-a) 0.2*(1-a)]);
            if ~useH{d}(k,i)
                set(hLink(d,i),'LineStyle','--','LineWidth',0.6);
            else
                set(hLink(d,i),'LineStyle','-','LineWidth',1.2);
            end
        end
        rmseNow(d)=sqrt(mean(sum((es(cfg.win:k,:)-tr(cfg.win:k,:)).^2,2)))*100;
    end
    set(hTxt,'String',sprintf('t = %4.1f s\nDrone 1 RMSE = %.1f cm\nDrone 2 RMSE = %.1f cm', ...
        (k-1)*cfg.dt, rmseNow(1), rmseNow(2)));
    drawnow;
    fr=getframe(f); img=fr.cdata;
    img=img(1:floor(end/2)*2, 1:floor(end/2)*2, :);
    writeVideo(vw,img);
end
close(vw);
fprintf('Saved %s  (Drone1 %.1f cm, Drone2 %.1f cm)\n', vname, rmseNow(1), rmseNow(2));
end

% ===================================================================
function path = lane_subset_path(cfg, d, D)
% Drone d flies lanes d, d+D, d+2D, ... (interleaved). Altitude band staggered
% per drone so the two are vertically separated and safe.
N=cfg.N; margin=1.5;
xLo=margin; xHi=cfg.W-margin;
laneY=linspace(margin, cfg.H-margin, cfg.nLanes);
myLanes=d:D:cfg.nLanes;
wp=[]; flip=false;
for li=1:numel(myLanes)
    k=myLanes(li);
    if ~flip, wp=[wp; xLo laneY(k); xHi laneY(k)]; %#ok<AGROW>
    else,     wp=[wp; xHi laneY(k); xLo laneY(k)]; end %#ok<AGROW>
    flip=~flip;
end
dd=[0; cumsum(sqrt(sum(diff(wp).^2,2)))];
s=linspace(0,dd(end),N).';
x=interp1(dd,wp(:,1),s); y=interp1(dd,wp(:,2),s);
t=(0:N-1).'*cfg.dt;
zlo=cfg.alt_min+0.5*(d-1); zhi=cfg.alt_max+0.5*(d-1);
z=zlo+(zhi-zlo)*0.5*(1-cos(2*pi*t/cfg.T*2));
path=[x y z];
end

% ===================================================================
function [truth, est, pHat, useHist] = fly_hardreject(cfg, net, path, seed)
rng(seed,'twister');
N=size(path,1); nA=cfg.nAnchors; A=cfg.anchors; dt=cfg.dt;
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
F=[eye(3) dt*eye(3); zeros(3) eye(3)];
G=[0.5*dt^2*eye(3); dt*eye(3)]; Q=G*G.'*0.6; Rmeas=0.04; pThr=0.5;
x=[path(1,:).';0;0;0]; P=eye(6); est=zeros(N,3); useHist=true(N,nA);
for k=1:N
    x=F*x; P=F*P*F.'+Q; p=x(1:3).';
    useIt=pHat(k,:)<pThr;
    if nnz(useIt)<3, [~,ord]=sort(pHat(k,:),'ascend'); useIt(:)=false; useIt(ord(1:3))=true; end
    useHist(k,:)=useIt;
    for i=1:nA
        if ~useIt(i), continue; end
        dpred=norm(p-A(i,:)); innov=rng_raw(k,i)-dpred;
        if abs(innov)>3.0, continue; end
        Hi=zeros(1,6); Hi(1:3)=(p-A(i,:))/max(dpred,1e-6);
        S=Hi*P*Hi.'+Rmeas; K=(P*Hi.')/S;
        x=x+K*innov; P=(eye(6)-K*Hi)*P; p=x(1:3).';
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