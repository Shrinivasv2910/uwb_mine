function m4_animate_hardreject(cfg, net)
% M4_ANIMATE_HARDREJECT  3D animation of one drone flying the mine, links
% coloured by the detector's NLOS belief, anchors that get HARD-REJECTED
% (P>=0.5) drawn dashed/faded. Live 3D RMSE counter. Saves an MP4.
%
% Same visual style as m4_animate.m. Uses the proven policy: learned detect
% + hard reject.

if nargin<2, S=load('nlos_detector.mat'); net=S.net; end
if nargin<1, cfg=m4_config_plan(); end

s = cfg.seedOffset + 500 + 3;
path = m4_trajectory(cfg);
[truth, est, pHat, useHist] = fly_hardreject(cfg, net, path, s);
diagL = sqrt(cfg.W^2+cfg.H^2+cfg.ceil_z^2);
A=cfg.anchors; N=size(truth,1);

f=figure('Color','w','Position',[80 80 960 720]);
ax=axes('Parent',f); hold(ax,'on'); grid(ax,'on'); view(ax,135,28);
axis(ax,[-1 cfg.W+1 -1 cfg.H+1 0 cfg.alt_max+1]);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Single-Drone UWB Localisation - Learned NLOS Detection + Hard Reject');

patch('XData',[0 cfg.W cfg.W 0],'YData',[0 0 cfg.H cfg.H], ...
    'ZData',[0 0 0 0],'FaceColor',[0.93 0.93 0.9],'FaceAlpha',0.6, ...
    'EdgeColor',[0.6 0.6 0.6]);
for i=1:cfg.nAnchors
    plot3(A(i,1),A(i,2),A(i,3),'k^','MarkerFaceColor',[0.2 0.2 0.7],'MarkerSize',10);
    text(A(i,1),A(i,2),A(i,3)+0.4,sprintf('A%d',i),'FontWeight','bold');
end

hTrail = plot3(ax,nan,nan,nan,'-','Color',[0.4 0.4 0.4],'LineWidth',1);
hEst   = plot3(ax,nan,nan,nan,'-','Color',[0.2 0.7 0.3],'LineWidth',1.2);
hDrone = plot3(ax,nan,nan,nan,'o','MarkerSize',12, ...
    'MarkerFaceColor',[0.1 0.5 0.9],'MarkerEdgeColor','k');
hLink = gobjects(cfg.nAnchors,1);
for i=1:cfg.nAnchors
    hLink(i)=plot3(ax,nan,nan,nan,'-','LineWidth',1.5);
end
hTxt = text(ax, 1, cfg.H-1, cfg.alt_max+0.5, '', 'FontWeight','bold', ...
    'BackgroundColor','w','EdgeColor','k');

vname = fullfile(cfg.figdir,'anim_single_hardreject.mp4');
vw=VideoWriter(vname,'MPEG-4'); vw.FrameRate=20; open(vw);

step=3;
runErr=zeros(N,1);
for k=cfg.win:step:N
    set(hTrail,'XData',truth(1:k,1),'YData',truth(1:k,2),'ZData',truth(1:k,3));
    set(hEst,  'XData',est(1:k,1),  'YData',est(1:k,2),  'ZData',est(1:k,3));
    set(hDrone,'XData',truth(k,1),'YData',truth(k,2),'ZData',truth(k,3));
    for i=1:cfg.nAnchors
        a=pHat(k,i);
        set(hLink(i),'XData',[truth(k,1) A(i,1)], ...
            'YData',[truth(k,2) A(i,2)],'ZData',[truth(k,3) A(i,3)], ...
            'Color',[a 0.6*(1-a) 0.2*(1-a)]);
        if ~useHist(k,i)                  % rejected this step -> dashed/faded
            set(hLink(i),'LineStyle','--','LineWidth',0.8);
        else
            set(hLink(i),'LineStyle','-','LineWidth',1.5);
        end
    end
    instErr = norm(est(k,:)-truth(k,:))*100;
    runErr(k)= sqrt(mean(sum((est(cfg.win:k,:)-truth(cfg.win:k,:)).^2,2)))*100;
    nRej = sum(~useHist(k,:));
    set(hTxt,'String',sprintf('t = %4.1f s\ninst err = %.1f cm\nrunning RMSE = %.1f cm\nanchors rejected: %d/%d', ...
        (k-1)*cfg.dt, instErr, runErr(k), nRej, cfg.nAnchors));
    drawnow;
    fr=getframe(f); img=fr.cdata;
    img=img(1:floor(end/2)*2, 1:floor(end/2)*2, :);
    writeVideo(vw,img);
end
close(vw); fprintf('Saved %s  (final RMSE %.1f cm)\n', vname, runErr(k));
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