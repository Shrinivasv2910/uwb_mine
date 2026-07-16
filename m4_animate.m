function m4_animate(cfg, net)
% M4_ANIMATE  3D animation of the drone flying the mine, with live UWB links
% coloured by the detector's NLOS belief. Saves an MP4 (and GIF fallback).
%
%   m4_animate(cfg, net)
%
% Green link  = detector thinks LOS (trusted)
% Red  link   = detector thinks NLOS (down-weighted)

s = cfg.seedOffset + 500 + 3;
sim = m4_simulate_mission(cfg, s);
diagL = sqrt(cfg.W^2+cfg.H^2+cfg.ceil_z^2);

% Precompute detector P(NLOS) for all anchors
pHat = zeros(cfg.N,cfg.nAnchors);
rawN = sim.rng_raw/diagL; drw=[zeros(1,cfg.nAnchors); diff(sim.rng_raw)];
for i=1:cfg.nAnchors
    ks=cfg.win:cfg.N; seqs={};
    for k=ks
        idx=(k-cfg.win+1):k;
        seqs{end+1,1}=[rawN(idx,i).'; sim.resid(idx,i).'; drw(idx,i).']; %#ok<AGROW>
    end
    [~,scr]=classify(net,seqs); pHat(ks,i)=scr(:,2);
end

A=cfg.anchors; truth=sim.truth; est=sim.ekf_pos;

f=figure('Color','w','Position',[80 80 960 720]);   % both even
ax=axes('Parent',f); hold(ax,'on'); grid(ax,'on'); view(ax,135,28);
axis(ax,[-1 cfg.W+1 -1 cfg.H+1 0 cfg.alt_max+1]);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('UWB-Anchored Drone Localisation with Live NLOS Detection');

% Draw mine floor and anchors
patch('XData',[0 cfg.W cfg.W 0],'YData',[0 0 cfg.H cfg.H], ...
    'ZData',[0 0 0 0],'FaceColor',[0.93 0.93 0.9],'FaceAlpha',0.6, ...
    'EdgeColor',[0.6 0.6 0.6]);
for i=1:cfg.nAnchors
    plot3(A(i,1),A(i,2),A(i,3),'k^','MarkerFaceColor',[0.2 0.2 0.7], ...
        'MarkerSize',10);
    text(A(i,1),A(i,2),A(i,3)+0.4,sprintf('A%d',i),'FontWeight','bold');
end

% Trajectory trail (ground truth) and estimate
hTrail = plot3(ax,nan,nan,nan,'-','Color',[0.4 0.4 0.4],'LineWidth',1);
hEst   = plot3(ax,nan,nan,nan,'-','Color',[0.2 0.7 0.3],'LineWidth',1.2);
hDrone = plot3(ax,nan,nan,nan,'o','MarkerSize',12, ...
    'MarkerFaceColor',[0.1 0.5 0.9],'MarkerEdgeColor','k');

% UWB link lines (one per anchor)
hLink = gobjects(cfg.nAnchors,1);
for i=1:cfg.nAnchors
    hLink(i)=plot3(ax,nan,nan,nan,'-','LineWidth',1.5);
end
hTxt = text(ax, 1, cfg.H-1, cfg.alt_max+0.5, '', 'FontWeight','bold', ...
    'BackgroundColor','w','EdgeColor','k');

% Video writer
vname = fullfile(cfg.figdir,'drone_uwb_animation.mp4');
useVid = true;
try
    vw=VideoWriter(vname,'MPEG-4'); vw.FrameRate=20; open(vw);
catch
    useVid=false;
    warning('MPEG-4 unavailable; falling back to GIF.');
    gifname=fullfile(cfg.figdir,'drone_uwb_animation.gif');
end

step = 3;   % render every 3rd sample to keep file size sane
for k=cfg.win:step:cfg.N
    set(hTrail,'XData',truth(1:k,1),'YData',truth(1:k,2),'ZData',truth(1:k,3));
    set(hEst,  'XData',est(1:k,1),  'YData',est(1:k,2),  'ZData',est(1:k,3));
    set(hDrone,'XData',truth(k,1),'YData',truth(k,2),'ZData',truth(k,3));
    for i=1:cfg.nAnchors
        set(hLink(i),'XData',[truth(k,1) A(i,1)], ...
            'YData',[truth(k,2) A(i,2)], ...
            'ZData',[truth(k,3) A(i,3)]);
        % colour by detector belief: green (LOS) -> red (NLOS)
        a=pHat(k,i);
        set(hLink(i),'Color',[a 0.6*(1-a) 0.2*(1-a)]);
    end
    instErr = norm(est(k,:)-truth(k,:))*100;
    set(hTxt,'String',sprintf('t = %4.1f s\nNLOS prob = %.2f\nerr = %.1f cm', ...
        sim.t(k), sim.p_nlos(k), instErr));
    drawnow;
    if useVid
        fr = getframe(f);
        img = fr.cdata;
        % force even dimensions for H.264
        img = img(1:floor(end/2)*2, 1:floor(end/2)*2, :);
        writeVideo(vw, img);
    else
        [im,map]=rgb2ind(frame2im(getframe(f)),256);
        if k==cfg.win
            imwrite(im,map,gifname,'gif','LoopCount',inf,'DelayTime',0.05);
        else
            imwrite(im,map,gifname,'gif','WriteMode','append','DelayTime',0.05);
        end
    end
end
if useVid; close(vw); fprintf('Saved %s\n',vname);
else; fprintf('Saved %s\n',gifname); end
end