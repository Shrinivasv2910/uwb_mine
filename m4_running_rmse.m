function m4_running_rmse(cfg, net)
% M4_RUNNING_RMSE  Running 3D RMSE over one mission for all four policies,
% with the non-stationary NLOS profile shaded behind. Matches the Months 1-3
% running-RMSE figure style. Saves fig_running_rmse.png.

s = cfg.seedOffset + 500 + 1;          % a representative test mission
sim = m4_simulate_mission(cfg, s);
diagL = sqrt(cfg.W^2 + cfg.H^2 + cfg.ceil_z^2);

% ---- detector P(NLOS) for every anchor/step ----
pHat = zeros(cfg.N, cfg.nAnchors);
rawN = sim.rng_raw/diagL; drw=[zeros(1,cfg.nAnchors); diff(sim.rng_raw)];
for i=1:cfg.nAnchors
    ks=cfg.win:cfg.N; seqs={};
    for k=ks
        idx=(k-cfg.win+1):k;
        seqs{end+1,1}=[rawN(idx,i).'; sim.resid(idx,i).'; drw(idx,i).']; %#ok<AGROW>
    end
    [~,scr]=classify(net,seqs); pHat(ks,i)=scr(:,2);
end

% ---- per-timestep error for each policy ----
eFix = ekf_err(cfg, sim, 'fixed',     pHat);
eHeu = ekf_err(cfg, sim, 'heuristic', pHat);
eLrn = ekf_err(cfg, sim, 'learned',   pHat);
eOra = ekf_err(cfg, sim, 'oracle',    pHat);

% ---- running (cumulative) RMSE ----
rr = @(e) sqrt(cumsum(e.^2)./(1:numel(e)).');
f=figure('Color','w','Position',[100 100 900 480]); hold on;

% shade NLOS profile behind (scaled to the y-range)
yl = 100;   % cm ceiling for shading reference
area(sim.t, 100*sim.p_nlos*yl/100, 'FaceColor',[1 0.9 0.9], ...
    'EdgeColor','none','FaceAlpha',0.6,'BaseValue',0);

plot(sim.t, 100*rr(eFix), 'Color',[0.5 0.5 0.5],'LineWidth',1.6);
plot(sim.t, 100*rr(eHeu), 'Color',[0.85 0.5 0.2],'LineWidth',1.4,'LineStyle','-.');
plot(sim.t, 100*rr(eLrn), 'Color',[0.2 0.6 0.3],'LineWidth',2.0);
plot(sim.t, 100*rr(eOra), 'Color',[0.2 0.3 0.7],'LineWidth',1.6,'LineStyle','--');

xlabel('Time (s)'); ylabel('Running 3D RMSE (cm)');
title('Running RMSE over 60 s mission - non-stationary NLOS');
legend({'NLOS profile (shaded)','Fixed-R','Heuristic-R','Learned-R','Oracle-R'}, ...
    'Location','northwest');
grid on; ylim([0 yl]); xlim([0 cfg.T]);
exportgraphics(f,fullfile(cfg.figdir,'fig_running_rmse.png'),'Resolution',200);
fprintf('Saved fig_running_rmse.png\n');
end

% ===================================================================
function err = ekf_err(cfg, sim, mode, pHat)
% Same EKF as m4_evaluate's run_ekf_rmse, but returns per-timestep 3D error.
N=cfg.N; nA=cfg.nAnchors; A=cfg.anchors; dt=cfg.dt;
F=[eye(3) dt*eye(3); zeros(3) eye(3)];
G=[0.5*dt^2*eye(3); dt*eye(3)]; Q=G*G.'*0.6;
Rmeas=0.04; pThr=0.5;
histLen=15; innovHist=nan(histLen,nA); hp=1;

x=[sim.truth(1,:).';0;0;0]; P=eye(6);
err=zeros(N,1);
for k=1:N
    x=F*x; P=F*P*F.'+Q; p=x(1:3).';
    for i=1:nA
        d=norm(p-A(i,:)); innov=sim.rng_raw(k,i)-d;
        switch mode
            case 'fixed',     useIt=true;
            case 'heuristic'
                h=innovHist(:,i); h=h(~isnan(h));
                if numel(h)<5, useIt=true;
                else
                    med=median(h); madv=1.4826*median(abs(h-med))+1e-3;
                    useIt=abs(innov-med)/madv<=3.5;
                end
            case 'learned',   useIt=pHat(k,i)<pThr;
            case 'oracle',    useIt=~sim.nlos(k,i);
        end
        innovHist(hp,i)=innov;
        if ~useIt, continue; end
        Hi=zeros(1,6); Hi(1:3)=(p-A(i,:))/max(d,1e-6);
        S=Hi*P*Hi.'+Rmeas; K=(P*Hi.')/S;
        x=x+K*innov; P=(eye(6)-K*Hi)*P; p=x(1:3).';
    end
    hp=mod(hp,histLen)+1;
    err(k)=norm(x(1:3).'-sim.truth(k,:));
end
end