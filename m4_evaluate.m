function results = m4_evaluate(cfg, net)
% M4_EVALUATE  Detector + system-level evaluation on held-out test missions.
%   results = m4_evaluate(cfg, net)
%
% Produces:
%   (1) Detector metrics: ROC/AUC, confusion matrix
%   (2) System metrics: position RMSE under four measurement-trust policies
%         - Fixed-R     : baseline, uses every anchor every step
%         - Heuristic-R : robust per-anchor MAD z-score rejection (classical)
%         - Learned-R   : reject anchor when detector P(NLOS) >= threshold
%         - Oracle-R    : reject anchor when TRUE NLOS label set (upper bound)
%
% All four use the SAME EKF; only the per-anchor accept/reject rule differs.
% That isolates the value of detection from the value of filtering.

    testSeeds = cfg.seedOffset + 500 + (1:cfg.nTestSeeds);

    diagL = sqrt(cfg.W^2 + cfg.H^2 + cfg.ceil_z^2);

    allScore = [];   % detector P(NLOS)
    allLabel = [];   % true NLOS

    rmseFix = []; rmseHeu = []; rmseLrn = []; rmseOra = [];

    for s = testSeeds
        sim = m4_simulate_mission(cfg, s);

        % ---- Run detector over this mission to get P(NLOS) per anchor/step ----
        pHat = zeros(cfg.N, cfg.nAnchors);   % default LOS for first win-1 steps
        rawN = sim.rng_raw / diagL;
        drw  = [zeros(1,cfg.nAnchors); diff(sim.rng_raw)];
        for i = 1:cfg.nAnchors
            seqs = {};
            ks = cfg.win:cfg.N;
            for k = ks
                idx = (k-cfg.win+1):k;
                seqs{end+1,1} = [rawN(idx,i).'; sim.resid(idx,i).'; drw(idx,i).']; %#ok<AGROW>
            end
            [~, scr] = classify(net, seqs);     % scr(:,2) = P(NLOS)
            pHat(ks, i) = scr(:,2);
        end

        allScore = [allScore; pHat(:)];            %#ok<AGROW>
        allLabel = [allLabel; double(sim.nlos(:))];%#ok<AGROW>

        % ---- Four EKF passes with different per-anchor accept/reject rule ----
        rmseFix(end+1,1) = run_ekf_rmse(cfg, sim, 'fixed',     pHat); %#ok<AGROW>
        rmseHeu(end+1,1) = run_ekf_rmse(cfg, sim, 'heuristic', pHat); %#ok<AGROW>
        rmseLrn(end+1,1) = run_ekf_rmse(cfg, sim, 'learned',   pHat); %#ok<AGROW>
        rmseOra(end+1,1) = run_ekf_rmse(cfg, sim, 'oracle',    pHat); %#ok<AGROW>
    end

    % ---------- Detector metrics ----------
    [Xroc,Yroc,~,AUC] = perfcurve(allLabel, allScore, 1);
    thr = 0.5;
    pred = allScore >= thr;
    TP = sum(pred & allLabel); FP = sum(pred & ~allLabel);
    FN = sum(~pred & allLabel);
    precision = TP/max(TP+FP,1);
    recall    = TP/max(TP+FN,1);
    f1        = 2*precision*recall/max(precision+recall,eps);

    fprintf('\n=== Detector metrics (threshold %.2f) ===\n', thr);
    fprintf('AUC = %.4f | Precision = %.3f | Recall = %.3f | F1 = %.3f\n', ...
        AUC, precision, recall, f1);

    % ---------- System metrics ----------
    fprintf('\n=== Position RMSE (cm), N=%d test missions ===\n', cfg.nTestSeeds);
    fprintf('Fixed-R     : %.2f +/- %.2f\n', 100*mean(rmseFix), 100*std(rmseFix));
    fprintf('Heuristic-R : %.2f +/- %.2f\n', 100*mean(rmseHeu), 100*std(rmseHeu));
    fprintf('Learned-R   : %.2f +/- %.2f\n', 100*mean(rmseLrn), 100*std(rmseLrn));
    fprintf('Oracle-R    : %.2f +/- %.2f\n', 100*mean(rmseOra), 100*std(rmseOra));

    % Paired Wilcoxon: learned vs heuristic
    pW = signrank(rmseLrn, rmseHeu);
    fprintf('Wilcoxon learned vs heuristic: p = %.4g\n', pW);

    % ---------- Figures ----------
    plot_roc(Xroc,Yroc,AUC,cfg);
    plot_confusion(allLabel,pred,cfg);
    plot_rmse_bars(rmseFix,rmseHeu,rmseLrn,rmseOra,cfg);
    plot_example_mission(cfg,net);  % time-series detector trace

    results = struct('AUC',AUC,'precision',precision,'recall',recall,'f1',f1, ...
        'rmseFix',rmseFix,'rmseHeu',rmseHeu,'rmseLrn',rmseLrn,'rmseOra',rmseOra, ...
        'pWilcoxon',pW);
    save(fullfile(pwd,'m4_results.mat'),'results');
end

% ===================================================================
function rmse = run_ekf_rmse(cfg, sim, mode, pHat)
% One EKF pass. NLOS handling = REJECT flagged anchors entirely.
% No innovation gate (a tight gate rejects good LOS anchors once P shrinks
% and causes catastrophic dead-reckoning divergence).
% Heuristic baseline = robust per-anchor MAD z-score on a rolling window of
% that anchor's own innovations (does not collapse when the estimate drifts).
    N=cfg.N; nA=cfg.nAnchors; A=cfg.anchors; dt=cfg.dt;
    F=[eye(3) dt*eye(3); zeros(3) eye(3)];
    G=[0.5*dt^2*eye(3); dt*eye(3)]; Q=G*G.'*0.6;   % process noise lets the
                                                   % prediction bridge dropped anchors
    Rmeas = 0.04;                  % DWM1001 calibrated variance (Month 3)
    pThr  = 0.5;                   % detector decision threshold

    histLen   = 15;                % rolling window length per anchor
    innovHist = nan(histLen, nA);  % recent innovations per anchor
    hp        = 1;                 % circular write pointer

    x=[sim.truth(1,:).';0;0;0]; P=eye(6);
    est=zeros(N,3);
    for k=1:N
        x=F*x; P=F*P*F.'+Q;
        p=x(1:3).';
        for i=1:nA
            d=norm(p-A(i,:)); innov=sim.rng_raw(k,i)-d;

            % decide whether to USE this anchor this step
            switch mode
                case 'fixed'
                    useIt = true;
                case 'heuristic'                   % robust MAD z-score vs own history
                    h = innovHist(:,i); h = h(~isnan(h));
                    if numel(h) < 5
                        useIt = true;              % warm-up: accept
                    else
                        med  = median(h);
                        madv = 1.4826*median(abs(h-med)) + 1e-3;
                        useIt = abs(innov - med)/madv <= 3.5;   % 3.5-sigma robust gate
                    end
                case 'learned'                     % reject if detector says NLOS
                    useIt = pHat(k,i) < pThr;
                case 'oracle'                      % reject true NLOS
                    useIt = ~sim.nlos(k,i);
            end

            % record this anchor's innovation into the rolling buffer
            innovHist(hp,i) = innov;

            if ~useIt, continue; end               % skip update for this anchor

            Hi=zeros(1,6); Hi(1:3)=(p-A(i,:))/max(d,1e-6);
            S=Hi*P*Hi.'+Rmeas;
            K=(P*Hi.')/S;
            x=x+K*innov; P=(eye(6)-K*Hi)*P; p=x(1:3).';
        end
        hp = mod(hp, histLen) + 1;                 % advance once per timestep
        est(k,:)=x(1:3).';
    end
    rmse = sqrt(mean(sum((est-sim.truth).^2,2)));   % 3D RMSE (m)
end

% ===================================================================
function plot_roc(X,Y,AUC,cfg)
    f=figure('Color','w','Position',[100 100 560 460]);
    plot(X,Y,'LineWidth',2); hold on; plot([0 1],[0 1],'k--');
    xlabel('False positive rate'); ylabel('True positive rate');
    title(sprintf('NLOS Detector ROC (AUC = %.3f)',AUC));
    grid on; axis square; legend('LSTM detector','chance','Location','SE');
    exportgraphics(f,fullfile(cfg.figdir,'fig_roc.png'),'Resolution',200);
end

function plot_confusion(label,pred,cfg)
    f=figure('Color','w','Position',[100 100 480 440]);
    cm=confusionchart(categorical(label,[0 1],{'LOS','NLOS'}), ...
                      categorical(double(pred),[0 1],{'LOS','NLOS'}));
    cm.Title='NLOS Detection Confusion Matrix';
    cm.RowSummary='row-normalized'; cm.ColumnSummary='column-normalized';
    exportgraphics(f,fullfile(cfg.figdir,'fig_confusion.png'),'Resolution',200);
end

function plot_rmse_bars(rF,rH,rL,rO,cfg)
    f=figure('Color','w','Position',[100 100 640 480]);
    m=100*[mean(rF) mean(rH) mean(rL) mean(rO)];
    e=100*[std(rF) std(rH) std(rL) std(rO)];
    b=bar(m,'FaceColor','flat'); hold on;
    b.CData=[0.5 0.5 0.5; 0.85 0.5 0.2; 0.2 0.6 0.3; 0.2 0.3 0.7];
    errorbar(1:4,m,e,'k','LineStyle','none','LineWidth',1.2);
    set(gca,'XTickLabel',{'Fixed-R','Heuristic-R','Learned-R','Oracle-R'});
    ylabel('3D Position RMSE (cm)');
    title('System RMSE by Measurement-Trust Policy');
    grid on;
    % keep the chart readable even if a policy diverges
    ymax = min(max(m+e)*1.15, 5*median(m+e));
    ylim([0 ymax]);
    for i=1:4
        yl = min(m(i)+e(i)+0.02*ymax, ymax*0.97);
        txt = sprintf('%.1f',m(i));
        if m(i) > ymax, txt = sprintf('%.0f (off-scale)',m(i)); end
        text(i,yl,txt,'HorizontalAlignment','center','FontWeight','bold');
    end
    exportgraphics(f,fullfile(cfg.figdir,'fig_rmse_policies.png'),'Resolution',200);
end

function plot_example_mission(cfg,net)
% Time-series: NLOS probability profile, detector output, and true events
    s = cfg.seedOffset + 500 + 1;
    sim = m4_simulate_mission(cfg, s);
    diagL=sqrt(cfg.W^2+cfg.H^2+cfg.ceil_z^2);
    anchor=5;  % ceiling anchor, most interesting for Z
    rawN=sim.rng_raw/diagL; drw=[0;diff(sim.rng_raw(:,anchor))];
    pHat=zeros(cfg.N,1); ks=cfg.win:cfg.N; seqs={};
    for k=ks
        idx=(k-cfg.win+1):k;
        seqs{end+1,1}=[rawN(idx,anchor).'; sim.resid(idx,anchor).'; drw(idx).']; %#ok<AGROW>
    end
    [~,scr]=classify(net,seqs); pHat(ks)=scr(:,2);

    f=figure('Color','w','Position',[100 100 900 500]);
    subplot(2,1,1);
    plot(sim.t,sim.p_nlos,'b','LineWidth',1.5); hold on;
    stem(sim.t(sim.nlos(:,anchor)),ones(sum(sim.nlos(:,anchor)),1),...
        'r','Marker','none');
    ylabel('NLOS prob / events'); ylim([0 1]);
    title(sprintf('Mission timeline (anchor A%d) - non-stationary NLOS',anchor));
    legend('NLOS probability profile','true NLOS events','Location','NW');
    subplot(2,1,2);
    plot(sim.t,pHat,'g','LineWidth',1.5); hold on; yline(0.5,'k--');
    xlabel('Time (s)'); ylabel('detector P(NLOS)'); ylim([0 1]);
    title('LSTM detector output'); legend('P(NLOS)','threshold','Location','NW');
    exportgraphics(f,fullfile(cfg.figdir,'fig_example_mission.png'),'Resolution',200);
end