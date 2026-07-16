function out = m4_run_on_path(cfg, net, path, seed, trustMode)
% M4_RUN_ON_PATH  Fly a given path, simulate ranges with the spatial NLOS
% field, run the detector + soft-trust EKF, report accuracy and NLOS exposure.
%   out = m4_run_on_path(cfg, net, path, seed, trustMode)
% trustMode: 'soft' (R-inflation by P(NLOS)) or 'fixed' (ablation).
%   out.rmse     scalar 3D RMSE (m)
%   out.err      N-by-1 per-step 3D error (m)
%   out.exposure mean true P(NLOS) along the path (planner's objective)
%
% N is taken from the PATH (not cfg.N): paths are resampled at a fixed step
% length, so longer routes have more timesteps at the same drone speed. This
% removes path length as a confound on RMSE.

rng(seed,'twister');
N=size(path,1); nA=cfg.nAnchors; A=cfg.anchors; dt=cfg.dt;
t=(0:N-1).'*dt;

rng_true=zeros(N,nA);
for i=1:nA, rng_true(:,i)=vecnorm(path - A(i,:),2,2); end
pf = m4_nlos_field(cfg, path);
nlos = rand(N,nA) < pf;
los_noise=cfg.sigma_los*randn(N,nA);
nlos_bias=(cfg.nlos_amp(1)+diff(cfg.nlos_amp)*rand(N,nA)).*nlos;
phase=2*pi*rand(1,nA); clk=cfg.clk_amp*sin(2*pi*cfg.clk_freq*t+phase);
rng_raw=max(round((rng_true+los_noise+nlos_bias+clk)/cfg.quant)*cfg.quant,0.1);
resid = baseline_resid(cfg, path, rng_raw);

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
G=[0.5*dt^2*eye(3); dt*eye(3)]; Q=G*G.'*0.6; Rmeas=0.04;
x=[path(1,:).';0;0;0]; P=eye(6); err=zeros(N,1);
for k=1:N
    x=F*x; P=F*P*F.'+Q; p=x(1:3).';

    % --- compute trust weight for every anchor first ---
    Reff = zeros(1,nA); useIt = false(1,nA);
    for i=1:nA
        switch trustMode
            case 'fixed', Reff(i)=Rmeas; useIt(i)=true;
            case 'soft'
                Reff(i)=min(Rmeas*(1+cfg.softLambda*pHat(k,i)),cfg.softRmax);
                useIt(i)=pHat(k,i)<cfg.pGateHard;
            otherwise, Reff(i)=Rmeas; useIt(i)=true;
        end
    end
    % GUARD: never drop every anchor - keep the 3 least-suspect ones.
    if nnz(useIt) < 3
        [~,ord]=sort(pHat(k,:),'ascend');
        useIt(:)=false; useIt(ord(1:3))=true;
    end

    for i=1:nA
        if ~useIt(i), continue; end
        dpred=norm(p-A(i,:)); innov=rng_raw(k,i)-dpred;
        if abs(innov) > 3.0, continue; end      % GUARD: reject wild NLOS spike
        Hi=zeros(1,6); Hi(1:3)=(p-A(i,:))/max(dpred,1e-6);
        S=Hi*P*Hi.'+Reff(i); K=(P*Hi.')/S;
        x=x+K*innov; P=(eye(6)-K*Hi)*P; p=x(1:3).';
    end
    P=(P+P.')/2;                                % keep symmetric
    P=min(P,1e4);                               % GUARD: bound covariance
    err(k)=norm(x(1:3).'-path(k,:));
end

out=struct('rmse',sqrt(mean(err.^2)),'err',err, ...
    'exposure',mean(pf(:)),'t',t,'pHat',pHat,'pfield',pf);
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