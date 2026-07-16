function [Rmap, Gmap, gx, gy] = m4_risk_map(cfg, net, seed)
% M4_RISK_MAP  Build the predicted NLOS risk map and the analytic GDOP map.
%   [Rmap, Gmap, gx, gy] = m4_risk_map(cfg, net, seed)
%
% The detector only emits a per-anchor belief at the drone's current location.
% To PLAN we need that belief as a field over the whole mine. Fly a lawnmower
% exploration pass, run the trained detector along it, scatter mean P(NLOS)
% onto a grid. Unvisited cells fall back to a geometry-only prior.
%   Rmap : ny-by-nx predicted NLOS risk [0,1]  (detector-derived)
%   Gmap : ny-by-nx normalised GDOP [0,1]       (analytic)

g  = cfg.plan_grid;
gx = (g/2 : g : cfg.W).';
gy = (g/2 : g : cfg.H).';
nx = numel(gx); ny = numel(gy);
nA = cfg.nAnchors; N = cfg.N; A = cfg.anchors;

% ---- GDOP map (analytic, altitude-averaged) --------------------------
Gmap = zeros(ny,nx);
zmid = 0.5*(cfg.alt_min+cfg.alt_max);
for ix=1:nx
    for iy=1:ny
        Gmap(iy,ix) = local_gdop(cfg, [gx(ix) gy(iy) zmid]);
    end
end
Gmap = Gmap / max(Gmap(:));

% ---- exploration mission (lawnmower) with spatial NLOS ----------------
rng(seed,'twister');
path = m4_trajectory(cfg);
dt = cfg.dt; t=(0:N-1).'*dt;
rng_true = zeros(N,nA);
for i=1:nA, rng_true(:,i)=vecnorm(path - A(i,:),2,2); end
pf = m4_nlos_field(cfg, path);
nlos = rand(N,nA) < pf;
los_noise = cfg.sigma_los*randn(N,nA);
nlos_bias = (cfg.nlos_amp(1)+diff(cfg.nlos_amp)*rand(N,nA)).*nlos;
phase = 2*pi*rand(1,nA); clk = cfg.clk_amp*sin(2*pi*cfg.clk_freq*t+phase);
rng_raw = max(round((rng_true+los_noise+nlos_bias+clk)/cfg.quant)*cfg.quant,0.1);
resid = baseline_resid(cfg, path, rng_raw);

% ---- detector P(NLOS), averaged across anchors -----------------------
diagL = sqrt(cfg.W^2+cfg.H^2+cfg.ceil_z^2);
rawN = rng_raw/diagL; drw = [zeros(1,nA); diff(rng_raw)];
ks = cfg.win:N; acc = zeros(N,1); cnt = 0;
for i=1:nA
    seqs={};
    for k=ks
        idx=(k-cfg.win+1):k;
        seqs{end+1,1}=[rawN(idx,i).'; resid(idx,i).'; drw(idx,i).']; %#ok<AGROW>
    end
    [~,scr]=classify(net,seqs);
    acc(ks)=acc(ks)+scr(:,2); cnt=cnt+1;
end
pmean = acc / max(cnt,1);

% ---- scatter onto the grid -------------------------------------------
pos = path(:,1:2);
Rmap = zeros(ny,nx); Wt = zeros(ny,nx); sig = 1.0;
for k=ks
    for ix=1:nx
        dx=pos(k,1)-gx(ix); if abs(dx)>3*sig, continue; end
        for iy=1:ny
            dy=pos(k,2)-gy(iy); if abs(dy)>3*sig, continue; end
            wgt=exp(-(dx*dx+dy*dy)/(2*sig^2));
            Rmap(iy,ix)=Rmap(iy,ix)+wgt*pmean(k);
            Wt(iy,ix)=Wt(iy,ix)+wgt;
        end
    end
end
seen = Wt>1e-6;
Rmap(seen)  = Rmap(seen)./Wt(seen);
Rmap(~seen) = cfg.nlos_floor + 0.15*Gmap(~seen);
Rmap = min(max(Rmap,0),1);
end

% ===================================================================
function gdop = local_gdop(cfg, p)
A=cfg.anchors; nA=cfg.nAnchors; Hh=zeros(nA,3);
for i=1:nA
    v=p-A(i,:); d=norm(v); if d<1e-6, d=1e-6; end
    Hh(i,:)=v/d;
end
M=Hh.'*Hh;
if rcond(M)<1e-9, gdop=10; return; end
gdop=sqrt(trace(inv(M)));
end

% ===================================================================
function resid = baseline_resid(cfg, truth, rng_raw)
N=cfg.N; nA=cfg.nAnchors; A=cfg.anchors; dt=cfg.dt;
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