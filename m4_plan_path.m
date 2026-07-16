function path = m4_plan_path(cfg, Rmap, Gmap, gx, gy, strategy)
% M4_PLAN_PATH  Generate a 3D coverage path under one planning strategy.
%   path = m4_plan_path(cfg, Rmap, Gmap, gx, gy, strategy)
%
% All strategies COVER the whole mine; they differ in lane ORDER and the
% WAYPOINTS within each lane. To isolate which objective drives RMSE, each
% router is fed ONLY its target map (the other is zeroed):
%   'lawnmower'   : Month 1-3 baseline, ignores both maps.
%   'nlos_only'   : route on NLOS risk only (the policy that loses).
%   'gdop_astar'  : NEW. route on GDOP only - optimise geometry per lane.
%   'gdop_global' : NEW. SA lane order + GDOP routing.
% Output: N-by-3 [x y z]. Resampled at a FIXED step length so all strategies
% fly at the same speed (longer paths take more timesteps, not faster flight).

nLanes = cfg.nLanes; margin = 1.5;
laneY = linspace(margin, cfg.H-margin, nLanes);
xLo = margin; xHi = cfg.W - margin;

switch strategy
    case 'lawnmower'                       % Month 1-3 baseline
        wp = serpentine_wps(1:nLanes, laneY, xLo, xHi);
    case 'nlos_only'                       % avoid NLOS only (losing policy)
        wp = astar_lane_wps(cfg, Rmap, zeros(size(Gmap)), gx, gy, ...
                            1:nLanes, laneY, xLo, xHi);
    case 'gdop_astar'                      % NEW: optimise geometry per lane
        wp = astar_lane_wps(cfg, zeros(size(Rmap)), Gmap, gx, gy, ...
                            1:nLanes, laneY, xLo, xHi);
    case 'gdop_global'                     % NEW: SA lane order + GDOP routing
        order = sa_lane_order(cfg, Rmap, Gmap, gx, gy, laneY, xLo, xHi);
        wp = astar_lane_wps(cfg, zeros(size(Rmap)), Gmap, gx, gy, ...
                            order, laneY, xLo, xHi);
    otherwise
        error('unknown strategy %s', strategy);
end

% --- strip consecutive duplicate waypoints (DP can repeat a point) ---
keep = [true; any(abs(diff(wp,1,1)) > 1e-9, 2)];
wp = wp(keep,:);

d = [0; cumsum(sqrt(sum(diff(wp).^2,2)))];
good = [true; diff(d) > 1e-9];        % drop any zero-length segments
d = d(good); wp = wp(good,:);

% Resample at a FIXED step length so every strategy flies at the SAME speed.
base_len = sum(sqrt(sum(diff(m4_trajectory(cfg)).^2,2)));
v_drone  = base_len / cfg.T;          % m/s
step     = v_drone * cfg.dt;          % m per timestep
Nsteps   = max(round(d(end)/step), cfg.win+2);
s = linspace(0, d(end), Nsteps).';
x = interp1(d, wp(:,1), s, 'linear');
y = interp1(d, wp(:,2), s, 'linear');
t = (0:Nsteps-1).'*cfg.dt;
z = cfg.alt_min + (cfg.alt_max-cfg.alt_min)*0.5*(1 - cos(2*pi*t/cfg.T*2));
path = [x y z];
end

% ===================================================================
function wp = serpentine_wps(order, laneY, xLo, xHi)
wp=[]; flip=false;
for li=1:numel(order)
    k=order(li);
    if ~flip, wp=[wp; xLo laneY(k); xHi laneY(k)]; %#ok<AGROW>
    else,     wp=[wp; xHi laneY(k); xLo laneY(k)]; end %#ok<AGROW>
    flip=~flip;
end
end

% ===================================================================
function c = lane_cost(Gmap, Rmap, gx, gy, laneY, xLo, xHi, wG, wN)
nL=numel(laneY); c=zeros(nL,1); xs=linspace(xLo,xHi,25);
for k=1:nL
    acc=0;
    for xi=xs
        acc=acc + wG*bilin(Gmap,gx,gy,xi,laneY(k)) ...
                + wN*bilin(Rmap,gx,gy,xi,laneY(k));
    end
    c(k)=acc/numel(xs);
end
end

% ===================================================================
function wp = astar_lane_wps(cfg, Rmap, Gmap, gx, gy, order, laneY, xLo, xHi)
% Per-lane DP (band A* in the y direction) over x-stations: minimise
% sum( step + wGDOP*GDOP + wNLOS*risk ) while sweeping the lane band.
% Maps passed in as zeros are effectively ignored, letting the caller select
% the routing objective (NLOS-only vs GDOP-only).
nL=numel(order);
xs=linspace(xLo,xHi,cfg.plan_nx);
yband=cfg.plan_yband;
wp=[]; flip=false;
for li=1:nL
    k=order(li); y0=laneY(k);
    ycand=linspace(max(y0-yband,1), min(y0+yband,cfg.H-1), cfg.plan_ny);
    ns=numel(xs); nc=numel(ycand);
    C=inf(ns,nc); B=zeros(ns,nc);
    for j=1:nc, C(1,j)=cellcost(cfg,Rmap,Gmap,gx,gy,xs(1),ycand(j)); end
    for s=2:ns
        for j=1:nc
            best=inf; bj=1;
            for jp=1:nc
                step=hypot(xs(s)-xs(s-1), ycand(j)-ycand(jp));
                val=C(s-1,jp)+step+cellcost(cfg,Rmap,Gmap,gx,gy,xs(s),ycand(j));
                if val<best, best=val; bj=jp; end
            end
            C(s,j)=best; B(s,j)=bj;
        end
    end
    [~,je]=min(C(ns,:));
    ypath=zeros(ns,1); ypath(ns)=ycand(je); j=je;
    for s=ns:-1:2, j=B(s,j); ypath(s-1)=ycand(j); end
    seg=[xs(:) ypath];
    if flip, seg=flipud(seg); end
    wp=[wp; seg]; flip=~flip; %#ok<AGROW>
end
end

% ===================================================================
function order = sa_lane_order(cfg, Rmap, Gmap, gx, gy, laneY, xLo, xHi)
% Simulated annealing over lane visiting order minimising total combined cost.
nL=numel(laneY); order=1:nL;
E=order_energy(order,cfg,Rmap,Gmap,gx,gy,laneY,xLo,xHi);
T=1.0; rng(7,'twister');
for it=1:3000
    a=randi(nL); b=randi(nL);
    if a==b, continue; end
    cand=order; cand([a b])=cand([b a]);
    Ec=order_energy(cand,cfg,Rmap,Gmap,gx,gy,laneY,xLo,xHi);
    if Ec<E || rand<exp((E-Ec)/max(T,1e-3))
        order=cand; E=Ec;
    end
    T=T*0.998;
end
end

function E = order_energy(order,cfg,Rmap,Gmap,gx,gy,laneY,xLo,xHi)
c=lane_cost(Gmap,Rmap,gx,gy,laneY,xLo,xHi,cfg.plan_wGDOP,cfg.plan_wNLOS);
E=sum(c(order));
for li=2:numel(order)
    E=E+0.3*abs(laneY(order(li))-laneY(order(li-1)));
end
end

% ===================================================================
function v = cellcost(cfg,Rmap,Gmap,gx,gy,x,y)
v = cfg.plan_wGDOP*bilin(Gmap,gx,gy,x,y) + cfg.plan_wNLOS*bilin(Rmap,gx,gy,x,y);
end

function v = bilin(M,gx,gy,x,y)
x=min(max(x,gx(1)),gx(end)); y=min(max(y,gy(1)),gy(end));
ix=find(gx<=x,1,'last'); iy=find(gy<=y,1,'last');
ix=min(ix,numel(gx)-1); iy=min(iy,numel(gy)-1);
tx=(x-gx(ix))/(gx(ix+1)-gx(ix)); ty=(y-gy(iy))/(gy(iy+1)-gy(iy));
v = (1-tx)*(1-ty)*M(iy,ix)   + tx*(1-ty)*M(iy,ix+1) ...
  + (1-tx)*ty*M(iy+1,ix)     + tx*ty*M(iy+1,ix+1);
end