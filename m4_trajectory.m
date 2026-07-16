function traj = m4_trajectory(cfg)
% M4_TRAJECTORY  3D boustrophedon (lawnmower) flight path.
%   traj is an N-by-3 array of [x y z] ground-truth positions.
%   Matches the Month 3 setup: 8-lane XY scan with sinusoidal altitude
%   between alt_min and alt_max.

N = cfg.N;
margin = 1.5;                       % keep clear of walls
xLo = margin;  xHi = cfg.W - margin;
yLo = margin;  yHi = cfg.H - margin;

nLanes = cfg.nLanes;
laneY = linspace(yLo, yHi, nLanes);

% Build the serpentine waypoint list in XY
wp = [];
for k = 1:nLanes
    if mod(k,2)==1
        wp = [wp; xLo laneY(k); xHi laneY(k)]; %#ok<AGROW>
    else
        wp = [wp; xHi laneY(k); xLo laneY(k)]; %#ok<AGROW>
    end
end

% Arc-length parameterise and resample to N points
d = [0; cumsum(sqrt(sum(diff(wp).^2,2)))];
s = linspace(0, d(end), N).';
x = interp1(d, wp(:,1), s);
y = interp1(d, wp(:,2), s);

% Sinusoidal altitude profile (drone tracks ceiling contour)
t = (0:N-1).'*cfg.dt;
z = cfg.alt_min + (cfg.alt_max-cfg.alt_min)*0.5*(1 - cos(2*pi*t/cfg.T*2));

traj = [x y z];
end