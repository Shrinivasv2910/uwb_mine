function p = m4_nlos_profile(cfg, t)
% M4_NLOS_PROFILE  Time-varying NLOS probability for a mission.
%   p = m4_nlos_profile(cfg, t) returns NLOS probability at each time in t.
%
% The drone passes through a "collapsed gallery" region mid-mission where
% multipath is severe: probability ramps from the baseline up to the peak
% and back down. This non-stationarity is what makes a *learned* detector
% (and later, adaptive filtering) genuinely useful rather than a fixed
% threshold being optimal. A second, sharper transient simulates passing a
% large metal machine late in the mission.

T = cfg.T;
% Smooth gallery ramp centred at 55% of the mission, ~16 s wide
c1 = 0.55*T;  w1 = 8;
gallery = exp(-((t - c1).^2)/(2*w1^2));

% Sharp machine transient near 82% of the mission, ~3 s wide
c2 = 0.82*T;  w2 = 1.5;
machine = exp(-((t - c2).^2)/(2*w2^2));

p = cfg.nlos_p_base ...
    + (cfg.nlos_p_peak - cfg.nlos_p_base)*gallery ...
    + 0.25*machine;

p = min(max(p, 0), 0.9);   % clamp to valid probability
end