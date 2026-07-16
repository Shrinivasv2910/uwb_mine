function cfg = m4_config_8anchor()
% M4_CONFIG_8ANCHOR  Config for the multi-drone handoff animation. Inherits
% the planner config, then overrides the anchor array to 8 anchors (4 corners
% + 4 mid-walls at varied heights) so drones have spare anchors to hand off to
% when some go NLOS. Detector (trained per-anchor on 5 anchors) still applies -
% it classifies each anchor independently regardless of array size.

cfg = m4_config_plan();

% 8-anchor array: 4 ground corners + 4 mid-wall anchors at mixed heights
cfg.anchors = [ 0    0    0.0;     % A1 corner
    20    0    2.0;     % A2 corner (raised)
    20   15    0.0;     % A3 corner
    0   15    2.0;     % A4 corner (raised)
    10    0    2.5;     % A5 mid-wall south, ceiling height
    20    7.5  2.5;     % A6 mid-wall east, ceiling height
    10   15    2.5;     % A7 mid-wall north, ceiling height
    0    7.5  2.5];    % A8 mid-wall west, ceiling height
cfg.nAnchors = size(cfg.anchors,1);

cfg.activeK = 4;        % each drone uses an active set of 4 anchors
end