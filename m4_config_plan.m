function cfg = m4_config_plan()
% M4_CONFIG_PLAN  Month 4 Pillar A config. Inherits everything from
% m4_config() (mine geometry, anchors, noise, detector window) then adds the
% spatial NLOS field and path-planner parameters. The trained detector in
% nlos_detector.mat stays valid because no inherited field changes.

cfg = m4_config();

% ---- Spatial NLOS field (NEW) -----------------------------------------
% NLOS in a real mine comes from collapsed galleries and rock walls near the
% PERIMETER. Hot-spots sited near edges/corners, away from the anchor centre.
% Used both to label the detector's training distribution and to build the
% predicted risk map that motivates the planner study.
cfg.gallery_centre = [17.5  3.0];   % m, collapsed gallery, lower-right wall
cfg.gallery_sigma  = 2.4;           % m, spatial spread
cfg.gallery_peak   = 0.60;          % peak NLOS prob for a ray through its core
cfg.machine_centre = [3.0 12.5];    % parked machine, upper-left wall
cfg.machine_sigma  = 2.0;
cfg.machine_peak   = 0.50;
cfg.nlos_floor     = 0.05;          % ambient NLOS probability everywhere

% ---- Soft measurement trust (NEW, replaces hard accept/reject) --------
% R_i = Rmeas * (1 + lambda * P(NLOS)_i), clamped. A flagged anchor is
% down-weighted, not discarded, so partial information is retained.
cfg.softLambda   = 25;              % trust-inflation gain
cfg.softRmax     = 1.0;             % m^2, ceiling on inflated R
cfg.pGateHard    = 0.95;            % only drop near-certain NLOS

% ---- GDOP-aware path planner (NEW) ------------------------------------
% Finding from the risk-map study: the learned detector + soft-trust absorbs
% NLOS so well that residual RMSE is GDOP-limited, not NLOS-limited. So the
% planner optimises GEOMETRY: it routes lanes toward low-GDOP regions where
% the anchor array best constrains position. NLOS is kept as a minor tie-break.
cfg.plan_wGDOP   = 8.0;             % dominant: optimise geometry
cfg.plan_wNLOS   = 1.0;             % minor tie-break only
cfg.plan_grid    = 0.5;             % m, planning/coverage grid resolution
cfg.plan_yband   = 2.5;             % m, max lane deviation the planner may use
cfg.plan_nx      = 31;              % x-stations per lane in the DP router
cfg.plan_ny      = 9;               % candidate y-offsets per station

% ---- Planner study ----------------------------------------------------
cfg.nPlanSeeds   = 20;              % held-out missions for the comparison
end