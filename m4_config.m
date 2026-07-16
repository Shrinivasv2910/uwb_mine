function cfg = m4_config()
% M4_CONFIG  Central configuration for Month 4 Pillar B (learned NLOS detection).
% Consistent with Months 1-3: 20x15 m mine, 5 anchors (4 ground + 1 ceiling),
% 10 Hz polling, boustrophedon scan, IEEE 802.15.4a CM4 industrial NLOS model.

% ---- Mine geometry ----
cfg.W = 20;            % mine width  (m)
cfg.H = 15;            % mine height (m) in plan view
cfg.ceil_z = 2.5;      % ceiling anchor elevation (m)

% 5-anchor non-coplanar array (matches Month 3 Task 2)
cfg.anchors = [ 0   0   0;        % A1
    20   0   0;        % A2
    20  15   0;        % A3
    0  15   0;        % A4
    10   7.5 2.5];     % A5 ceiling
cfg.nAnchors = size(cfg.anchors,1);

% ---- Timing ----
cfg.fs       = 10;     % Hz polling rate
cfg.dt       = 1/cfg.fs;
cfg.T        = 60;     % mission duration (s)
cfg.N        = cfg.T*cfg.fs + 1;   % number of timesteps (601)

% ---- Flight path (altitude band, matches Month 3) ----
cfg.nLanes   = 8;
cfg.alt_min  = 3.5;    % m
cfg.alt_max  = 5.5;    % m

% ---- Noise model (IEEE 802.15.4a CM4 industrial NLOS) ----
cfg.sigma_los   = 0.25;       % m, Gaussian LOS noise (thermal + jitter)
cfg.nlos_amp    = [0.4 1.2];  % m, NLOS positive-bias range (uniform)
cfg.clk_amp     = 0.005;      % m, clock-drift sinusoid amplitude
cfg.clk_freq    = 0.01;       % Hz
cfg.quant       = 0.001;      % m, ADC quantisation step

% NLOS probability. For Pillar B we use a NON-STATIONARY profile so the
% detector has a genuine regime to learn (see m4_nlos_profile.m).
cfg.nlos_p_base = 0.10;       % baseline NLOS probability
cfg.nlos_p_peak = 0.45;       % peak probability inside the "collapsed gallery"

% ---- Detector windowing ----
cfg.win = 9;          % sliding-window length (samples) fed to the detector
% 9 samples @10Hz = 0.9 s of context
cfg.nFeat = 3;        % features per anchor per sample: [raw range, residual, delta]

% ---- Dataset sizing ----
cfg.nTrainSeeds = 40; % missions used for training
cfg.nValSeeds   = 10; % missions used for validation
cfg.nTestSeeds  = 20; % missions used for held-out test

% ---- Reproducibility ----
cfg.seedOffset  = 1000;   % base offset so train/val/test seeds never overlap

% ---- Output ----
cfg.figdir = fullfile(pwd,'figures');
if ~exist(cfg.figdir,'dir'); mkdir(cfg.figdir); end
end