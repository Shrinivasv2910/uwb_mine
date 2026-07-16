function sim = m4_simulate_mission(cfg, seed)
% M4_SIMULATE_MISSION  Simulate one 60 s mission and return everything the
% detector and evaluator need.
%
%   sim = m4_simulate_mission(cfg, seed)
%
% Output struct fields (all sized over the mission):
%   sim.t        N-by-1     time (s)
%   sim.truth    N-by-3     ground-truth [x y z]
%   sim.rng_true N-by-nA    true geometric ranges
%   sim.rng_raw  N-by-nA    noisy measured ranges (LOS noise + NLOS bias + drift)
%   sim.nlos     N-by-nA    ground-truth NLOS label (1 = NLOS this sample/anchor)
%   sim.resid    N-by-nA    innovation (measured - predicted) from a baseline EKF
%   sim.ekf_pos  N-by-3     baseline-EKF position estimate (range-only, fixed R)
%   sim.p_nlos   N-by-1     NLOS probability profile used (for plotting)

rng(seed,'twister');

N  = cfg.N;
nA = cfg.nAnchors;
A  = cfg.anchors;
dt = cfg.dt;
t  = (0:N-1).'*dt;

truth = m4_trajectory(cfg);

% --- True ranges ---
rng_true = zeros(N,nA);
for i = 1:nA
    rng_true(:,i) = vecnorm(truth - A(i,:), 2, 2);
end

% --- NLOS labels from the non-stationary profile ---
p_nlos = m4_nlos_profile(cfg, t);
nlos = rand(N,nA) < p_nlos;          % independent per anchor per sample

% --- Build noisy measured ranges ---
los_noise = cfg.sigma_los * randn(N,nA);
nlos_bias = (cfg.nlos_amp(1) + diff(cfg.nlos_amp)*rand(N,nA)) .* nlos;
% per-anchor clock drift: random phase per anchor
phase = 2*pi*rand(1,nA);
clk = cfg.clk_amp * sin(2*pi*cfg.clk_freq*t + phase);
rng_raw = rng_true + los_noise + nlos_bias + clk;
rng_raw = round(rng_raw / cfg.quant) * cfg.quant;   % quantise
rng_raw = max(rng_raw, 0.1);                        % physical floor

% --- Baseline EKF (constant-velocity, range-only, FIXED R) ---
% State x = [px py pz vx vy vz]
F = [eye(3) dt*eye(3); zeros(3) eye(3)];
q = 0.05;                                  % process noise scale
G = [0.5*dt^2*eye(3); dt*eye(3)];
Q = G*G.'*q;
Rfix = 0.04;                               % m^2 per anchor (DWM1001, Month 3)

x = [truth(1,:).'; 0;0;0];                 % init at truth (favourable, fine for baseline)
P = diag([1 1 1 1 1 1]);

ekf_pos = zeros(N,3);
resid   = zeros(N,nA);

for k = 1:N
    % Predict
    x = F*x;
    P = F*P*F.' + Q;

    % Update with all anchors (sequential scalar updates)
    p = x(1:3).';
    for i = 1:nA
        d_pred = norm(p - A(i,:));
        innov  = rng_raw(k,i) - d_pred;     % <-- the detector feature
        resid(k,i) = innov;

        Hi = zeros(1,6);
        Hi(1:3) = (p - A(i,:)) / max(d_pred,1e-6);
        S = Hi*P*Hi.' + Rfix;
        K = (P*Hi.') / S;
        x = x + K*innov;
        P = (eye(6) - K*Hi)*P;
        p = x(1:3).';
    end
    ekf_pos(k,:) = x(1:3).';
end

sim = struct('t',t,'truth',truth,'rng_true',rng_true,'rng_raw',rng_raw, ...
    'nlos',logical(nlos),'resid',resid,'ekf_pos',ekf_pos, ...
    'p_nlos',p_nlos);
end