function [X, Y, meta] = m4_build_dataset(cfg, seeds)
% M4_BUILD_DATASET  Turn missions into per-anchor windowed samples.
%
%   [X, Y, meta] = m4_build_dataset(cfg, seeds)
%
% For every anchor at every timestep we form a sliding window of length
% cfg.win ending at that step, with cfg.nFeat features per sample:
%   feature 1: normalised raw range          (range / diag of mine)
%   feature 2: innovation / residual (m)
%   feature 3: first difference of raw range  (m), a crude velocity/jump cue
%
% Output:
%   X    : cell array of [nFeat x win] sequences  (for sequence network)
%   Y    : categorical vector ["LOS","NLOS"] per sample
%   meta : struct with .seed .k .anchor for traceability

win   = cfg.win;
nA    = cfg.nAnchors;
diagL = sqrt(cfg.W^2 + cfg.H^2 + cfg.ceil_z^2);

X = {};
Yv = [];
mseed = []; mk = []; manch = [];

for s = seeds
    sim = m4_simulate_mission(cfg, s);
    rawN = sim.rng_raw / diagL;                 % normalised range
    res  = sim.resid;
    drw  = [zeros(1,nA); diff(sim.rng_raw)];     % first difference

    for i = 1:nA
        for k = win:cfg.N
            idx = (k-win+1):k;
            seq = [rawN(idx,i).'; res(idx,i).'; drw(idx,i).'];  % nFeat x win
            X{end+1,1} = seq;                                   %#ok<AGROW>
            Yv(end+1,1) = sim.nlos(k,i);                        %#ok<AGROW>
            mseed(end+1,1)= s; mk(end+1,1)=k; manch(end+1,1)=i; %#ok<AGROW>
        end
    end
end

Y = categorical(Yv, [0 1], {'LOS','NLOS'});
meta = struct('seed',mseed,'k',mk,'anchor',manch);

fprintf('Built %d samples from %d missions (%.1f%% NLOS).\n', ...
    numel(X), numel(seeds), 100*mean(Yv));
end