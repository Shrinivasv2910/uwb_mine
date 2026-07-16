function p = m4_nlos_field(cfg, pos)
% M4_NLOS_FIELD  Per-anchor NLOS probability for a drone at position(s) pos.
%   pos : M-by-3 drone positions [x y z]
%   p   : M-by-nAnchors NLOS probability for the ray drone->anchor
%
% Whether a UWB ray is blocked depends on whether the straight line from the
% drone to that anchor passes near an obstacle. Two drones (or one drone at
% different times/places) therefore see different anchors blocked - the
% structure the planner exploits. The detector only sees ranges/residuals, so
% this field is hidden ground truth used to generate labels, never an input.

A  = cfg.anchors; nA = cfg.nAnchors;
M  = size(pos,1);
p  = cfg.nlos_floor * ones(M,nA);

obst = { cfg.gallery_centre, cfg.gallery_sigma, cfg.gallery_peak; ...
    cfg.machine_centre,  cfg.machine_sigma,  cfg.machine_peak };

for i = 1:nA
    ai = A(i,1:2);                       % blockage modelled in plan view
    for m = 1:M
        dm = pos(m,1:2);
        seg = ai - dm;  L = norm(seg);
        if L < 1e-6, continue; end
        u = seg / L;
        acc = 0;
        for o = 1:size(obst,1)
            c = obst{o,1}; s = obst{o,2}; pk = obst{o,3};
            t = max(0, min(L, dot(c - dm, u)));   % closest approach on segment
            closest = dm + t*u;
            d2 = sum((c - closest).^2);
            acc = acc + pk * exp(-d2 / (2*s^2));
        end
        p(m,i) = min(0.95, cfg.nlos_floor + acc);
    end
end
end