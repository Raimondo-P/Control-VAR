function out = joint_from_path(res, jt, Js, H, use_steps)
%JOINT_FROM_PATH  Rebuild the controls and the treated outcome from a path of the joint VAR,
%   re-estimate the weights and the VAR in (R, W, G), and return the responses.
if nargin < 5, use_steps = true; end
o = res.opts; p = o.p; nR = res.nR; r = jt.r; K = jt.K;
X1s = [jt.Xr(1, 1:r); jt.Xr(1, 1:r) + cumsum(Js(:, 1:r), 1)];
Ss = [jt.Xr(1, r + 1:end); Js(:, r + 1:K)];
Y0s = [X1s, Ss] * jt.Dinv;
npol = size(res.pol, 2);
pols = [res.pol(1, :); Js(:, K + 1:K + npol)];
G = res.Z(:, nR + 2);
Gs = [G(1); Js(:, K + npol + 1)];
u = res.units(1);
if use_steps && ~isempty(u.steps), st = u.steps; else, st = []; end
if isempty(st)
    ys = u.a + Y0s * u.b + Gs;
else
    ys = u.a + Y0s * u.b + st * u.shift + Gs;
end
[a2, b2, sh2] = cvar.dols(ys, Y0s, pols, u.D1, st, o.q, o.L, o.constrained);
if isempty(st)
    Gh = ys - a2 - Y0s * b2;
else
    Gh = ys - a2 - Y0s * b2 - st * sh2;
end
if use_steps, ve = res.var_exog; else, ve = []; end
[A2, U2] = cvar.var_fit([pols, Gh], p, ve);
Rr = cvar.irf(A2, U2, p, nR + 1, H);
g = Rr(:, nR + 2:end); psi = Rr(:, nR + 1);
out = struct('cycle', g, 'policy', psi, 'ice', cvar.ice_from(g, psi));
end
