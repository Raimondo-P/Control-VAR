function out = draw_conditional(res, idx, H, no_controls)
%DRAW_CONDITIONAL  One draw of the residual bootstrap that re-estimates the weights (steps i to iv
%   of the bootstrap): the VAR in (R, W, G) is resampled with the controls held at their observed
%   values; idx (length T - p) are the residual rows.
if nargin < 4, no_controls = true; end
o = res.opts; p = o.p; nR = res.nR; T = size(res.Z, 1);
Zs = cvar.var_sim(res.A, res.U, res.Z(1:p, :), p, T, idx, res.var_exog);
pols = Zs(:, 1:nR + 1);
Gs = zeros(T, res.m);
for j = 1:res.m
    u = res.units(j);
    if u.imposed
        Gs(:, j) = Zs(:, nR + 1 + j);
        continue
    end
    st = u.steps;
    ys = u.a + res.controls{j} * u.b + st * u.shift + Zs(:, nR + 1 + j);
    if isempty(st), stx = []; else, stx = st; end
    [a2, b2, sh2] = cvar.dols(ys, res.controls{j}, pols, u.D1, stx, o.q, o.L, o.constrained);
    Gs(:, j) = ys - a2 - res.controls{j} * b2 - st * sh2;
end
[A2, U2] = cvar.var_fit([pols, Gs], p, res.var_exog);
Rr = cvar.irf(A2, U2, p, nR + 1, H);
g = cvar.cumresp(Rr(:, nR + 2:end), res.diff);
psi = Rr(:, nR + 1);
out = struct('cycle', g, 'policy', psi, 'ice', cvar.ice_from(g, psi));
if no_controls
    out.no_controls = cvar.nc_draw(res, idx, H);
end
end
