function out = responses(res, H)
%RESPONSES  Point estimates: response of the cycles (gamma_h), of the policy to its own
%   innovation (psi_h), impulse causal effects (Theorem 4(a)) and response of the VAR without
%   controls, to a unit innovation in the policy.
if nargin < 2, H = 36; end
p = res.opts.p; nR = res.nR;
Rr = cvar.irf(res.A, res.U, p, nR + 1, H);
g = cvar.cumresp(Rr(:, nR + 2:end), res.diff);
psi = Rr(:, nR + 1);
out = struct('cycle', g, 'policy', psi, 'ice', cvar.ice_from(g, psi), 'all', Rr);
Rv = cvar.irf(res.nc.A, res.nc.U, p, nR + 1, H);
out.no_controls = cvar.cumresp(Rv(:, nR + 2:end), res.diff);
end
