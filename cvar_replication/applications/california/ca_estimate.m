function est = ca_estimate(S, pc, pv, B, ve, branch, seed, H)
%CA_ESTIMATE  CVAR (equal weights, differenced branch, or estimated weights in levels) and the VAR
%   without controls, with residual-bootstrap percentile (90 and 68 percent) and Hall bands.
w = ones(size(S.C, 2), 1) / size(S.C, 2);
[rc, rv] = ca_fits(S, pc, pv, ve, branch);
bc = cvar.bootstrap(rc, B, H, struct('seed', seed, 'no_controls', false));
bv = cvar.bootstrap(rv, B, H, struct('seed', seed + 1, 'no_controls', true));
est = struct();
for x = {'cvar', bc, 'cycle'; 'var', bv, 'no_controls'}'
    [lo, hi] = cvar.bands(x{2}, x{3}, 0.10, 'percentile');
    [hl, hh] = cvar.bands(x{2}, x{3}, 0.10, 'hall');
    [l6, h6] = cvar.bands(x{2}, x{3}, 0.32, 'percentile');
    est.(x{1}) = struct('pt', x{2}.point.(x{3})(:, 1), 'lo', lo(:, 1), 'hi', hi(:, 1), 'hlo', hl(:, 1), ...
        'hhi', hh(:, 1), 'lo68', l6(:, 1), 'hi68', h6(:, 1));
end
end
