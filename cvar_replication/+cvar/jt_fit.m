function jt = jt_fit(res)
%JT_FIT  VAR in J_t = (change in the trend directions of the controls, stationary combinations of
%   the controls, policies, cycle), for the joint bootstrap of Theorem thm:joint (one outcome with
%   weights estimated in levels).
u = res.units(1);
Y0 = res.controls{1};
K = size(Y0, 2);
D1 = u.D1;
if size(D1, 2) < K, Hn = null(D1'); else, Hn = zeros(K, 0); end
Dfull = [D1, Hn];
r = size(D1, 2);
Xr = Y0 * Dfull;
G = res.Z(:, res.nR + 2);
Jm = [diff(Xr(:, 1:r), 1, 1), Xr(2:end, r + 1:end), res.pol(2:end, :), G(2:end)];
if isempty(res.var_exog), veJ = []; else, veJ = res.var_exog(2:end, :); end
[A, U] = cvar.var_fit(Jm, res.opts.p, veJ);
jt = struct('Dfull', Dfull, 'Dinv', inv(Dfull), 'Xr', Xr, 'r', r, 'K', K, 'Jm', Jm, ...
    'A', A, 'U', U, 'veJ', veJ);
end
