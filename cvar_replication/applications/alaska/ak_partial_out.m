function R = ak_partial_out(A, M, q)
%AK_PARTIAL_OUT  Changes of the columns of M (aligned with A.dates) with a constant and the current
%   value and q lags of (oil, dividend) projected out.
pol = [A.doil, A.W]; pol = pol(2:end, :);
F = [ones(size(M, 1) - 1, 1), cvar.lagmat(pol, 0:q)];
Dm = diff(M, 1, 1);
ok = all(~isnan(F), 2);
F = F(ok, :); Dm = Dm(ok, :);
R = Dm - F * (pinv(F) * Dm);
end
