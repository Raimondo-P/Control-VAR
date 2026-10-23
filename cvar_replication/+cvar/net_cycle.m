function [cn, off] = net_cycle(c, pol, q)
%NET_CYCLE  The cycle net of the current value and q lags of the policies, an estimate of the
%   counterfactual cycle. off is the number of leading observations dropped.
if nargin < 3 || isempty(q), q = 12; end
c = c(:);
T = numel(c);
Z = [ones(T, 1), cvar.lagmat(pol, 0:q)];
ok = all(~isnan(Z), 2);
co = lsqminnorm(Z(ok, :), c(ok));
cn = c(ok) - Z(ok, 2:end) * co(2:end);
off = sum(~ok);
end
