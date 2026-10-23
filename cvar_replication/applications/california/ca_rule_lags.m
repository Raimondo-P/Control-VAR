function [pc, pv] = ca_rule_lags(S, ve)
%CA_RULE_LAGS  Lag orders of the CVAR and of the VAR without controls by the Ljung-Box rule.
if nargin < 2, ve = []; end
[rc, rv] = ca_fits(S, 1, 1, ve, 'differenced');
pc = cvar.lag_rule(rc.Z, rc.var_exog, [1 2], 12);
pv = cvar.lag_rule(rv.nc.Z, rv.var_exog, [1 2], 12);
end
