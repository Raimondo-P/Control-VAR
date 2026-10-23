function p = lag_rule(Z, exog, cols, pmax, m, level)
%LAG_RULE  Smallest order in 1..pmax with no Ljung-Box rejection (m lags, `level`) in the
%   equations `cols` (1-based) of a VAR in Z; pmax if no order passes. This is the lag rule of
%   the applications.
if nargin < 5 || isempty(m), m = 12; end
if nargin < 6 || isempty(level), level = 0.05; end
for p = 1:pmax
    [~, U] = cvar.var_fit(Z, p, exog);
    ok = true;
    for i = cols(:)'
        if cvar.ljungbox_p(U(:, i), m) <= level
            ok = false; break
        end
    end
    if ok, return, end
end
p = pmax;
end
