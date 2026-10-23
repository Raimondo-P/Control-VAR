function [aic, hq, bic] = sim_ic_lags(Z, pmax)
%SIM_IC_LAGS  Lag lengths chosen by AIC, HQ and BIC on a common sample (k = p n^2 parameters).
if nargin < 2, pmax = 12; end
n = size(Z, 2); c = zeros(pmax, 3);
for p = 1:pmax
    [~, U] = cvar.var_fit(Z(pmax - p + 1:end, :), p);
    Te = size(U, 1);
    ld = 2 * sum(log(diag(chol(U' * U / Te))));
    k = p * n * n;
    c(p, :) = [ld + 2 * k / Te, ld + 2 * k * log(log(Te)) / Te, ld + k * log(Te) / Te];
end
[~, aic] = min(c(:, 1)); [~, hq] = min(c(:, 2)); [~, bic] = min(c(:, 3));
end
