function k = select_order_aic(Y, maxlags)
%SELECT_ORDER_AIC  Lags in differences chosen by AIC, as statsmodels' vecm.select_order with
%   deterministic = 'ci': VARs in levels with a constant and 1..maxlags+1 lags on a common
%   sample; the selected VECM order is the VAR order minus one.
[T, n] = size(Y);
nobs = T - maxlags - 1;
ys = Y(maxlags + 2:T, :);
aic = zeros(maxlags + 1, 1);
for p = 1:maxlags + 1
    X = ones(nobs, 1);
    for j = 1:p
        X = [X, Y(maxlags + 2 - j:T - j, :)]; %#ok<AGROW>
    end
    B = lsqminnorm(X, ys);
    E = ys - X * B;
    Sm = E' * E / nobs;
    aic(p) = 2 * sum(log(diag(chol(Sm)))) + 2 / nobs * (p * n^2);
end
[~, i] = min(aic);
k = i - 1;
end
