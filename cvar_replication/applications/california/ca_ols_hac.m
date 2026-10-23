function [b, V, u] = ca_ols_hac(y, X, nlags)
%CA_OLS_HAC  OLS with a HAC covariance (Bartlett, nlags lags, no small-sample correction), as
%   statsmodels' cov_type='HAC'. X must include the constant if wanted.
b = pinv(X) * y; u = y - X * b;
xu = X .* u; S = xu' * xu;
for l = 1:nlags
    s = xu(l + 1:end, :)' * xu(1:end - l, :);
    S = S + (1 - l / (nlags + 1)) * (s + s');
end
Hi = pinv(X' * X); V = Hi * S * Hi;
end
