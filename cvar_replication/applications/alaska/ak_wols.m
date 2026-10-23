function [b, V, r2] = ak_wols(dy, X, F, nlags)
%AK_WOLS  OLS weights in changes with a HAC covariance (Bartlett kernel, nlags lags, no
%   small-sample correction, as statsmodels' cov_type='HAC'). Returns the weights, their
%   covariance and the R-squared.
if nargin < 4, nlags = 12; end
Z = [X, F];
beta = pinv(Z) * dy;
u = dy - Z * beta;
xu = Z .* u;
S = xu' * xu;
for l = 1:nlags
    s = xu(l + 1:end, :)' * xu(1:end - l, :);
    S = S + (1 - l / (nlags + 1)) * (s + s');
end
Hi = pinv(Z' * Z);
Vf = Hi * S * Hi;
k = size(X, 2);
b = beta(1:k); V = Vf(1:k, 1:k);
r2 = 1 - (u' * u) / sum((dy - mean(dy)).^2);
end
