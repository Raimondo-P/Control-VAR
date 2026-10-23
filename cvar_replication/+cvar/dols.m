function [a, b, g] = dols(y, Y0, pol, D1, exog, q, L, constrained)
%DOLS  Dynamic least squares weights, regression (eq:dols) of the paper:
%   y_t = a + b'Y0_t + sum_{j=0..q} k_j' pol_{t-j} + sum_{j=-L..L} d_j' dX1_{t+j} + g' exog_t + e_t,
%   with dX1 = D1' dY0, and b >= 0 when constrained (non-negative least squares after partialling
%   out the other regressors). Returns the intercept a, the weights b and the coefficients g on exog.
if nargin < 5, exog = []; end
if nargin < 6 || isempty(q), q = 12; end
if nargin < 7 || isempty(L), L = 1; end
if nargin < 8 || isempty(constrained), constrained = true; end
y = y(:);
[T, ~] = size(Y0);
F = ones(T, 1);
if ~isempty(pol)
    F = [F, cvar.lagmat(pol, 0:q)];
end
if size(D1, 2) > 0
    dX1 = [nan(1, size(D1, 2)); diff(Y0, 1, 1) * D1];
    F = [F, cvar.lagmat(dX1, -L:L)];
end
ne = 0;
if ~isempty(exog)
    ex = reshape(exog, T, []);
    ne = size(ex, 2);
    F = [F, ex];
end
ok = all(~isnan(F), 2);
yy = y(ok); X = Y0(ok, :); F = F(ok, :);
if constrained
    P = pinv(F);
    b = lsqnonneg(X - F * (P * X), yy - F * (P * yy));
    gg = P * (yy - X * b);
else
    c = lsqminnorm([F, X], yy);
    gg = c(1:size(F, 2));
    b = c(size(F, 2) + 1:end);
end
a = gg(1);
if ne > 0
    g = gg(end - ne + 1:end);
else
    g = zeros(0, 1);
end
end
