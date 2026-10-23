function [A, U] = var_fit(Z, p, exog)
%VAR_FIT  OLS VAR(p) with a constant and optional exogenous regressors.
%   [A, U] = cvar.var_fit(Z, p, exog) returns the stacked coefficients A, of size
%   (1 + n*p + ne) x n (constant, lag 1, ..., lag p, exogenous regressors), and the residuals U,
%   of size (T - p) x n. exog is T x ne or empty.
if nargin < 3, exog = []; end
[T, n] = size(Z);
X = ones(T - p, 1);
for j = 1:p
    X = [X, Z(p - j + 1:T - j, :)]; %#ok<AGROW>
end
if ~isempty(exog)
    exog = reshape(exog, T, []);
    X = [X, exog(p + 1:T, :)];
end
A = lsqminnorm(X, Z(p + 1:T, :));
U = Z(p + 1:T, :) - X * A;
end
