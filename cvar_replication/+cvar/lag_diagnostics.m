function D = lag_diagnostics(Z, lags, exog, extra, names)
%LAG_DIAGNOSTICS  Information criteria on a common sample and residual autocorrelation tests at
%   each candidate order of the VAR in Z. Returns a table with one row per order.
if nargin < 2 || isempty(lags), lags = [1 2 3 4 6 8 12]; end
if nargin < 3, exog = []; end
if nargin < 4 || isempty(extra), extra = 12; end
[T, n] = size(Z);
if nargin < 5 || isempty(names), names = arrayfun(@(i) sprintf('eq%d', i), 1:n, 'UniformOutput', false); end
pm = max(lags);
rows = {};
for p = lags(:)'
    Zs = Z(pm - p + 1:end, :);
    if isempty(exog), ex = []; else, ex = exog(pm - p + 1:end, :); end
    [A, U] = cvar.var_fit(Zs, p, ex);
    Te = size(U, 1);
    k = size(A, 1) * n;
    ld = 2 * sum(log(diag(chol(U' * U / Te))));
    row = struct('p', p, 'AIC', ld + 2 * k / Te, 'BIC', ld + log(Te) * k / Te, ...
        'HQ', ld + 2 * log(log(Te)) * k / Te, 'portmanteau_p', portmanteau_p(U, p + extra, p));
    for i = 1:n
        row.(matlab.lang.makeValidName(['LB_', names{i}])) = cvar.ljungbox_p(U(:, i), extra);
    end
    rows{end + 1} = row; %#ok<AGROW>
end
D = struct2table([rows{:}]);
end

function pv = portmanteau_p(U, h, p)
U = U - mean(U, 1);
[T, n] = size(U);
C0i = inv(U' * U / T);
Q = 0;
for j = 1:h
    Cj = U(j + 1:end, :)' * U(1:end - j, :) / T;
    Q = Q + trace(Cj' * C0i * Cj * C0i) / (T - j);
end
Q = Q * T^2;
pv = cvar.chi2_sf(Q, n * n * (h - p));
end
