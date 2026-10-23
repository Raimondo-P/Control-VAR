function [m, D1] = rank_step(Y0, method, level, k_ar_diff, maxlags)
%RANK_STEP  Cointegrating relations among the controls and the directions of their trends.
%   [m, D1] = cvar.rank_step(Y0, method, level, k_ar_diff, maxlags)
%   m:  number of cointegrating relations (r = K - m common trends)
%   D1: K x (K - m) basis of the orthogonal complement of the cointegrating vectors (eye(K)
%       when m = 0, zeros(K, 0) when every control is stationary).
%   method: 'johansen' (trace test at `level`: 0.10, 0.05 or 0.01), 'bic' (information criterion
%   on the eigenvalues with penalty log T, Cheng and Phillips 2009), or a number fixing m.
%   k_ar_diff: 'aic' (at least 1) or the number of lagged differences.
if nargin < 2 || isempty(method), method = 'johansen'; end
if nargin < 3 || isempty(level), level = 0.05; end
if nargin < 4 || isempty(k_ar_diff), k_ar_diff = 'aic'; end
if nargin < 5 || isempty(maxlags), maxlags = 12; end
K = size(Y0, 2);
if K == 1
    m = 0; D1 = 1; return
end
if ischar(k_ar_diff) || isstring(k_ar_diff)
    k = max(cvar.select_order_aic(Y0, maxlags), 1);
else
    k = k_ar_diff;
end
jo = cvar.johansen(Y0, k);
if isnumeric(method)
    m = method;
elseif strcmp(method, 'johansen')
    col = find(abs([0.10 0.05 0.01] - level) < 1e-12, 1);
    m = find(jo.lr1 < jo.cvt(:, col), 1) - 1;
    if isempty(m), m = K; end
elseif strcmp(method, 'bic')
    Teff = size(Y0, 1) - k - 1;
    lam = min(max(sort(jo.eig, 'descend'), 0), 1 - 1e-12);
    ic = zeros(K + 1, 1);
    for mm = 0:K
        ic(mm + 1) = Teff * sum(log(1 - lam(mm + 1:end))) + log(Teff) * (2 * K * mm - mm^2);
    end
    [~, i] = min(ic);
    m = i - 1;
else
    error('method must be ''johansen'', ''bic'' or a number');
end
if m >= K
    m = K; D1 = zeros(K, 0); return
end
if m == 0
    D1 = eye(K);
else
    D1 = null(jo.evec(:, 1:m)');
end
end
