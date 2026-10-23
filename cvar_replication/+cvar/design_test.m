function out = design_test(y, controls, policy, to)
%DESIGN_TEST  Test of the design (Section 5): point-optimal GLS test of the null that the treated
%   unit has a stochastic trend of its own, with the sieve-bootstrap p-value. Small values reject,
%   and rejection supports the CVAR in levels.
%   out = cvar.design_test(y, controls, policy, to), options to:
%   q        current value and q lags of the policies (12)
%   exog     further regressors of the weights regression, such as step dummies ([])
%   cbar     -7 (constant; tabulated critical values of Elliott, Rothenberg and Stock 1996)
%   B        bootstrap draws (999; 0 skips the bootstrap)
%   p_sieve  'rule' for ceil(16 (T/200)^(1/3)) or a number
%   pmax     largest lag of the modified AIC for the long-run variance (12)
%   seed     seed of the bootstrap draws
%   keep_draws  return the bootstrap statistics
if nargin < 3, policy = []; end
if nargin < 4, to = struct(); end
d = struct('q', 12, 'exog', [], 'cbar', -7, 'B', 999, 'p_sieve', 'rule', 'pmax', 12, 'seed', [], ...
    'keep_draws', false);
f = fieldnames(d);
for i = 1:numel(f), if ~isfield(to, f{i}), to.(f{i}) = d.(f{i}); end, end
if to.cbar ~= -7, error('critical values are tabulated for cbar = -7 with a constant'); end
y = y(:);
T0 = numel(y);
X = [ones(T0, 1), reshape(controls, T0, [])];
if ~isempty(policy), X = [X, cvar.lagmat(policy, 0:to.q)]; end
if ~isempty(to.exog), X = [X, reshape(to.exog, T0, [])]; end
ok = all(~isnan(X), 2);
y = y(ok); X = X(ok, :);
v = std(X, 1, 1);
X = [X(:, 1), X(:, [false, v(2:end) > 1e-12])];
T = numel(y);
[stat, k] = po_statistic(y, X, to.cbar, to.pmax);
if ischar(to.p_sieve) || isstring(to.p_sieve), ps = cvar.sieve_order(T); else, ps = to.p_sieve; end
draws = [];
if to.B > 0
    s = cvar.rng_stream(to.seed);
    [pval, draws] = sieve_pvalue(y, X, stat, to.B, s, ps, to.cbar, to.pmax, 100);
else
    pval = NaN;
end
cv = struct('p01', 1.99, 'p025', 2.55, 'p05', 3.26, 'p10', 4.48);
out = struct('statistic', stat, 'pvalue_bootstrap', pval, 'critical_values', cv, 'T', T, ...
    'p_sieve', ps, 'maic_lag', k, 'cbar', to.cbar, 'B', to.B);
if to.keep_draws, out.draws = draws; end
end

function s = ssr_qd(y, X, a)
yd = [y(1); y(2:end) - a * y(1:end - 1)];
Xd = [X(1, :); X(2:end, :) - a * X(1:end - 1, :)];
b = lsqminnorm(Xd, yd);
e = yd - Xd * b;
s = e' * e;
end

function u = gls_resid(y, X, a)
yd = [y(1); y(2:end) - a * y(1:end - 1)];
Xd = [X(1, :); X(2:end, :) - a * X(1:end - 1, :)];
b = lsqminnorm(Xd, yd);
u = y - X * b;
end

function [lrv_best, pbest] = maic_lrv(u, pmax)
% Autoregressive spectral estimator of the long-run variance from the ADF regression on the GLS
% residual, without deterministic terms, lag chosen by the modified AIC of Ng and Perron (2001).
du = diff(u);
n = numel(du) - pmax;
best = Inf; pbest = 0; lrv_best = NaN;
for p = 0:pmax
    Yv = du(pmax + 1:end);
    Zm = u(pmax + 1:end - 1);
    for j = 1:p
        Zm = [Zm, du(pmax - j + 1:end - j)]; %#ok<AGROW>
    end
    b = lsqminnorm(Zm, Yv);
    e = Yv - Zm * b;
    s2 = e' * e / n;
    ul = u(pmax + 1:end - 1);
    tau = b(1)^2 * (ul' * ul) / s2;
    crit = log(s2) + 2 * (tau + p) / n;
    if p > 0, lrv = s2 / (1 - sum(b(2:end)))^2; else, lrv = s2; end
    if crit < best
        best = crit; pbest = p; lrv_best = lrv;
    end
end
end

function [stat, k] = po_statistic(y, X, cbar, pmax)
T = numel(y);
a = 1 + cbar / T;
num = ssr_qd(y, X, a) - a * ssr_qd(y, X, 1);
[w2, k] = maic_lrv(gls_resid(y, X, a), pmax);
stat = num / w2;
end

function [pval, draws] = sieve_pvalue(y, X, stat, B, s, p, cbar, pmax, burn)
% Sieve bootstrap under the null: residuals of the regression in differences are fitted by an
% AR(p), its residuals resampled, and the treated outcome rebuilt by cumulating the fitted changes
% and the bootstrap errors, with the regressors held at their observed values.
dy = diff(y);
dX = diff(X, 1, 1);
keep = sum(abs(dX), 1) > 1e-12;
Zc = [ones(numel(dy), 1), dX(:, keep)];
g = lsqminnorm(Zc, dy);
fitv = Zc * g;
e = dy - fitv;
if p > 0
    Zp = zeros(numel(e) - p, p);
    for j = 1:p
        Zp(:, j) = e(p - j + 1:end - j);
    end
    phi = lsqminnorm(Zp, e(p + 1:end));
    rs = e(p + 1:end) - Zp * phi;
else
    phi = zeros(0, 1); rs = e;
end
rs = rs - mean(rs);
draws = zeros(B, 1);
for i = 1:B
    es = filter(1, [1; -phi], rs(randi(s, numel(rs), numel(e) + burn, 1)));
    es = es(burn + 1:end);
    ys = [y(1); y(1) + cumsum(fitv + es)];
    draws(i) = po_statistic(ys, X, cbar, pmax);
end
pval = (sum(draws <= stat) + 1) / (B + 1);
end
