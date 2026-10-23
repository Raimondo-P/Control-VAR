function ca_a4(cfg)
%CA_A4  Does the LCFS cost react to the California-specific price cycle (Assumption 4)? (Python
%   ca_3_a4.py) For each control set, policy measure (spliced, CARB formula), sample (full, 2017
%   onward) and lag order (2, 6, 12):
%     granger_cycle    change in the cost on its own lags and lags of the change in the cycle; HAC Wald
%     granger_gap      the same with the lagged level of the gap; HAC Wald and the z-test on the gap
%     var_causality    Wald test that the cycle Granger-causes the policy in the VAR (policy, change in
%                      the cycle), statsmodels' test_causality(kind='wald'): the test reported
%     granger_controls the policy on lags of the controls' own price changes (context)
%   Output: results/california/G_a4_tests.csv
if nargin < 1, cfg = config(); end
K = ca_constants();
out = fullfile(cfg.results, 'california'); if ~exist(out, 'dir'), mkdir(out); end
[D, first] = ca_load(cfg);
rows = struct([]);
for cs = fieldnames(K.sets)'
    cc = K.sets.(cs{1});
    for meas = {'spliced', 'tv'}
        for smp = {'full', NaT; '2017_end', datetime(2017, 1, 1)}'
            Ds = ca_sub(D, smp{2}, NaT);
            k = ~isnan(Ds.(['cost_', meas{1}]));
            dts = Ds.dates(k); mon = month(dts);
            y = ca_sa_changes(100 * Ds.gas_ca(k), {mon});
            cb = zeros(sum(k), 1);
            for j = 1:numel(cc), cb = cb + ca_sa_changes(100 * Ds.(cc{j})(k), {mon}); end
            cb = cb / numel(cc);
            gap = y - cb; dg = [NaN; diff(gap)]; dc = [NaN; diff(cb)];
            lev = Ds.(['cost_', meas{1}])(k); dose = [NaN; diff(lev)];
            if strcmp(meas{1}, 'spliced') && ~isnat(first)
                i = find(dts == first, 1); if ~isempty(i) && i > 1, dose(i) = 0; end
            end
            for p = [2 6 12]
                nl = max(p, 3);
                base = lagm(dose, p);
                X = [base, lagm(dg, p)]; ok = ~isnan(dose) & all(~isnan(X), 2);
                [s1, p1] = hacwald(dose(ok), X(ok, :), p + 1:2 * p, nl);
                X2 = [base, lagm(dg, p), [NaN; gap(1:end - 1)]]; ok2 = ~isnan(dose) & all(~isnan(X2), 2);
                [s2, p2, tg, pg] = hacwald(dose(ok2), X2(ok2, :), p + 1:2 * p + 1, nl);
                Z = [dose, dg]; Z = Z(2:end, :);
                [s3, p3] = var_granger(Z, p);
                X4 = [base, lagm(dc, p)]; ok4 = ~isnan(dose) & all(~isnan(X4), 2);
                [s4, p4] = hacwald(dose(ok4), X4(ok4, :), p + 1:2 * p, nl);
                r = struct('controls', string(cs{1}), 'measure', string(meas{1}), 'sample', string(smp{1}), 'p', p, 'n', sum(ok), ...
                    'granger_cycle_stat', s1, 'granger_cycle_p', p1, 'granger_gap_stat', s2, 'granger_gap_p', p2, ...
                    'gap_t', tg, 'gap_p', pg, 'var_causality_stat', s3, 'var_causality_p', p3, ...
                    'granger_controls_stat', s4, 'granger_controls_p', p4);
                if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
            end
        end
    end
end
T = struct2table(rows); writetable(T, fullfile(out, 'G_a4_tests.csv'));
for c = {'granger_cycle_p', 'granger_gap_p', 'var_causality_p', 'granger_controls_p'}
    fprintf('  %s: %.2f of %d reject at 5 percent\n', c{1}, mean(T.(c{1}) <= 0.05), sum(~isnan(T.(c{1}))));
end
end

function L = lagm(v, p)
n = numel(v); L = nan(n, p);
for j = 1:p, L(j + 1:n, j) = v(1:n - j); end
end

function [st, pv, tlast, plast] = hacwald(y, X, idx, nl)
% Wald (chi-square) test that the coefficients on the columns idx of X are zero, OLS with a constant
% first and a HAC covariance; also the z-statistic of the last coefficient.
Xc = [ones(numel(y), 1), X];
[b, V] = ca_ols_hac(y, Xc, nl);
j = idx + 1;
st = b(j)' * (V(j, j) \ b(j));
pv = cvar.chi2_sf(st, numel(j));
tlast = b(end) / sqrt(V(end, end)); plast = 2 * cvar.norm_sf(abs(tlast));
end

function [st, pv] = var_granger(Z, p)
% statsmodels VARResults.test_causality(caused=0, causing=[1], kind='wald'): residual covariance with
% the degrees-of-freedom correction (T - K p - 1), chi-square with p degrees of freedom.
[A, U] = cvar.var_fit(Z, p);
[T, k] = size(Z); n = T - p;
X = ones(n, 1);
for j = 1:p, X = [X, Z(p - j + 1:T - j, :)]; end %#ok<AGROW>
Su = (U' * U) / (n - (k * p + 1));
iZZ = inv(X' * X);
rws = arrayfun(@(j) 1 + (j - 1) * k + 2, 1:p);       % coefficients of equation 1 on the lags of variable 2
b = A(rws, 1);
V = Su(1, 1) * iZZ(rws, rws);
st = b' * (V \ b);
pv = cvar.chi2_sf(st, p);
end
