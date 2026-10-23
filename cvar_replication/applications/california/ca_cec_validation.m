function ca_cec_validation(cfg)
%CA_CEC_VALIDATION  The LCFS cost per gallon reported by refiners to the California Energy
%   Commission (cec_lcfs_monthly.csv) against the cost implied by the CARB formula and by the
%   constant factor: correlations in levels and changes and OLS (HAC, 3 lags) over the overlapping
%   months (validation part of Python ca_1_cec.py).
%   The CEC series is shipped as applications/data/california/cec_lcfs_monthly.csv, parsed from the
%   CEC spreadsheet "Estimated Gasoline Price Breakdown and Margins" (SB 1322,
%   https://www.energy.ca.gov/estimated-gasoline-price-breakdown-and-margins), row "Low Carbon Fuel
%   Standard", in cents per gallon.
%   Outputs (results/california): 1_cec_validation.csv, 0_formula_factor.csv
if nargin < 1, cfg = config(); end
K = ca_constants();
out = fullfile(cfg.results, 'california'); if ~exist(out, 'dir'), mkdir(out); end
D = ca_load(cfg);
ov = ~isnan(D.cost_cec);
c = D.cost_cec(ov);
rows = struct([]);
for m = {'cost_tv', 'cost_const'}
    x = D.(m{1})(ov);
    [b, V] = ca_ols_hac(c, [ones(numel(c), 1), x], 3);
    [bd, Vd] = ca_ols_hac(diff(c), [ones(numel(c) - 1, 1), diff(x)], 3);
    r = struct('measure', string(m{1}), 'months', numel(c), 'corr_levels', corrc(c, x), 'corr_changes', corrc(diff(c), diff(x)), ...
        'slope_levels', b(2), 'slope_levels_se', sqrt(V(2, 2)), 'intercept', b(1), 'slope_changes', bd(2), ...
        'slope_changes_se', sqrt(Vd(2, 2)), 'mean_cec', mean(c), 'mean_measure', mean(x), 'months_2024', NaN, 'mean_abs_gap_2024', NaN);
    pre = D.dates(ov) < datetime(2025, 1, 1);
    if sum(pre) > 3, r.months_2024 = sum(pre); r.mean_abs_gap_2024 = mean(abs(c(pre) - x(pre))); end
    if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
end
writetable(struct2table(rows), fullfile(out, '1_cec_validation.csv'));
yrs = 2014:2026;
fac = table(yrs', arrayfun(@(y) K.ci_std(y), yrs)', arrayfun(@(y) K.ci_carbob(y), yrs)', ...
    arrayfun(@(y) ca_t_per_gal(y, K), yrs)', arrayfun(@(y) 1e4 * ca_t_per_gal(y, K), yrs)', ...
    'VariableNames', {'year', 'CI_benchmark', 'CI_CARBOB', 't_per_gal', 'cents_per_gal_at_100_usd'});
writetable(fac, fullfile(out, '0_formula_factor.csv'));
disp(struct2table(rows)); disp(fac);
end

function r = corrc(a, b)
C = corrcoef(a, b); r = C(1, 2);
end
