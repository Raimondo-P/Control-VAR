function eb_lags(cfg)
%EB_LAGS  Lag order of the differenced branch by the rule (smallest order with no Ljung-Box
%   rejection at 5 percent in the EB and outcome equations), for the CVAR and for the VAR without
%   controls; AIC and BIC for reference. Output: results/eb/3_lags.csv
if nargin < 1, cfg = config(); end
E = eb_load(cfg); out = eb_out(cfg); rows = struct([]);
for s = E.elig
    r = eb_fit_diff(E, s{1}, 1);
    Zc = r.Z; Zv = r.nc.Z;
    dc = cvar.lag_diagnostics(Zc, 1:12); dv = cvar.lag_diagnostics(Zv, 1:12);
    [~, ac] = min(dc.AIC); [~, bc] = min(dc.BIC); [~, av] = min(dv.AIC); [~, bv] = min(dv.BIC);
    row = struct('st', string(s{1}), 'p_cvar', cvar.lag_rule(Zc, [], [2 3], 12), 'p_var', cvar.lag_rule(Zv, [], [2 3], 12), ...
        'aic_cvar', dc.p(ac), 'bic_cvar', dc.p(bc), 'aic_var', dv.p(av), 'bic_var', dv.p(bv));
    if isempty(rows), rows = row; else, rows(end + 1) = row; end %#ok<AGROW>
end
T = struct2table(rows); writetable(T, fullfile(out, '3_lags.csv')); disp(T);
end
