function eb_states(cfg)
%EB_STATES  Every treated state in the differenced branch: cumulated response of the unemployment
%   rate to a ten-week EB innovation (EUC first), CVAR against the VAR without controls, 90 percent
%   percentile bands; each model with its rule lags ('rule') and both at the larger order
%   ('common'). Requires eb_lags. Output: results/eb/4_states.csv
if nargin < 1, cfg = config(); end
E = eb_load(cfg); out = eb_out(cfg); H = cfg.H; B = cfg.B_app;
L = readtable(fullfile(out, '3_lags.csv'), 'TextType', 'string');
sts = E.elig(ismember(E.elig, cellstr(L.st)));
n = numel(sts); res = cell(n, 1);
parfor i = 1:n
    s = sts{i}; k = find(L.st == s, 1); pc = L.p_cvar(k); pv = L.p_var(k);
    rows = struct([]);
    specs = {'rule', pc, pv; 'common', max(pc, pv), max(pc, pv)};
    for sp = 1:2
        bc = cvar.bootstrap(eb_fit_diff(E, s, specs{sp, 2}), B, H, struct('seed', 11, 'no_controls', false));
        bv = cvar.bootstrap(eb_fit_diff(E, s, specs{sp, 3}), B, H, struct('seed', 12, 'no_controls', true));
        [lc, hc] = cvar.bands(bc, 'cycle'); [lv, hv] = cvar.bands(bv, 'no_controls');
        for h = [0 6 12 24 36]
            r = struct('st', string(s), 'spec', string(specs{sp, 1}), 'h', h, 'p_cvar', specs{sp, 2}, 'p_var', specs{sp, 3}, ...
                'cvar', bc.point.cycle(h + 1, 1), 'cvar_lo', lc(h + 1, 1), 'cvar_hi', hc(h + 1, 1), 'ice', bc.point.ice(h + 1, 1), ...
                'var', bv.point.no_controls(h + 1, 1), 'var_lo', lv(h + 1, 1), 'var_hi', hv(h + 1, 1), ...
                'width_ratio', (hc(h + 1, 1) - lc(h + 1, 1)) / (hv(h + 1, 1) - lv(h + 1, 1)));
            if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
        end
    end
    res{i} = rows;
end
writetable(struct2table([res{:}]), fullfile(out, '4_states.csv'));
fprintf('written to %s\n', out);
end
