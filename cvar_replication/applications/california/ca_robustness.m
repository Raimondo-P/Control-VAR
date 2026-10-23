function ca_robustness(cfg, sections)
%CA_ROBUSTNESS  Pass-through of the LCFS cost to the California retail gasoline price (Python
%   ca_2_robustness.py; the delta-method bands of the Python version, used in no exhibit, are left out).
%   B_main.csv     control sets x policy measures (spliced, spliced with a January dummy, tv, tv with
%                  a January dummy, const): CVAR and VAR without controls, percentile and Hall bands,
%                  width ratios, design-test p-values
%   C_lags.csv     both models at every lag order 1 to 12 (Gulf, PADDs 1 to 4)
%   D_samples.csv  2014-2019, 2014-2024, 2017 onward, 2020 onward, full, full with 2020-2022 dummied
%   E_controls.csv PADDs 1 to 4 leaving one region out
%   F_lp.csv       local projections (Newey-West): linear, increases and decreases, high and low credit price
%   sections: subset of 'BCDEF' (default all).
if nargin < 1 || isempty(cfg), cfg = config(); end
if nargin < 2, sections = 'BCDEF'; end
K = ca_constants(); H = K.H; HOR = K.horizons; B = cfg.B_app; Bt = cfg.B_test;
out = fullfile(cfg.results, 'california'); if ~exist(out, 'dir'), mkdir(out); end
[D, first] = ca_load(cfg);
fprintf('sample %s to %s; CEC cost from %s\n', datestr(D.dates(1), 'yyyy-mm'), datestr(D.dates(end), 'yyyy-mm'), datestr(first, 'yyyy-mm'));
sets = fieldnames(K.sets)';
dp = @(S) cvar.design_test(S.y, S.C, S.dose, struct('q', 12, 'B', Bt, 'seed', 2)).pvalue_bootstrap;
if any(sections == 'B')
    items = {};
    for cs = sets, for meas = {'spliced', 'spliced_jan', 'tv', 'tv_jan', 'const'}, items(end + 1, :) = {cs{1}, meas{1}}; end, end %#ok<AGROW>
    R = cell(size(items, 1), 1);
    parfor i = 1:size(items, 1)
        [cs, meas] = items{i, :};
        parts = strsplit(meas, '_');
        S = ca_series(D, K.sets.(cs), parts{1}, first);
        if endsWith(meas, '_jan'), ve = double(month(S.dates) == 1); else, ve = []; end
        [pc, pv] = ca_rule_lags(S, ve);
        est = ca_estimate(S, pc, pv, B, ve, 'differenced', 1, H);
        R{i} = ca_rows(sprintf('%s_%s', cs, meas), est, struct('controls', string(cs), 'measure', string(meas), ...
            'p_cvar', pc, 'p_var', pv, 'design_p', dp(S)), HOR);
    end
    writetable(struct2table([R{:}]), fullfile(out, 'B_main.csv'));
end
if any(sections == 'C')
    items = {};
    for cs = {'gulf', 'padd1_4'}, for p = 1:12, items(end + 1, :) = {cs{1}, p}; end, end %#ok<AGROW>
    R = cell(size(items, 1), 1); Bc = max(199, floor(B / 2));
    parfor i = 1:size(items, 1)
        [cs, p] = items{i, :};
        S = ca_series(D, K.sets.(cs), 'spliced', first);
        est = ca_estimate(S, p, p, Bc, [], 'differenced', p, H);
        R{i} = ca_rows(sprintf('%s_p%d', cs, p), est, struct('controls', string(cs), 'p', p), HOR);
    end
    writetable(struct2table([R{:}]), fullfile(out, 'C_lags.csv'));
end
if any(sections == 'D')
    smp = {'2014_2019', datetime(2014, 1, 1), datetime(2019, 12, 1); '2014_2024', datetime(2014, 1, 1), datetime(2024, 12, 1); ...
           '2017_end', datetime(2017, 1, 1), NaT; '2020_end', datetime(2020, 1, 1), NaT; 'full', NaT, NaT};
    R = {};
    for cs = {'gulf', 'padd1_4'}
        for k = 1:size(smp, 1)
            Ds = ca_sub(D, smp{k, 2}, smp{k, 3});
            S = ca_series(Ds, K.sets.(cs{1}), 'spliced', first);
            d = dp(S);
            if d <= 0.05, branch = 'levels'; else, branch = 'differenced'; end
            [pc, pv] = ca_rule_lags(S);
            for lt = {'rule', pc, pv; 'p2', 2, 2}'
                est = ca_estimate(S, lt{2}, lt{3}, B, [], branch, 1, H);
                R{end + 1} = ca_rows(sprintf('%s_%s_%s', cs{1}, smp{k, 1}, lt{1}), est, struct('controls', string(cs{1}), ...
                    'sample', string(smp{k, 1}), 'lags', string(lt{1}), 'p_cvar', lt{2}, 'p_var', lt{3}, ...
                    'T', numel(Ds.dates), 'design_p', d, 'branch', string(branch)), HOR); %#ok<AGROW>
            end
            fprintf('  %s %s: T %d, design p %.3f (%s), rule lags %d/%d\n', cs{1}, smp{k, 1}, numel(Ds.dates), d, branch, pc, pv);
        end
        S = ca_series(D, K.sets.(cs{1}), 'spliced', first);
        ex = find(S.dates >= datetime(2020, 1, 1) & S.dates <= datetime(2022, 12, 1));
        ve = zeros(numel(S.dates), numel(ex)); for j = 1:numel(ex), ve(ex(j), j) = 1; end
        est = ca_estimate(S, 2, 2, B, ve, 'differenced', 1, H);
        R{end + 1} = ca_rows(sprintf('%s_dummies2020_22_p2', cs{1}), est, struct('controls', string(cs{1}), ...
            'sample', "full, 2020-22 dummied", 'lags', "p2", 'p_cvar', 2, 'p_var', 2, 'T', NaN, 'design_p', NaN, 'branch', ""), HOR); %#ok<AGROW>
    end
    writetable(struct2table([R{:}]), fullfile(out, 'D_samples.csv'));
end
if any(sections == 'E')
    R = {};
    for drop = K.sets.padd1_4
        cc = K.sets.padd1_4(~strcmp(K.sets.padd1_4, drop{1}));
        S = ca_series(D, cc, 'spliced', first);
        [pc, pv] = ca_rule_lags(S);
        est = ca_estimate(S, pc, pv, B, [], 'differenced', 1, H);
        R{end + 1} = ca_rows(['without_', drop{1}], est, struct('dropped', string(drop{1}), 'p_cvar', pc, 'p_var', pv), HOR); %#ok<AGROW>
    end
    writetable(struct2table([R{:}]), fullfile(out, 'E_controls.csv'));
end
if any(sections == 'F')
    R = struct([]);
    for cs = {'gulf', 'padd1_4'}
        S = ca_series(D, K.sets.(cs{1}), 'spliced', first);
        g = S.y - mean(S.C, 2); dose = S.dose; cr = S.credit;
        hi = [0; double(cr(1:end - 1) > median(cr))];
        regs = {'linear', {dose}; 'increase_decrease', {max(dose, 0), min(dose, 0)}; 'high_low_credit', {dose .* hi, dose .* (1 - hi)}};
        for oc = {'cvar_cycle', g; 'no_controls', S.y}'
            s = oc{2}; ds = [NaN; diff(s)]; n = numel(s);
            for p = [2 6]
                lagX = zeros(n, 2 * p);
                for j = 1:p, lagX(:, j) = [nan(j, 1); ds(1:end - j)]; lagX(:, p + j) = [nan(j, 1); dose(1:end - j)]; end
                for kr = 1:size(regs, 1)
                    X0 = cell2mat(regs{kr, 2});
                    for h = HOR
                        lhs = nan(n, 1); lhs(2:n - h) = s(2 + h:n) - s(1:n - h - 1);
                        X = [X0, lagX];
                        ok = ~isnan(lhs) & all(~isnan(X), 2);
                        [b, V] = ca_ols_hac(lhs(ok), [ones(sum(ok), 1), X(ok, :)], h + 1);
                        for j = 1:size(X0, 2)
                            se = sqrt(V(1 + j, 1 + j));
                            r = struct('controls', string(cs{1}), 'outcome', string(oc{1}), 'p', p, 'kind', string(regs{kr, 1}), ...
                                'term', string(sprintf('coef%d', j)), 'h', h, 'coef', b(1 + j), 'lo', b(1 + j) - 1.645 * se, ...
                                'hi', b(1 + j) + 1.645 * se, 'n', sum(ok));
                            if isempty(R), R = r; else, R(end + 1) = r; end %#ok<AGROW>
                        end
                    end
                end
            end
        end
    end
    writetable(struct2table(R), fullfile(out, 'F_lp.csv'));
end
fprintf('written to %s\n', out);
end
