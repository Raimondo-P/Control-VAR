function ak_screen(cfg)
%AK_SCREEN  Screening of the design (Python ak_0_screen.py): the test of the design for each
%   outcome, control set and sample, with estimated weights (with and without automatic level
%   shifts), equal weights, and Alaska alone.
%   Outputs (results/alaska): 0_screen.csv, 0_weights.csv
if nargin < 1, cfg = config(); end
K = ak_constants();
out = fullfile(cfg.results, 'alaska'); if ~exist(out, 'dir'), mkdir(out); end
D = ak_load(cfg, true);
B = cfg.B_test;
items = {};
for o = K.outcomes
    for si = 1:size(K.samples_screen, 1)
        smp = K.samples_screen{si, 1}; last = K.samples_screen{si, 2};
        items(end + 1, :) = {o{1}, smp, 'none', {}, 'alaska_alone', last}; %#ok<AGROW>
        for cs = fieldnames(K.sets)'
            cc = K.sets.(cs{1});
            A = ak_arrays(D, o{1}, cc, last);
            if numel(A.y) < 120
                fprintf('  skipped %s %s %s: %d months\n', o{1}, smp, cs{1}, numel(A.y)); continue
            end
            for sp = {'estimated', 'estimated_shifts', 'equal'}
                items(end + 1, :) = {o{1}, smp, cs{1}, cc, sp{1}, last}; %#ok<AGROW>
            end
        end
    end
end
n = size(items, 1);
fprintf('%d tests, B = %d\n', n, B);
rows = cell(n, 1); wts = cell(n, 1);
parfor i = 1:n
    [rows{i}, wts{i}] = one_test(D, items(i, :), B, K.q);
end
R = struct2table(vertcat(rows{:}));
writetable(R, fullfile(out, '0_screen.csv'));
Wt = [wts{:}];                        % one 1-by-K row per test, K differing across control sets
if ~isempty(Wt), writetable(struct2table(Wt), fullfile(out, '0_weights.csv')); end
disp(R(:, {'outcome', 'sample', 'controls', 'spec', 'T', 'stat', 'p'}));
fprintf('written to %s\n', out);
end

function [row, wt] = one_test(D, it, B, q)
[outcome, sample, cs, cc, spec, last] = it{:};
A = ak_arrays(D, outcome, cc, last);
T = numel(A.y);
row = struct('outcome', string(outcome), 'sample', string(sample), 'controls', string(cs), ...
    'spec', string(spec), 'T', T, 'start', string(datestr(A.dates(1), 'yyyy-mm')), ...
    'end', string(datestr(A.dates(end), 'yyyy-mm')), 'stat', NaN, 'p', NaN, 'sum_weights', NaN, ...
    'weights', "", 'rank', NaN, 'shifts', "", 'error', "");
wt = struct('outcome', {}, 'sample', {}, 'controls', {}, 'spec', {}, 'control', {}, 'weight', {});
try
    pol = [A.doil, A.W];
    switch spec
        case {'estimated', 'estimated_shifts'}
            if strcmp(spec, 'estimated_shifts'), br = 'auto'; else, br = []; end
            r = cvar.fit(A.y, A.C, A.W, struct('q', q, 'breaks', br, 'other', A.doil));
            u = r.units(1);
            if isempty(u.steps), ex = []; else, ex = u.steps; end
            t = cvar.design_test(A.y, A.C, r.pol, struct('q', q, 'exog', ex, 'B', B, 'seed', 7));
            [bs, ord] = sort(u.b, 'descend');
            lab = strjoin(arrayfun(@(k) sprintf('%s %.2f', cc{ord(k)}, bs(k)), find(bs > 0.005)', ...
                'UniformOutput', false), ', ');
            row.stat = t.statistic; row.p = t.pvalue_bootstrap; row.sum_weights = sum(u.b);
            row.weights = string(lab); row.rank = u.rank;
            if ~isempty(u.breaks)
                row.shifts = string(strjoin(cellstr(datestr(A.dates(u.breaks), 'yyyy-mm')), ' '));
            end
            for k = 1:numel(cc)
                wt(k) = struct('outcome', string(outcome), 'sample', string(sample), 'controls', string(cs), ...
                    'spec', string(spec), 'control', string(cc{k}), 'weight', u.b(k));
            end
        case 'equal'
            g = A.y - mean(A.C, 2);
            t = cvar.design_test(g, zeros(T, 0), pol, struct('q', q, 'B', B, 'seed', 5));
            row.stat = t.statistic; row.p = t.pvalue_bootstrap;
        otherwise
            t = cvar.design_test(A.y, zeros(T, 0), pol, struct('q', q, 'B', B, 'seed', 5));
            row.stat = t.statistic; row.p = t.pvalue_bootstrap;
    end
catch e
    row.error = string(e.message);
end
end
