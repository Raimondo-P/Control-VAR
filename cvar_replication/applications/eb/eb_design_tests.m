function eb_design_tests(cfg)
%EB_DESIGN_TESTS  Test of the design for every treated state: (a) estimated weights, policy terms
%   EUC and EB with q in {12, 6, 3, 0}, with and without automatic level shifts; (b) equal weights,
%   on Y1 - mean(controls), with q = 12 and without policy terms.
%   Outputs: results/eb/2a_estimated_weights.csv, 2b_imposed_weights.csv
if nargin < 1, cfg = config(); end
E = eb_load(cfg); out = eb_out(cfg); B = cfg.B_test;
items = {};
for s = E.elig, for q = [12 6 3 0], for b = {[], 'auto'}, items(end + 1, :) = {s{1}, q, b{1}}; end, end, end %#ok<AGROW>
n = size(items, 1); ra = cell(n, 1);
parfor i = 1:n
    [s, q, brk] = items{i, :};
    cc = eb_controls_of(E, s);
    r = cvar.fit(eb_col(E, E.UR, s), eb_col(E, E.UR, cc), eb_col(E, E.EB, s), struct('q', q, 'breaks', brk, 'other', eb_col(E, E.EUC, s)));
    u = r.units(1); if isempty(u.steps), ex = []; else, ex = u.steps; end
    t = cvar.design_test(r.Y, r.controls{1}, r.pol, struct('q', q, 'exog', ex, 'B', B, 'seed', 7));
    if isempty(brk), sh = "none"; else, sh = "auto"; end
    ra{i} = struct('st', string(s), 'q', q, 'shifts', sh, 'n_shifts', numel(u.breaks), 'K', numel(cc), ...
        'P_T', t.statistic, 'p_boot', t.pvalue_bootstrap);
end
writetable(struct2table([ra{:}]), fullfile(out, '2a_estimated_weights.csv'));
ne = numel(E.elig); rb = cell(ne, 1); el = E.elig;
parfor i = 1:ne
    s = el{i}; cc = eb_controls_of(E, s);
    g = eb_col(E, E.UR, s) - mean(eb_col(E, E.UR, cc), 2);
    pol = [eb_col(E, E.EUC, s), eb_col(E, E.EB, s)];
    t = cvar.design_test(g, zeros(numel(g), 0), pol, struct('q', 12, 'B', B, 'seed', 5));
    t0 = cvar.design_test(g, zeros(numel(g), 0), [], struct('q', 0, 'B', B, 'seed', 5));
    rb{i} = struct('st', string(s), 'P_T', t.statistic, 'p_boot', t.pvalue_bootstrap, ...
        'P_T_nopolicy', t0.statistic, 'p_boot_nopolicy', t0.pvalue_bootstrap);
end
writetable(struct2table([rb{:}]), fullfile(out, '2b_imposed_weights.csv'));
fprintf('written to %s\n', out);
end
