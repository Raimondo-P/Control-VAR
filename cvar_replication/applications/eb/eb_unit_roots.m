function eb_unit_roots(cfg)
%EB_UNIT_ROOTS  Point-optimal test (constant, cbar = -7) with the sieve bootstrap on the
%   unemployment rate of every treated state and every control. Output: results/eb/1_unit_roots.csv
if nargin < 1, cfg = config(); end
E = eb_load(cfg); out = eb_out(cfg);
sts = sort(union(E.elig, E.ctrl)); n = numel(sts); rows = cell(n, 1); B = cfg.B_test;
parfor i = 1:n
    x = eb_col(E, E.UR, sts{i});
    t = cvar.design_test(x, zeros(numel(x), 0), [], struct('q', 0, 'B', B, 'seed', 1));
    rows{i} = struct('state', string(sts{i}), 'P_T', t.statistic, 'p_boot', t.pvalue_bootstrap, 'reject5', t.pvalue_bootstrap <= 0.05);
end
T = struct2table([rows{:}]); writetable(T, fullfile(out, '1_unit_roots.csv')); disp(T);
fprintf('share rejecting at 5%%: %.3f\n', mean(T.reject5));
end
