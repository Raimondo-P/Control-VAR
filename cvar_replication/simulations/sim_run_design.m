function rows = sim_run_design(d, T, reps, B, outdir, seed0)
%SIM_RUN_DESIGN  All replications of one design: coverage of gamma_h and theta_h by the 90 percent
%   percentile bands, mean width, bias and RMSE by scheme and horizon. Writes summary_<tag>.csv and
%   meta_<tag>.json in outdir and returns the rows.
HR = [0 1 2 3 6 12 24];
check_stable(d);
t0 = tic;
tr = sim_truth(d);
res = cell(reps, 1);
parfor i = 1:reps
    res{i} = sim_one_rep(d, T, B, seed0 + i - 1, tr);
end
tg = sim_tag(d, T);
rows = struct([]);
for k = {'oracle', 'fixed', 'refit', 'joint', 'jointrc', 'plainvar'}
    pts = cell2mat(cellfun(@(r) r.(k{1}).point, res, 'UniformOutput', false));
    lo = cell2mat(cellfun(@(r) r.(k{1}).lo, res, 'UniformOutput', false));
    hi = cell2mat(cellfun(@(r) r.(k{1}).hi, res, 'UniformOutput', false));
    for h = HR
        for tn = {'gamma', 'theta'}
            tv = tr.(tn{1})(h + 1);
            r = struct('design', string(tg), 'T', T, 'K', d.K, 'r', d.r, 'feedback', d.feedback, ...
                'constrained', d.constrained, 'boundary', d.boundary, 'lagrule', string(d.lagrule), ...
                'scheme', string(k{1}), 'h', h, 'target', string(tn{1}), 'truth', tv, ...
                'coverage', mean(lo(:, h + 1) <= tv & tv <= hi(:, h + 1)), 'width', mean(hi(:, h + 1) - lo(:, h + 1)), ...
                'bias', mean(pts(:, h + 1)) - tv, 'rmse', sqrt(mean((pts(:, h + 1) - tv).^2)));
            if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
        end
    end
end
ranks = cellfun(@(r) r.rank_coint, res); ps = cellfun(@(r) r.p, res);
meta = struct('design', tg, 'reps', reps, 'B', B, 'level', 0.90, 'true_coint_rank_controls', d.K - d.r, ...
    'freq_coint_rank', freq(ranks), 'lagrule', string(d.lagrule), 'mean_p', mean(ps), 'freq_p', freq(ps), ...
    'share_ljungbox_reject_cycle', mean(cellfun(@(r) r.lb_rej, res)), 'gamma', tr.gamma', 'theta', tr.theta', ...
    'b_star', tr.b_star', 'seconds', toc(t0));
if ~exist(outdir, 'dir'), mkdir(outdir); end
writetable(struct2table(rows), fullfile(outdir, ['summary_', tg, '.csv']));
fid = fopen(fullfile(outdir, ['meta_', tg, '.json']), 'w'); fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true)); fclose(fid);
fprintf('[%s] done in %.0f s\n', tg, meta.seconds);
end

function f = freq(x)
u = unique(x); f = struct();
for v = u(:)', f.(sprintf('v%d', v)) = mean(x == v); end
end

function check_stable(d)
% Policy-effect loop: W_t = 0.5 W_{t-1} + f E_{t-1} + ..., E_t = 0.8 E_{t-1} + theta0 W_t.
M = [0.5, d.feedback; 0.5 * d.theta0, 0.8 + d.theta0 * d.feedback];
if max(abs(eig(M))) >= 0.99
    error('design is explosive or near a unit root: lower feedback or theta0 (theta0 * feedback < 0.1)');
end
end
