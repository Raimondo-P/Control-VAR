function run_simulations(cfg, steps)
%RUN_SIMULATIONS  The simulations of the supplementary appendix. steps: cell array of step names
%   (default all available): sim1, lags, grid, joint, diff. The numbers of replications and draws
%   are in config.m. The grid is by far the longest step.
if nargin < 1 || isempty(cfg), cfg = config(); end
if nargin < 2 || isempty(steps), steps = {'sim1', 'lags', 'grid', 'joint', 'diff'}; end
for s = steps
    t0 = tic; fprintf('\n=== %s\n', s{1});
    switch s{1}
        case {'sim1', 'lags', 'grid'}, sim_estimator(cfg, s{1});
        case 'joint', sim_estimator(cfg, 'joint'); sim_joint_table(cfg);
        case 'diff', sim_diff(cfg);
        otherwise, error('unknown step %s', s{1});
    end
    fprintf('=== %s done in %.0f s\n', s{1}, toc(t0));
end
end
