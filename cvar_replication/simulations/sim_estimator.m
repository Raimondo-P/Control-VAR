function sim_estimator(cfg, step)
%SIM_ESTIMATOR  The estimator and bootstrap simulations of the supplementary appendix (Python
%   mc_bootstrap_cvar.py as called by run_all.py):
%     'sim1'  VAR without controls against the CVAR, T = 200, K = 6, r = 5, feedback 0.3, 12 lags (tab:mc-var)
%     'lags'  the same with AIC, 8 and 12 lags (tab:mc-lags)
%     'grid'  17 designs: T in {200, 400}, (K, r) in {(3,3), (3,2), (6,6), (6,5)}, feedback in {0, 0.3},
%             and the boundary design, 12 lags (tab:mc-boot, tab:mc-grid)
%     'joint' feedback 0.3 and 0.45 at T = 200 and feedback 0.3 at T = 400 (tab:mc-joint)
%   Outputs: results/simulations/<run>/summary_*.csv, meta_*.json, all_summaries.csv
if nargin < 1 || isempty(cfg), cfg = config(); end
base = struct('c_eta', 0.5, 'theta0', 0.2, 'constrained', true, 'boundary', false, 'lagrule', '12', 'joint_lags', [], 'recentre_mult', 50);
mk = @(K, r, fb, lr) setf(base, 'K', K, 'r', r, 'feedback', fb, 'lagrule', lr);
switch step
    case 'sim1'
        runs = {'mc_sim1', {mk(6, 5, 0.3, '12'), 200}};
    case 'lags'
        runs = {'mc_lag_aic', {mk(6, 5, 0.3, 'aic'), 200}; 'mc_lag_8', {mk(6, 5, 0.3, '8'), 200}; 'mc_lag_12', {mk(6, 5, 0.3, '12'), 200}};
    case 'grid'
        ds = {};
        for T = [200 400], for kr = [3 3; 3 2; 6 6; 6 5]', for fb = [0 0.3]
            ds(end + 1, :) = {mk(kr(1), kr(2), fb, '12'), T}; %#ok<AGROW>
        end, end, end
        bd = mk(3, 3, 0.3, '12'); bd.boundary = true; ds(end + 1, :) = {bd, 200};
        runs = {'mc_grid_p12', ds};
    case 'joint'
        runs = {'mc_joint_fb03', {mk(6, 5, 0.3, '12'), 200}; 'mc_joint_fb045', {mk(6, 5, 0.45, '12'), 200}; ...
                'mc_joint_T400', {mk(6, 5, 0.3, '12'), 400}};
    otherwise
        error('step must be sim1, lags, grid or joint');
end
for k = 1:size(runs, 1)
    od = fullfile(cfg.results, 'simulations', runs{k, 1});
    ds = runs{k, 2};                     % designs: one row per design, {settings, T}
    allr = struct([]);
    for i = 1:size(ds, 1)
        rr = sim_run_design(ds{i, 1}, ds{i, 2}, cfg.sim_reps, cfg.sim_boot, od, 20261012 + 100000 * (i - 1));
        if isempty(allr), allr = rr; else, allr = [allr, rr]; end %#ok<AGROW>
    end
    writetable(struct2table(allr), fullfile(od, 'all_summaries.csv'));
end
end

function s = setf(s, varargin)
for i = 1:2:numel(varargin), s.(varargin{i}) = varargin{i + 1}; end
end
