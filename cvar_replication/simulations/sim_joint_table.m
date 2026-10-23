function sim_joint_table(cfg)
%SIM_JOINT_TABLE  LaTeX table of the bootstrap schemes with estimated weights (tab:mc-joint), from the
%   outputs of sim_estimator(cfg, 'joint'). Output: results/paper/tab_mc_joint.tex
if nargin < 1, cfg = config(); end
RUNS = {'mc_joint_fb03', '$T=200$, $\phi=0.3$'; 'mc_joint_fb045', '$T=200$, $\phi=0.45$'; 'mc_joint_T400', '$T=400$, $\phi=0.3$'};
SCH = {'oracle', 'Known weights'; 'fixed', 'Fixed weights'; 'refit', 'Re-estimated, controls fixed'; ...
       'joint', 'Joint, not recentred'; 'jointrc', 'Joint, recentred (Theorem \ref{thm:joint})'};
H = [0 1 3 6 12 24]; lines = {};
for k = 1:size(RUNS, 1)
    f = fullfile(cfg.results, 'simulations', RUNS{k, 1}, 'all_summaries.csv');
    if ~exist(f, 'file'), fprintf('missing %s\n', f); continue, end
    d = readtable(f, 'TextType', 'string'); d = d(d.target == "gamma", :);
    lines{end + 1} = ['\multicolumn{8}{l}{', RUNS{k, 2}, '} \\']; %#ok<AGROW>
    for j = 1:size(SCH, 1)
        z = d(d.scheme == SCH{j, 1}, :);
        if isempty(z), continue, end
        c = arrayfun(@(h) sprintf('%.2f', z.coverage(z.h == h)), H, 'UniformOutput', false);
        lines{end + 1} = [strjoin([SCH(j, 2), c, {sprintf('%.2f', z.width(z.h == 12))}], ' & '), ' \\']; %#ok<AGROW>
    end
    lines{end + 1} = '\midrule'; %#ok<AGROW>
end
if ~isempty(lines), lines = lines(1:end - 1); end
note = sprintf(['Six controls with one cointegrating relation among them, feedback $\\phi$ from the lagged cycle to ', ...
    'the policy, 12 lags in the VAR of the CVAR and in the joint VAR, %d replications and %d bootstrap draws. Known ', ...
    'weights: pseudo-true weights. Fixed weights: estimated weights held fixed. Re-estimated, controls fixed: weights ', ...
    're-estimated in every draw with the controls at their observed values. Joint: the controls are regenerated with ', ...
    'the policies and the cycle and the weights re-estimated, without recentring, or with the draws recentred at the ', ...
    'response of the bootstrap world as in Theorem \\ref{thm:joint}.'], cfg.sim_reps, cfg.sim_boot);
t = [{'\begin{table}[htbp]', '\centering', ...
      '\caption{Coverage of the 90 percent bands for $\gamma_h$ with estimated weights: bootstrap schemes\label{tab:mc-joint}}', ...
      '\small', '\begin{tabular}{lccccccc}', '\toprule', ...
      ' & \multicolumn{6}{c}{Coverage, months after the surprise} & Width \\', '\cmidrule(lr){2-7}\cmidrule(lr){8-8}', ...
      [' & ', strjoin(arrayfun(@num2str, H, 'UniformOutput', false), ' & '), ' & 12 \\'], '\midrule'}, lines, ...
     {'\bottomrule', '\end{tabular}', ['\par\smallskip\parbox{0.95\textwidth}{\footnotesize ', note, '}'], '\end{table}'}];
op = cfg.paper; if ~exist(op, 'dir'), mkdir(op); end
fid = fopen(fullfile(op, 'tab_mc_joint.tex'), 'w'); fprintf(fid, '%s\n', t{:}); fclose(fid);
fprintf('tab_mc_joint.tex written\n');
end
