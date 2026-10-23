function eb_exhibits(cfg)
%EB_EXHIBITS  Figures and LaTeX tables of the extended benefits application (Python eb_6_exhibits.py).
%   Requires eb_unit_roots, eb_design_tests, eb_lags, eb_states. Re-estimates the average across
%   states with the joint bootstrap for the full paths.
%   Outputs (results/paper): fig_eb_average.pdf/.png/.csv, fig_eb_states.pdf/.png, tab_eb_tests.tex,
%   tab_eb_average.tex, tab_eb_bounds.tex, tab_eb_states.tex
if nargin < 1, cfg = config(); end
E = eb_load(cfg); src = eb_out(cfg); op = cfg.paper; if ~exist(op, 'dir'), mkdir(op); end
rd = @(f) readtable(fullfile(src, f), 'TextType', 'string');
UR1 = rd('1_unit_roots.csv'); T2a = rd('2a_estimated_weights.csv'); T2b = rd('2b_imposed_weights.csv'); S4 = rd('4_states.csv');
H = cfg.H; B = cfg.B_app;
[p, sts] = eb_common_lag(E, src); N = numel(sts);
fprintf('%d states, common lag order %d\n', N, p);
rs = cellfun(@(s) eb_fit_diff(E, s, p), sts, 'UniformOutput', false);
[~, avg] = cvar.joint_bootstrap(rs, B, H, 21);
[~, avgc] = cvar.joint_controls_response(rs, B, H, 22);
keys = fieldnames(avg.point)'; ckeys = fieldnames(avgc.point)';
C = table((0:H)', 'VariableNames', {'h'});
for k = keys
    [l9, h9] = cvar.bands(avg, k{1}, 0.10); [l6, h6] = cvar.bands(avg, k{1}, 0.32);
    C.(k{1}) = avg.point.(k{1})(:, 1); C.([k{1}, '_lo90']) = l9(:, 1); C.([k{1}, '_hi90']) = h9(:, 1);
    C.([k{1}, '_lo68']) = l6(:, 1); C.([k{1}, '_hi68']) = h6(:, 1);
end
for k = ckeys
    [l9, h9] = cvar.bands(avgc, k{1}, 0.10);
    C.(['ctrl_', k{1}]) = avgc.point.(k{1})(:, 1); C.(['ctrl_', k{1}, '_lo90']) = l9(:, 1); C.(['ctrl_', k{1}, '_hi90']) = h9(:, 1);
end
writetable(C, fullfile(op, 'fig_eb_average.csv'));
draw_average(C, H, op);
draw_states(S4, sts, op);

% ------------------------------------------------------------------ tests
rows = {};
et = UR1(ismember(UR1.state, sts), :); ec = UR1(ismember(UR1.state, E.ctrl), :);
rows{end + 1} = sprintf('Unemployment rate, treated states & ERS & %d of %d & %s \\\\', sum(et.reject5), height(et), f2(median(et.p_boot)));
rows{end + 1} = sprintf('Unemployment rate, controls & ERS & %d of %d & %s \\\\', sum(ec.reject5), height(ec), f2(median(ec.p_boot)));
rows{end + 1} = '\midrule';
for sh = {'none', 'no level shifts'; 'auto', 'level shifts'}'
    for q = [12 6 3 0]
        z = T2a(T2a.shifts == sh{1} & T2a.q == q, :);
        rows{end + 1} = sprintf('Cycle, estimated weights, %s & $q=%d$ & %d of %d & %s \\\\', sh{2}, q, sum(z.p_boot <= 0.05), height(z), f2(median(z.p_boot))); %#ok<AGROW>
    end
end
rows{end + 1} = '\midrule';
rows{end + 1} = sprintf('Cycle, equal weights & $q=12$ & %d of %d & %s \\\\', sum(T2b.p_boot <= 0.05), height(T2b), f2(median(T2b.p_boot)));
rows{end + 1} = sprintf('Cycle, equal weights, no policy terms & & %d of %d & %s \\\\', sum(T2b.p_boot_nopolicy <= 0.05), height(T2b), f2(median(T2b.p_boot_nopolicy)));
wtab(op, 'tab_eb_tests.tex', 'Unit roots and the test of the design', 'tab:eb-tests', 'llcc', ...
    'Series & Specification & Rejections at 5 percent & Median p-value', rows, ...
    ['ERS: point-optimal test of \citet{ERS1996} with a constant, on the unemployment rate. Cycle: test of ', ...
     'Section \ref{sec:test} of the null that the treated state has a stochastic trend of its own, with the ', ...
     'current value and $q$ lags of the weeks of EUC and EB, controls the never-triggered states that do not ', ...
     'border the treated state, weights estimated by \eqref{eq:dols} or set equal. Level shifts: sequential ', ...
     'sup-Wald tests on the cycle net of the policy terms. Sieve-bootstrap p-values.']);

% ------------------------------------------------------------------ average
rows = {};
for h = [0 6 12 24 36]
    i = h + 1;
    w = (C.cycle_hi90(i) - C.cycle_lo90(i)) / (C.no_controls_hi90(i) - C.no_controls_lo90(i));
    rows{end + 1} = sprintf('%d & %s & %s & %s & %s & %s \\\\', h, f2(C.cycle(i)), f2(C.no_controls(i)), f2(w), f2(C.ctrl_controls(i)), f2(C.ice(i))); %#ok<AGROW>
    rows{end + 1} = ['\footnotesize & ', band(C.cycle_lo90(i), C.cycle_hi90(i)), ' & ', band(C.no_controls_lo90(i), C.no_controls_hi90(i)), ...
        ' & & ', band(C.ctrl_controls_lo90(i), C.ctrl_controls_hi90(i)), ' & ', band(C.ice_lo90(i), C.ice_hi90(i)), ' \\[2pt]']; %#ok<AGROW>
end
wtab(op, 'tab_eb_average.tex', 'Response of the unemployment rate to ten additional weeks of EB, average across states', ...
    'tab:eb-avg', 'cccccc', 'Months & CVAR & VAR without controls & Width ratio & Weighted controls & Impulse causal effect', rows, ...
    sprintf(['Average across the %d states with at least twelve months of EB, October 2002 to December 2019. Cumulated ', ...
     'response in percentage points to an EB innovation of ten weeks, with EUC ordered first. CVAR: VAR in the ', ...
     'weeks of EUC and EB and the change in the cycle, equal weights on the controls. VAR without controls: the ', ...
     'same with the change in the unemployment rate of the treated state. Weighted controls: response in the VAR ', ...
     'with both the treated state and the weighted controls (Proposition \\ref{prop:placebo}). Impulse causal ', ...
     'effect of one week of EB from Theorem \\ref{thm:direct}. %d lags in every state; 90 percent bands from ', ...
     '%d draws of a residual bootstrap that draws the same months for every state.'], N, p, B));

% ------------------------------------------------------------------ bounds
bd = cvar.bounds(avgc, [0 0.25 0.5], [6 12]);
rows = {};
for k = 1:height(bd)
    if isfinite(bd.breakdown_lambda(k)), bl = f2(bd.breakdown_lambda(k)); else, bl = '$>3$'; end
    rows{end + 1} = sprintf('%d & %s & %s & %s & %s & %s \\\\', bd.h(k), f2(bd.treated_response(k)), ...
        band(bd.lam0_lower(k), bd.lam0_upper(k)), band(bd.lam0_25_lower(k), bd.lam0_25_upper(k)), ...
        band(bd.lam0_5_lower(k), bd.lam0_5_upper(k)), bl); %#ok<AGROW>
end
wtab(op, 'tab_eb_bounds.tex', 'Sensitivity to the exogeneity assumption, average across states', 'tab:eb-bounds', 'cccccc', ...
    'Months & VAR with controls added & $\lambda=0$ & $\lambda=0.25$ & $\lambda=0.5$ & $\lambda$ reaching the VAR', rows, ...
    ['Response of the treated state in the VAR with the weighted controls added, and 90 percent bands for the ', ...
     'bounds of Proposition \ref{prop:bounds}, with the bias specific to the treated state at most $\lambda$ ', ...
     'times the bias from the common trends. Last column: smallest $\lambda$ at which the band contains the ', ...
     'response of the treated state.']);

% ------------------------------------------------------------------ states (supplement)
rows = {};
for s = sort(sts)
    z = S4(S4.spec == "rule" & S4.st == s{1} & S4.h == 12, :);
    rows{end + 1} = sprintf('%s & %d/%d & %s %s & %s %s & %s \\\\', s{1}, z.p_cvar(1), z.p_var(1), f2(z.cvar(1)), ...
        band(z.cvar_lo(1), z.cvar_hi(1)), f2(z.var(1)), band(z.var_lo(1), z.var_hi(1)), f2(z.width_ratio(1))); %#ok<AGROW>
end
wtab(op, 'tab_eb_states.tex', 'Response of the unemployment rate to ten additional weeks of EB after twelve months, by state', ...
    'tab:eb-states', 'lcccc', 'State & Lags & CVAR & VAR without controls & Width ratio', rows, ...
    sprintf(['Cumulated response in percentage points, with 90 percent bands from %d residual bootstrap draws. Lags: ', ...
     'CVAR / VAR without controls, chosen by the rule of Section \\ref{sec:eb}. Width ratio: width of the ', ...
     'CVAR band over that of the VAR without controls.'], B));
fprintf('written to %s\n', op);
end

function draw_average(C, H, op)
hh = (0:H)'; c0 = [0.12 0.47 0.71]; c1 = [1 0.5 0.05]; c2 = [0.17 0.63 0.17]; c3 = [0.84 0.15 0.16];
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 13 3.9]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
shade(ax, hh, C.cycle_lo90, C.cycle_hi90, c0, 0.18, 'CVAR, 90%'); shade(ax, hh, C.cycle_lo68, C.cycle_hi68, c0, 0.35, 'CVAR, 68%');
plot(ax, hh, C.cycle, 'Color', c0, 'LineWidth', 1.8, 'DisplayName', 'CVAR');
plot(ax, hh, C.no_controls, '--', 'Color', c3, 'LineWidth', 1.2, 'DisplayName', 'VAR without controls');
plot(ax, hh, C.no_controls_lo90, ':', 'Color', c3, 'LineWidth', 0.7, 'HandleVisibility', 'off');
plot(ax, hh, C.no_controls_hi90, ':', 'Color', c3, 'LineWidth', 0.7, 'DisplayName', 'VAR, 90%');
title(ax, '(a) Unemployment rate', 'FontWeight', 'normal', 'FontSize', 10); ylabel(ax, 'percentage points'); fin(ax, H);
ax = nexttile(tl); hold(ax, 'on');
for k = {'treated', c3, 'treated state'; 'controls', c2, 'weighted controls'; 'difference', c0, 'difference'}'
    plot(ax, hh, C.(['ctrl_', k{1}]), 'Color', k{2}, 'LineWidth', 1.5, 'DisplayName', k{3});
    shade(ax, hh, C.(['ctrl_', k{1}, '_lo90']), C.(['ctrl_', k{1}, '_hi90']), k{2}, 0.12, '');
end
title(ax, '(b) Treated state and weighted controls', 'FontWeight', 'normal', 'FontSize', 10); fin(ax, H);
ax = nexttile(tl); hold(ax, 'on');
plot(ax, hh, C.policy, 'Color', c1, 'LineWidth', 1.5, 'DisplayName', 'weeks of EB (tens)');
shade(ax, hh, C.policy_lo90, C.policy_hi90, c1, 0.15, '');
plot(ax, hh, C.ice, 'Color', c0, 'LineWidth', 1.5, 'DisplayName', 'impulse causal effect');
shade(ax, hh, C.ice_lo90, C.ice_hi90, c0, 0.15, '');
title(ax, '(c) Policy path and impulse causal effect', 'FontWeight', 'normal', 'FontSize', 10); fin(ax, H);
exportgraphics(fig, fullfile(op, 'fig_eb_average.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(op, 'fig_eb_average.png'), 'Resolution', 200);
close(fig);
end

function draw_states(S4, sts, op)
z12 = S4(S4.spec == "rule" & S4.h == 12 & ismember(S4.st, sts), :);
[~, o] = sort(z12.cvar); z12 = z12(o, :);
n = height(z12); y = (0:n - 1)';
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 12 0.22 * n + 1.5]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl, [1 2]); hold(ax, 'on');
errorbar(ax, z12.cvar, y + 0.15, [], [], z12.cvar - z12.cvar_lo, z12.cvar_hi - z12.cvar, 'o', 'MarkerSize', 3, ...
    'Color', [0.12 0.47 0.71], 'CapSize', 0, 'DisplayName', 'CVAR');
errorbar(ax, z12.var, y - 0.15, [], [], z12.var - z12.var_lo, z12.var_hi - z12.var, 's', 'MarkerSize', 3, ...
    'Color', [0.84 0.15 0.16], 'CapSize', 0, 'DisplayName', 'VAR without controls');
xline(ax, 0, 'k', 'HandleVisibility', 'off');
set(ax, 'YTick', y, 'YTickLabel', cellstr(z12.st), 'FontSize', 7); ylim(ax, [-1 n]);
xlabel(ax, 'percentage points, 12 months'); legend(ax, 'Location', 'southeast', 'Box', 'off');
ax = nexttile(tl); hold(ax, 'on');
mk = {6, 'o', [1 0.5 0.05]; 12, 's', [0.12 0.47 0.71]; 24, '^', [0.17 0.63 0.17]};
for i = 1:3
    z = S4(S4.spec == "rule" & S4.h == mk{i, 1}, :);
    [~, j] = ismember(z12.st, z.st);
    plot(ax, z.width_ratio(j), y, mk{i, 2}, 'MarkerSize', 3, 'Color', mk{i, 3}, 'DisplayName', sprintf('%d months', mk{i, 1}));
end
xline(ax, 1, 'k', 'HandleVisibility', 'off');
set(ax, 'YTick', y, 'YTickLabel', []); ylim(ax, [-1 n]);
xlabel(ax, 'band width, CVAR / VAR'); legend(ax, 'Box', 'off');
exportgraphics(fig, fullfile(op, 'fig_eb_states.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(op, 'fig_eb_states.png'), 'Resolution', 200);
close(fig);
end

function shade(ax, x, lo, hi, c, a, nm)
if isempty(nm)
    fill(ax, [x; flipud(x)], [lo; flipud(hi)], c, 'FaceAlpha', a, 'EdgeColor', 'none', 'HandleVisibility', 'off');
else
    fill(ax, [x; flipud(x)], [lo; flipud(hi)], c, 'FaceAlpha', a, 'EdgeColor', 'none', 'DisplayName', nm);
end
end

function fin(ax, H)
yline(ax, 0, 'k', 'LineWidth', 0.7, 'HandleVisibility', 'off'); xlim(ax, [0 H]); xlabel(ax, 'months');
legend(ax, 'FontSize', 7, 'Box', 'off'); box(ax, 'on');
end

function s = f2(x)
if isempty(x) || ~isfinite(x), s = ''; else, s = strrep(sprintf('%.2f', x), '-', '$-$'); end
end

function s = band(lo, hi)
s = ['[', f2(lo), ', ', f2(hi), ']'];
end

function wtab(op, name, caption, label, colspec, header, rows, note)
t = [{'\begin{table}[htbp]', '\centering', ['\caption{', caption, '\label{', label, '}}'], '\small', ...
      ['\begin{tabular}{', colspec, '}'], '\toprule', [header, ' \\'], '\midrule'}, rows, ...
     {'\bottomrule', '\end{tabular}', ['\par\smallskip\parbox{0.95\textwidth}{\footnotesize ', note, '}'], '\end{table}'}];
fid = fopen(fullfile(op, name), 'w'); fprintf(fid, '%s\n', t{:}); fclose(fid);
fprintf('%s written\n', name);
end
