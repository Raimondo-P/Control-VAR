function ak_exhibits(cfg, tables_only)
%AK_EXHIBITS  Figure and LaTeX tables of the Alaska application (Python ak_3_exhibits.py and the
%   tables of ak_6_diagnostic.py). Re-estimates the main specification (pre-2020, six donors,
%   weights in changes, oil first, 12 lags) for the full response paths, with the seeds of
%   ak_main, and reads the rest from results/alaska: 0_screen, 1_responses, 2_bounds, 2_seasonal,
%   6_diagnostic, 6_responses, 6_weights.
%   Outputs (results/paper): fig_alaska_main.pdf/.png/.csv, tab_alaska_weights.tex,
%   tab_alaska_main.tex, tab_alaska_bounds.tex, tab_alaska_adjusted.tex, and for the supplement
%   tab_alaska_samples.tex, tab_alaska_seasonal.tex, tab_alaska_pools.tex.
%   tables_only = true rebuilds the tables from fig_alaska_main.csv without re-estimating.
if nargin < 1 || isempty(cfg), cfg = config(); end
if nargin < 2, tables_only = false; end
K = ak_constants();
src = fullfile(cfg.results, 'alaska');
op = cfg.paper; if ~exist(op, 'dir'), mkdir(op); end
outc = {'epop', 'ur', 'lfpr', 'nonfarm', 'leisure', 'retail'};
LAB = containers.Map(outc, {'Employment-population ratio', 'Unemployment rate', 'Participation rate', ...
    'Payroll employment', 'Leisure and hospitality', 'Retail trade'});
UNIT = containers.Map(outc, {'percentage points', 'percentage points', 'percentage points', 'percent', 'percent', 'percent'});
SHORT = containers.Map(outc, {'Employment rate', 'Unemployment', 'Participation', 'Payroll', 'Leisure', 'Retail'});
rd = @(f) readtable(fullfile(src, f), 'TextType', 'string');
SCR = rd('0_screen.csv'); RS = rd('1_responses.csv'); BD = rd('2_bounds.csv'); SS = rd('2_seasonal.csv');
DG = rd('6_diagnostic.csv'); RS6 = rd('6_responses.csv'); WT6 = rd('6_weights.csv');
H = cfg.H; figcsv = fullfile(op, 'fig_alaska_main.csv');
cc = K.sets.jm_emp; last = datetime(2019, 12, 1);

% ------------------------------------------------------------------ full paths and figure
if ~tables_only
    D = ak_load(cfg, true);
    PT = cell(numel(outc), 1); DR = cell(numel(outc), 1);
    B = cfg.B_app; Bc = min(cfg.B_app, cfg.B_cr); ell = cfg.block;
    parfor i = 1:numel(outc)
        [PT{i}, DR{i}] = main_paths(ak_arrays(D, outc{i}, cc, last), K, B, Bc, ell, H);
    end
    C = table((0:H)', 'VariableNames', {'h'});
    keys = {'cycle', 'ice', 'policy', 'var', 'cr_treated', 'cr_controls'};
    for i = 1:numel(outc)
        for k = keys
            q = cvar.quantile7(DR{i}.(k{1}), [0.05 0.95]);
            C.([outc{i}, '_', k{1}]) = PT{i}.(k{1});
            C.([outc{i}, '_', k{1}, '_lo90']) = q(1, :)'; C.([outc{i}, '_', k{1}, '_hi90']) = q(2, :)';
        end
        q = cvar.quantile7(DR{i}.cycle, [0.16 0.84]);
        C.([outc{i}, '_cycle_lo68']) = q(1, :)'; C.([outc{i}, '_cycle_hi68']) = q(2, :)';
    end
    writetable(C, figcsv);
    draw_figure(C, outc, LAB, UNIT, H, op);
end
C = readtable(figcsv);

% ------------------------------------------------------------------ weights and diagnostics
dg = @(o, w, k) pick(DG, DG.outcome == o & DG.weights == w, k);
scr = SCR(SCR.sample == "pre2020", :);
pdes = @(o, sp, ctl) f2(pick(scr, scr.outcome == o & scr.spec == sp & scr.controls == ctl, 'p'));
rows = {};
for c = cc
    v = cellfun(@(o) f2(pick(WT6, WT6.outcome == o & WT6.state == c{1}, 'weight')), outc, 'UniformOutput', false);
    rows{end + 1} = [c{1}, ' & ', strjoin(v, ' & '), ' \\']; %#ok<AGROW>
end
rows = [rows, {'\midrule', ['Sum & ', jn(@(o) f2(dg(o, "est", 'sum_w')), outc), ' \\'], '\midrule', ...
    ['Common share & ', jn(@(o) f2(dg(o, "est", 'share_pc1')), outc), ' \\'], ...
    ['Reproduced, estimated & ', jn(@(o) f2(dg(o, "est", 'ratio_pc1')), outc), ' \\'], ...
    ['Reproduced, equal & ', jn(@(o) f2(dg(o, "equal", 'ratio_pc1')), outc), ' \\'], '\midrule', ...
    ['$p$, design & ', jn(@(o) pdes(o, "estimated", "jm_emp"), outc), ' \\'], ...
    ['$p$, Alaska & ', jn(@(o) pdes(o, "alaska_alone", "none"), outc), ' \\']}];
wtab(op, 'tab_alaska_weights.tex', 'Weights on the donor states, reproduction of the common movement and tests of the design', ...
    'tab:ak-weights', ['l', repmat('c', 1, numel(outc))], {[' & ', jn(@(o) SHORT(o), outc), ' \\']}, rows, ...
    ['Columns: outcomes of Alaska. Non-negative weights from the regression of the change in the Alaska outcome on the ', ...
     'changes of the six donor states, with a constant and the current value and twelve lags of the dividend and of the ', ...
     'change in the oil price, January 1976 (payroll series January 1990) to December 2019. Common share: share of the ', ...
     'variance of the changes of the 50 other states explained by their first principal component, the main common ', ...
     'movement. Reproduced: ratio of the loading of the weighted donors (estimated or equal weights) to that of Alaska on ', ...
     'that component (Proposition \ref{prop:wchanges}); one means that the weights leave none of it in the cycle. $p$, ', ...
     'design: test of Section \ref{sec:test} with estimated weights (null: a stochastic trend specific to Alaska). $p$, ', ...
     'Alaska: the same test on the Alaska outcome without controls.'], '4pt');

% ------------------------------------------------------------------ main responses
g = @(o, k, h) [C.([o, '_', k])(h + 1), C.([o, '_', k, '_lo90'])(h + 1), C.([o, '_', k, '_hi90'])(h + 1)];
rs = RS(RS.controls == "jm_emp" & RS.sample == "pre2020" & RS.pmode == "p12", :);
rows = {};
for i = 1:numel(outc)
    o = outc{i};
    rows = [rows, head(LAB(o), 6, i == 1)]; %#ok<AGROW>
    for h = [3 12 24]
        c = g(o, 'cycle', h); v = g(o, 'var', h); s = g(o, 'cr_controls', h);
        eq = pick(rs, rs.outcome == o & rs.spec == "equal" & rs.object == "cycle" & rs.h == h, 'estimate');
        rows{end + 1} = sprintf('%d & %s%s & %s & %s & %s%s & %s%s \\\\', h, f2(c(1)), star(c(2), c(3)), f2(eq), ...
            f2(C.([o, '_ice'])(h + 1)), f2(v(1)), star(v(2), v(3)), f2(s(1)), star(s(2), s(3))); %#ok<AGROW>
        rows{end + 1} = sprintf(' & %s & & & %s & %s \\\\', band(c(2), c(3)), band(v(2), v(3)), band(s(2), s(3))); %#ok<AGROW>
    end
end
wtab(op, 'tab_alaska_main.tex', 'Response to a dividend of \$1,000 per person', 'tab:ak-main', 'lccccc', ...
    {'Months & CVAR & Equal & Impulse causal & VAR without & Weighted \\', ' & & weights & effect & controls & controls \\'}, rows, ...
    ['Cumulated response to an innovation of \$1,000 (2024 dollars) per person in the dividend, in percentage ', ...
     'points (rates) or percent (employment), with 90 percent bands below the estimates. CVAR: VAR in the change in ', ...
     'the oil price, the dividend and the change in the cycle, with the weights of Table \ref{tab:ak-weights}; bands ', ...
     'from a residual bootstrap that re-estimates the weights in each draw on a moving-block resample. Equal ', ...
     'weights: the same with weights $1/6$. Impulse causal effect: Theorem \ref{thm:direct}, later dividends held ', ...
     'fixed. VAR without controls: the same with the change in the Alaska outcome. Weighted controls: response in ', ...
     'the VAR with Alaska and the weighted controls (Proposition \ref{prop:placebo}). $^{*}$: the 90 percent band ', ...
     'excludes zero.'], '');

% ------------------------------------------------------------------ bounds
rows = {}; first = true;
for o = {'epop', 'lfpr', 'nonfarm', 'leisure'}
    z0 = BD(BD.outcome == o{1}, :);
    if isempty(z0), continue, end
    rows = [rows, head(LAB(o{1}), 7, first)]; first = false; %#ok<AGROW>
    for h = [12 24]
        z = z0(z0.h == h, :);
        if isempty(z), continue, end
        if isfinite(z.zero_lambda(1)), zl = f2(z.zero_lambda(1)); else, zl = '$>3$'; end
        rows{end + 1} = sprintf('%d & %s & %s & %s & %s & %s & %s \\\\', h, f2(z.difference(1)), f2(z.controls(1)), ...
            band(z.lam0_lower(1), z.lam0_upper(1)), band(z.lam0_5_lower(1), z.lam0_5_upper(1)), ...
            band(z.lam1_lower(1), z.lam1_upper(1)), zl); %#ok<AGROW>
    end
end
wtab(op, 'tab_alaska_bounds.tex', 'Sensitivity to a bias specific to Alaska', 'tab:ak-bounds', 'lcccccc', ...
    {'Months & CVAR & Weighted & $\lambda=0$ & $\lambda=0.5$ & $\lambda=1$ & $\lambda$ reaching \\', ' & & controls & & & & zero \\'}, rows, ...
    ['From the VAR with Alaska and the weighted controls (Proposition \ref{prop:placebo}), 90 percent bands for ', ...
     'the bounds of Proposition \ref{prop:bounds}, with the bias specific to Alaska at most $\lambda$ times the ', ...
     'bias from the common trends, measured by the response of the weighted controls. Weights held at their ', ...
     'estimates. Last column: smallest $\lambda$ at which the band contains zero.'], '');

% ------------------------------------------------------------------ adjusted responses
g6 = @(o, s, h) RS6(RS6.outcome == o & RS6.spec == s & RS6.h == h, :);
rows = {};
for i = 1:numel(outc)
    o = outc{i};
    rows = [rows, head(LAB(o), 5, i == 1)]; %#ok<AGROW>
    for h = [12 24]
        zs = arrayfun(@(s) g6(o, s, h), ["six_est", "six_est_adjusted", "six_equal", "six_equal_adjusted"], 'UniformOutput', false);
        rows{end + 1} = [num2str(h), ' & ', strjoin(cellfun(@estcell, zs, 'UniformOutput', false), ' & '), ' \\']; %#ok<AGROW>
        rows{end + 1} = [' & ', strjoin(cellfun(@bandcell, zs, 'UniformOutput', false), ' & '), ' \\']; %#ok<AGROW>
    end
end
wtab(op, 'tab_alaska_adjusted.tex', 'Responses adjusted for the common movement left in the cycle', 'tab:ak-adjusted', 'lcccc', ...
    {' & \multicolumn{2}{c}{Estimated weights} & \multicolumn{2}{c}{Equal weights} \\', '\cmidrule(lr){2-3}\cmidrule(lr){4-5}', ...
     'Months & CVAR & Adjusted & CVAR & Adjusted \\'}, rows, ...
    ['Response to a dividend of \$1,000 per person with the six donors, 90 percent bands below the estimates. CVAR: as ', ...
     'in Table \ref{tab:ak-main}. Adjusted: response of Alaska relative to the weighted donors in the VAR with both, ', ...
     'minus $(1-r)/r$ times the response of the weighted donors, with $r$ the reproduced share in Table ', ...
     '\ref{tab:ak-weights} (Proposition \ref{prop:wchanges}), computed in each bootstrap draw with $r$ and the weights ', ...
     'held at their estimates. $^{*}$: the 90 percent band excludes zero.'], '');

% ------------------------------------------------------------------ samples (supplement)
rows = {};
for i = 1:numel(outc)
    o = outc{i};
    rows = [rows, head(LAB(o), 4, i == 1)]; %#ok<AGROW>
    for h = [12 24]
        est = {}; bnd = {};
        for smp = ["pre2020", "pre2016", "pre2008"]
            z = RS(RS.outcome == o & RS.sample == smp & RS.controls == "jm_emp" & RS.pmode == "p12" & ...
                RS.spec == "est_full" & RS.object == "cycle" & RS.h == h, :);
            est{end + 1} = estcell(z); bnd{end + 1} = bandcell(z); %#ok<AGROW>
        end
        rows{end + 1} = [num2str(h), ' & ', strjoin(est, ' & '), ' \\']; %#ok<AGROW>
        rows{end + 1} = [' & ', strjoin(bnd, ' & '), ' \\']; %#ok<AGROW>
    end
end
wtab(op, 'tab_alaska_samples.tex', 'Alaska: samples ending before the legislated dividends and before 2008', 'tab:ak-samples', 'lccc', ...
    {'Months & To 2019 & To 2015 & To 2007 \\'}, rows, ...
    ['CVAR of Table \ref{tab:ak-main} estimated on samples ending in December 2019 (main), December 2015 (before ', ...
     'the legislature set the amounts) and December 2007 (before the 2008 dividend, the energy rebate and the ', ...
     'recession), with 90 percent bands below the estimates. Rates from January 1976, payroll series from January ', ...
     '1990. Weights re-estimated in each sample. $^{*}$: the 90 percent band excludes zero.'], '');

% ------------------------------------------------------------------ seasonality (supplement)
SEAS = {'SA', 'Adjusted'; 'SA + months', 'Adjusted, month effects'; 'NSA + months', 'Unadjusted, month effects'; ...
        'NSA + months x trend', 'Unadjusted, month effects $\times$ trend'; 'NSA + months x decade', 'Unadjusted, month effects by decade'};
ss = SS(SS.spec == "cvar", :);
rows = {};
for i = 1:numel(outc)
    o = outc{i};
    rows = [rows, head(LAB(o), 6, i == 1)]; %#ok<AGROW>
    for v = 1:size(SEAS, 1)
        gg = ss(ss.outcome == o & ss.version == SEAS{v, 1}, :);
        if isempty(gg), continue, end
        cells = arrayfun(@(h) estcell(gg(gg.h == h, :)), [1 3 12 24], 'UniformOutput', false);
        rows{end + 1} = sprintf('%s & %s & %s \\\\', SEAS{v, 2}, f2(gg.sum_weights(1)), strjoin(cells, ' & ')); %#ok<AGROW>
    end
end
wtab(op, 'tab_alaska_seasonal.tex', 'Alaska: treatment of seasonality', 'tab:ak-seasonal', 'lccccc', ...
    {'Series & Sum of & \multicolumn{4}{c}{Months} \\', '\cmidrule(lr){3-6}', ' & weights & 1 & 3 & 12 & 24 \\'}, rows, ...
    ['CVAR of Table \ref{tab:ak-main}, January 1976 (payroll series January 1990) to December 2019, with the ', ...
     'seasonally adjusted series (main), the adjusted series with month effects, and the unadjusted series with ', ...
     'month effects, with month effects interacted with a linear trend, and with month effects by decade, in the ', ...
     'weights regression and in the VAR. Weights re-estimated in each version and held at their estimates in the ', ...
     'residual bootstrap. $^{*}$: the 90 percent band excludes zero.'], '');

% ------------------------------------------------------------------ larger pools (supplement)
rows = {};
for i = 1:numel(outc)
    o = outc{i};
    rows = [rows, head(LAB(o), 4, i == 1)]; %#ok<AGROW>
    rows{end + 1} = ['Reproduced & ', strjoin(cellfun(@(w) f2(dg(o, w, 'ratio_pc1')), {"est", "west_est", "all_est"}, ...
        'UniformOutput', false), ' & '), ' \\']; %#ok<AGROW>
    for h = [12 24]
        zs = arrayfun(@(s) g6(o, s, h), ["six_est", "west_est", "all_est"], 'UniformOutput', false);
        rows{end + 1} = [num2str(h), ' months & ', strjoin(cellfun(@estcell, zs, 'UniformOutput', false), ' & '), ' \\']; %#ok<AGROW>
        rows{end + 1} = [' & ', strjoin(cellfun(@bandcell, zs, 'UniformOutput', false), ' & '), ' \\']; %#ok<AGROW>
    end
end
wtab(op, 'tab_alaska_pools.tex', 'Alaska: larger donor pools', 'tab:ak-pools', 'lccc', ...
    {' & Six donors & West & All states \\'}, rows, ...
    ['CVAR of Table \ref{tab:ak-main} with weights estimated in changes from the six donors of the main specification, ', ...
     'from the 12 other states of the Census West region, and from all 50 other states and the District of Columbia, ', ...
     'with 90 percent bands below the estimates. Reproduced: share of the main common movement of Alaska reproduced ', ...
     'by the weighted donors, as in Table \ref{tab:ak-weights}. $^{*}$: the 90 percent band excludes zero.'], '');
fprintf('written to %s\n', op);
end

% ====================================================================== helpers
function [PT, DR] = main_paths(A, K, B, Bc, ell, H)
% The main specification with the seeds of ak_main: full response paths and their draws.
[dy, X, F] = ak_design_changes(A, K.q);
w = ak_wnnls(dy, X, F);
rows = ak_mbb(cvar.rng_stream(22), numel(dy), ell, B);
r = ak_fit_changes(A, w, 12);
nU = size(r.U, 1);
idx = randi(cvar.rng_stream(21), nU, B, nU);
DR = struct('cycle', zeros(B, H + 1), 'ice', zeros(B, H + 1), 'policy', zeros(B, H + 1), 'var', zeros(B, H + 1));
for b = 1:B
    rb = rows(b, :);
    d = cvar.draw_conditional(ak_fit_changes(A, ak_wnnls(dy(rb), X(rb, :), F(rb, :)), 12), idx(b, :)', H, false);
    DR.cycle(b, :) = d.cycle(:, 1)'; DR.ice(b, :) = d.ice(:, 1)'; DR.policy(b, :) = d.policy';
    DR.var(b, :) = cvar.nc_draw(r, idx(b, :)', H)';
end
pt = cvar.responses(r, H);
PT = struct('cycle', pt.cycle(:, 1), 'ice', pt.ice(:, 1), 'policy', pt.policy, 'var', pt.no_controls(:, 1));
cr = cvar.controls_response(r, H, Bc, 13);
PT.cr_treated = cr.point.treated(:, 1); PT.cr_controls = cr.point.controls(:, 1);
DR.cr_treated = cr.draws.treated(:, :, 1); DR.cr_controls = cr.draws.controls(:, :, 1);
end

function draw_figure(C, outc, LAB, UNIT, H, op)
hh = (0:H)';
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 13 7.2]);
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
blue = [0.12 0.47 0.71]; red = [0.84 0.15 0.16]; green = [0.17 0.63 0.17];
for i = 1:numel(outc)
    o = outc{i};
    ax = nexttile(tl); hold(ax, 'on');
    fill(ax, [hh; flipud(hh)], [C.([o, '_cycle_lo90']); flipud(C.([o, '_cycle_hi90']))], blue, 'FaceAlpha', 0.18, 'EdgeColor', 'none', 'DisplayName', 'CVAR, 90%');
    fill(ax, [hh; flipud(hh)], [C.([o, '_cycle_lo68']); flipud(C.([o, '_cycle_hi68']))], blue, 'FaceAlpha', 0.35, 'EdgeColor', 'none', 'DisplayName', 'CVAR, 68%');
    plot(ax, hh, C.([o, '_cycle']), 'Color', blue, 'LineWidth', 1.8, 'DisplayName', 'CVAR');
    plot(ax, hh, C.([o, '_var']), '--', 'Color', red, 'LineWidth', 1.2, 'DisplayName', 'VAR without controls');
    plot(ax, hh, C.([o, '_var_lo90']), ':', 'Color', red, 'LineWidth', 0.7, 'HandleVisibility', 'off');
    plot(ax, hh, C.([o, '_var_hi90']), ':', 'Color', red, 'LineWidth', 0.7, 'DisplayName', 'VAR, 90%');
    plot(ax, hh, C.([o, '_cr_controls']), '-.', 'Color', green, 'LineWidth', 1.2, 'DisplayName', 'Weighted controls');
    yline(ax, 0, 'k', 'LineWidth', 0.6, 'HandleVisibility', 'off');
    title(ax, LAB(o), 'FontSize', 10, 'FontWeight', 'normal');
    ylabel(ax, UNIT(o), 'FontSize', 9); xlabel(ax, 'months', 'FontSize', 9);
    xlim(ax, [0 H]); box(ax, 'on');
    if i == 1, legend(ax, 'Location', 'southwest', 'FontSize', 7); end
end
exportgraphics(fig, fullfile(op, 'fig_alaska_main.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(op, 'fig_alaska_main.png'), 'Resolution', 200);
close(fig);
fprintf('fig_alaska_main written\n');
end

function v = pick(T, sel, col)
z = T(sel, :);
if isempty(z), v = NaN; else, v = z.(col)(1); end
end

function s = jn(f, list)
s = strjoin(cellfun(f, list, 'UniformOutput', false), ' & ');
end

function s = f2(x)
if isempty(x) || ~isfinite(x), s = ''; else, s = strrep(sprintf('%.2f', x), '-', '$-$'); end
end

function s = band(lo, hi)
s = ['[', f2(lo), ', ', f2(hi), ']'];
end

function s = star(lo, hi)
if lo <= 0 && 0 <= hi, s = ''; else, s = '$^{*}$'; end
end

function s = estcell(z)
if isempty(z) || height(z) == 0, s = ''; else, s = [f2(z.estimate(1)), star(z.lo90(1), z.hi90(1))]; end
end

function s = bandcell(z)
if isempty(z) || height(z) == 0, s = ''; else, s = band(z.lo90(1), z.hi90(1)); end
end

function r = head(lab, ncol, first)
if first, r = {}; else, r = {'\addlinespace'}; end
r{end + 1} = sprintf('\\multicolumn{%d}{l}{%s} \\\\', ncol, lab);
end

function wtab(op, name, caption, label, colspec, header, rows, note, sep)
t = {'\begin{table}[htbp]', '\centering', ['\caption{', caption, '\label{', label, '}}'], '\footnotesize'};
if ~isempty(sep), t{end + 1} = ['\setlength{\tabcolsep}{', sep, '}']; end
t = [t, {['\begin{tabular}{', colspec, '}'], '\toprule'}, header, {'\midrule'}, rows, ...
     {'\bottomrule', '\end{tabular}', ['\par\smallskip\parbox{0.95\textwidth}{\footnotesize ', note, '}'], '\end{table}'}];
fid = fopen(fullfile(op, name), 'w');
fprintf(fid, '%s\n', t{:});
fclose(fid);
fprintf('%s written\n', name);
end
