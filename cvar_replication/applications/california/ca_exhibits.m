function ca_exhibits(cfg)
%CA_EXHIBITS  Figure and LaTeX tables of the California application (Python ca_4_exhibits.py).
%   Requires ca_robustness. Re-estimates the main specification (spliced cost, equal weights, rule
%   lags from B_main) for the full paths.
%   Outputs (results/paper): fig_california_main.pdf/.png/.tex, fig_california_<controls>.csv,
%   tab_california_main.tex, tab_california_robust.tex, tab_california_appendix.tex
%   Marks: dagger, the 68 percent band excludes zero; *, the 90 percent band excludes zero.
if nargin < 1, cfg = config(); end
K = ca_constants(); H = K.H; B = cfg.B_app;
src = fullfile(cfg.results, 'california'); op = cfg.paper; if ~exist(op, 'dir'), mkdir(op); end
rd = @(f) readtable(fullfile(src, f), 'TextType', 'string');
Bm = rd('B_main.csv'); Cl = rd('C_lags.csv'); Ds = rd('D_samples.csv'); Ec = rd('E_controls.csv'); Fl = rd('F_lp.csv');
[D, first] = ca_load(cfg);
sets = fieldnames(K.sets)';
P = struct();
for cs = sets
    S = ca_series(D, K.sets.(cs{1}), 'spliced', first);
    z = Bm(Bm.spec == string([cs{1}, '_spliced']) & Bm.h == 0, :);
    pc = z.p_cvar(1); pv = z.p_var(1);
    w = ones(size(S.C, 2), 1) / size(S.C, 2);
    bc = cvar.bootstrap(cvar.fit(S.y, S.C, S.dose, struct('p', pc, 'weights', w, 'difference', true)), B, H, struct('seed', 1, 'no_controls', false));
    bv = cvar.bootstrap(cvar.fit(S.y, S.C, S.dose, struct('p', pv, 'weights', w, 'difference', true)), B, H, struct('seed', 2, 'no_controls', true));
    [l9, h9] = cvar.bands(bc, 'cycle', 0.10); [l6, h6] = cvar.bands(bc, 'cycle', 0.32); [vl, vh] = cvar.bands(bv, 'no_controls', 0.10);
    T = table((0:H)', bc.point.cycle(:, 1), l9(:, 1), h9(:, 1), l6(:, 1), h6(:, 1), bv.point.no_controls(:, 1), vl(:, 1), vh(:, 1), ...
        'VariableNames', {'h', 'cvar', 'cvar_lo90', 'cvar_hi90', 'cvar_lo68', 'cvar_hi68', 'var', 'var_lo90', 'var_hi90'});
    writetable(T, fullfile(op, ['fig_california_', cs{1}, '.csv']));
    P.(cs{1}) = T;
    fprintf('%s: lags %d/%d, impact %.2f, 12m %.2f\n', K.setlab.(cs{1}), pc, pv, T.cvar(1), T.cvar(min(12, H) + 1));
end
draw(P, sets, K, H, op);
ftex = {'\begin{figure}[htbp]', '\centering', '\includegraphics[width=\textwidth]{fig_california_main.pdf}', ...
    '\caption{Pass-through of the LCFS cost to the California retail gasoline price}', '\label{fig:california_main}', ...
    '\begin{minipage}{0.95\textwidth}\footnotesize', ...
    ['Note: cumulated response of the California retail price, in cents per gallon, to an increase of 1 cent per ', ...
     'gallon in the LCFS cost. Solid line and shaded areas: CVAR in the differenced branch with equal weights on ', ...
     'the controls named in each panel, with 68 and 90 percent residual bootstrap bands. Dashed and dotted lines: ', ...
     'VAR without controls in the LCFS cost and the change in the California price, with its 90 percent band. ', ...
     'Dash-dotted line: full pass-through.'], '\end{minipage}', '\end{figure}'};
wfile(op, 'fig_california_main.tex', ftex);
NOTE = ['Note: cumulated response of the California retail price, in cents per gallon, to an increase of 1 cent ', ...
    'per gallon in the LCFS cost, from the CVAR in the differenced branch with equal weights on the controls. ', ...
    'Policy: the cost per gallon implied by the CARB formula before May 2024 and the cost reported by refiners ', ...
    'to the California Energy Commission from then on, in monthly changes, unless stated otherwise. Lags: CVAR / ', ...
    'VAR without controls. 90 percent residual bootstrap bands in brackets; $^{\dagger}$ the 68 percent band ', ...
    'excludes zero; $^{*}$ the 90 percent band excludes zero. Width ratio: width of the CVAR band over that of ', ...
    'the VAR without controls, at 12 months.'];
% main table
rows = {};
for cs = sets
    rows = [rows, cells_short(Bm, [cs{1}, '_spliced'], K.setlab.(cs{1}), lagtxt(Bm, [cs{1}, '_spliced']))]; %#ok<AGROW>
end
t = [{'\begin{table}[htbp]', '\centering', '\caption{Pass-through of the LCFS cost to the California retail gasoline price\label{tab:california_main}}', ...
      '\small', '\begin{tabular}{lcccc}', '\toprule', 'Controls & Lags & Impact & 1 month & Width ratio \\', '\midrule'}, rows, ...
     {'\bottomrule', '\end{tabular}', '\begin{minipage}{0.95\textwidth}\footnotesize', ...
      [strrep(NOTE, 'at 12 months', 'at one month'), ' The full paths are in Figure \ref{fig:california_main}, and robustness ', ...
       'checks in Tables \ref{tab:california_robust} and \ref{tab:california_appendix} of Supplementary Appendix \ref{sec:app-eb}.'], ...
      '\end{minipage}', '\end{table}'}];
wfile(op, 'tab_california_main.tex', t);
% robustness table
ml = {};
for cs = sets, ml = [ml, cells(Bm, [cs{1}, '_spliced'], K.setlab.(cs{1}), lagtxt(Bm, [cs{1}, '_spliced']), '')]; end %#ok<AGROW>
body = block('Main specification', ml);
smp = {'2017_end', '2017 onward'; '2020_end', '2020 onward'; '2014_2024', '2014 to 2024'};
sl = {};
for cs = {'gulf', 'padd1_4'}
    for k = 1:size(smp, 1)
        sp = sprintf('%s_%s_rule', cs{1}, smp{k, 1});
        sl = [sl, cells(Ds, sp, sprintf('%s, %s', K.setlab.(cs{1}), smp{k, 2}), lagtxt(Ds, sp), '')]; %#ok<AGROW>
    end
    sl = [sl, cells(Ds, [cs{1}, '_dummies2020_22_p2'], [K.setlab.(cs{1}), ', 2020 to 2022 dummied out'], '2/2', '')]; %#ok<AGROW>
end
body = [body, block('Samples', sl)];
el = {};
for k = 1:size(K.region, 1)
    sp = ['without_', K.region{k, 1}];
    el = [el, cells(Ec, sp, ['PADDs 1 to 4 without ', K.region{k, 2}], lagtxt(Ec, sp), '')]; %#ok<AGROW>
end
body = [body, block('Leaving one PADD out', el)];
table6(op, 'tab_california_robust.tex', 'Pass-through of the LCFS cost: samples and controls', 'tab:california_robust', body, NOTE, '');
% appendix table
ll = {};
for cs = {'gulf', 'padd1_4'}
    for p = [1 2 4 6 8 12]
        ll = [ll, cells(Cl, sprintf('%s_p%d', cs{1}, p), sprintf('%s, lag order %d', K.setlab.(cs{1}), p), sprintf('%d/%d', p, p), '')]; %#ok<AGROW>
    end
end
body = block('Lag order (both models)', ll);
MEAS = {'spliced_jan', 'spliced, January dummy'; 'tv', 'CARB formula only'; 'const', 'constant factor'};
mlines = {};
for cs = {'gulf', 'padd1_4'}
    for k = 1:size(MEAS, 1)
        sp = sprintf('%s_%s', cs{1}, MEAS{k, 1});
        mlines = [mlines, cells(Bm, sp, sprintf('%s, %s', K.setlab.(cs{1}), MEAS{k, 2}), lagtxt(Bm, sp), '')]; %#ok<AGROW>
    end
end
body = [body, block('Policy measure', mlines)];
al = {};
for cs = {'gulf', 'padd1_4'}
    for lt = {'rule', 'p2'}
        sp = sprintf('%s_2014_2019_%s', cs{1}, lt{1});
        z = Ds(Ds.spec == sp, :);
        if ~isempty(z)
            if z.branch(1) == "levels", nt = '$^{a}$'; else, nt = ''; end
            al = [al, cells(Ds, sp, [K.setlab.(cs{1}), ', 2014 to 2019'], lagtxt(Ds, sp), nt)]; %#ok<AGROW>
        end
    end
end
body = [body, block('Sample 2014 to 2019', al)];
LPK = {'linear', 'coef1', 'linear'; 'increase_decrease', 'coef1', 'increases'; 'increase_decrease', 'coef2', 'decreases'; ...
       'high_low_credit', 'coef1', 'high credit price'; 'high_low_credit', 'coef2', 'low credit price'};
lp = {};
for cs = {'gulf', 'padd1_4'}
    for k = 1:size(LPK, 1)
        z = Fl(Fl.controls == cs{1} & Fl.outcome == "cvar_cycle" & Fl.p == 2 & Fl.kind == LPK{k, 1} & Fl.term == LPK{k, 2}, :);
        if isempty(z), continue, end
        top = {sprintf('%s, local projection, %s', K.setlab.(cs{1}), LPK{k, 3}), '2'}; bot = {'', ''};
        for h = [0 1 12]
            r = z(z.h == h, :);
            if r.lo(1) > 0 || r.hi(1) < 0, st = '$^{*}$'; else, st = ''; end
            top{end + 1} = [f2(r.coef(1)), st]; bot{end + 1} = sprintf('[%s, %s]', f2(r.lo(1)), f2(r.hi(1))); %#ok<AGROW>
        end
        top{end + 1} = ''; bot{end + 1} = ''; %#ok<AGROW>
        lp = [lp, {[strjoin(top, ' & '), ' \\'], ['\footnotesize ', strjoin(bot, ' & '), ' \\[2pt]']}]; %#ok<AGROW>
    end
end
body = [body, block('Local projections on the cycle', lp)];
extra = [' In 2014 to 2019 the cost per gallon implied by the CARB formula is small, so the policy varies little. ', ...
    '$^{a}$ The design test rejects in this sample and the CVAR is in levels with estimated weights, where the ', ...
    'response of the stationary cycle to a permanent change in the cost returns to zero by construction. Local ', ...
    'projections: regression of the change in the cycle between $t-1$ and $t+h$ on the change in the cost (split ', ...
    'into increases and decreases, or interacted with a credit price above or below the sample median at $t-1$), ', ...
    'two lags of the changes in the cycle and in the cost, and a constant; 90 percent Newey-West bands.'];
table6(op, 'tab_california_appendix.tex', ...
    'Pass-through of the LCFS cost: lag orders, policy measures, the 2014 to 2019 sample and local projections', ...
    'tab:california_appendix', body, NOTE, extra);
fprintf('written to %s\n', op);
end

% ====================================================================== helpers
function draw(P, sets, K, H, op)
hh = (0:H)'; c0 = [0.12 0.47 0.71]; c3 = [0.84 0.15 0.16];
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 13 3.9]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
axs = gobjects(1, numel(sets));
for k = 1:numel(sets)
    T = P.(sets{k}); ax = nexttile(tl); axs(k) = ax; hold(ax, 'on');
    fill(ax, [hh; flipud(hh)], [T.cvar_lo90; flipud(T.cvar_hi90)], c0, 'FaceAlpha', 0.18, 'EdgeColor', 'none', 'DisplayName', 'CVAR, 90%');
    fill(ax, [hh; flipud(hh)], [T.cvar_lo68; flipud(T.cvar_hi68)], c0, 'FaceAlpha', 0.35, 'EdgeColor', 'none', 'DisplayName', 'CVAR, 68%');
    plot(ax, hh, T.cvar, 'Color', c0, 'LineWidth', 1.8, 'DisplayName', 'CVAR');
    plot(ax, hh, T.var, '--', 'Color', c3, 'LineWidth', 1.2, 'DisplayName', 'VAR without controls');
    plot(ax, hh, T.var_lo90, ':', 'Color', c3, 'LineWidth', 0.7, 'HandleVisibility', 'off');
    plot(ax, hh, T.var_hi90, ':', 'Color', c3, 'LineWidth', 0.7, 'DisplayName', 'VAR, 90%');
    yline(ax, 0, 'k', 'LineWidth', 0.7, 'HandleVisibility', 'off');
    yline(ax, 1, '-.', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'DisplayName', 'full pass-through');
    title(ax, K.setlab.(sets{k}), 'FontWeight', 'normal', 'FontSize', 10); xlabel(ax, 'months'); xlim(ax, [0 H]); box(ax, 'on');
    if k == 1, ylabel(ax, 'cents per gallon'); legend(ax, 'Location', 'northwest', 'FontSize', 7, 'Box', 'off'); end
end
linkaxes(axs, 'y');
exportgraphics(fig, fullfile(op, 'fig_california_main.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(op, 'fig_california_main.png'), 'Resolution', 200);
close(fig);
end

function s = mark(r)
if r.cvar_lo > 0 || r.cvar_hi < 0, s = '$^{*}$';
elseif r.cvar_lo68 > 0 || r.cvar_hi68 < 0, s = '$^{\dagger}$';
else, s = ''; end
end

function L = cells(df, spec, label, lags, note)
z = df(df.spec == spec, :);
if isempty(z), L = {}; return, end
top = {[label, note], lags}; bot = {'', ''};
for h = [0 1 12]
    r = z(z.h == h, :); r = r(1, :);
    top{end + 1} = [f2(r.cvar), mark(r)]; bot{end + 1} = sprintf('[%s, %s]', f2(r.cvar_lo), f2(r.cvar_hi)); %#ok<AGROW>
end
r12 = z(z.h == 12, :);
top{end + 1} = f2(r12.width_ratio_boot(1)); bot{end + 1} = '';
L = {[strjoin(top, ' & '), ' \\'], ['\footnotesize ', strjoin(bot, ' & '), ' \\[2pt]']};
end

function L = cells_short(df, spec, label, lags)
z = df(df.spec == spec, :);
if isempty(z), L = {}; return, end
r0 = z(z.h == 0, :); r0 = r0(1, :); r1 = z(z.h == 1, :); r1 = r1(1, :);
L = {[strjoin({label, lags, [f2(r0.cvar), mark(r0)], [f2(r1.cvar), mark(r1)], f2(r1.width_ratio_boot)}, ' & '), ' \\'], ...
     ['\footnotesize ', strjoin({'', '', sprintf('[%s, %s]', f2(r0.cvar_lo), f2(r0.cvar_hi)), ...
      sprintf('[%s, %s]', f2(r1.cvar_lo), f2(r1.cvar_hi)), ''}, ' & '), ' \\[2pt]']};
end

function s = lagtxt(df, spec)
z = df(df.spec == spec, :);
if isempty(z), s = ''; else, s = sprintf('%d/%d', z.p_cvar(1), z.p_var(1)); end
end

function L = block(title, lines)
if isempty(lines), L = {}; else, L = [{['\multicolumn{6}{l}{', title, '} \\'], '\midrule'}, lines, {'\midrule'}]; end
end

function table6(op, name, caption, label, body, NOTE, extra)
if ~isempty(body) && strcmp(body{end}, '\midrule'), body = body(1:end - 1); end
t = [{'\begin{table}[htbp]', '\centering', ['\caption{', caption, '}'], ['\label{', label, '}'], '\small', ...
      '\begin{tabular}{lccccc}', '\toprule', ' & Lags & Impact & 1 month & 12 months & Width ratio \\', '\midrule'}, body, ...
     {'\bottomrule', '\end{tabular}', '\begin{minipage}{0.95\textwidth}\footnotesize', [NOTE, extra], '\end{minipage}', '\end{table}'}];
wfile(op, name, t);
end

function wfile(op, name, lines)
fid = fopen(fullfile(op, name), 'w'); fprintf(fid, '%s\n', lines{:}); fclose(fid);
fprintf('%s written\n', name);
end

function s = f2(x)
if isempty(x) || ~isfinite(x), s = ''; else, s = sprintf('%.2f', x); end
end
