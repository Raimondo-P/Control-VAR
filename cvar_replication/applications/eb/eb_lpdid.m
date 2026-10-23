function eb_lpdid(cfg)
%EB_LPDID  Local-projection difference-in-differences (Dube, Girardi, Jorda and Taylor 2023) on the
%   EB panel, beside the CVAR (Python eb_7_lpdid.py). Event: EB switches on (weeks positive in month t,
%   zero in each of the twelve previous months). For h = 0, 6, 12, 24, 36:
%     y_{i,t+h} - y_{i,t-1} = beta_h event_it + delta_t + sum_{l=1..4} g_l dy_{i,t-l} + sum_{l=0..4} k_l dEUC_{i,t-l} + e,
%   on the event observations and the controls ('never': never-triggered states; 'clean': state-months
%   with no EB in the twelve months before t and from t to t+h), month fixed effects, standard
%   errors clustered by state (CRV1), 90 percent bands. Requires eb_exhibits (fig_eb_average.csv)
%   for the CVAR columns. Outputs: results/eb/7_lpdid.csv, results/paper/tab_eb_lpdid.tex
if nargin < 1, cfg = config(); end
E = eb_load(cfg); out = eb_out(cfg); op = cfg.paper;
states = [E.elig, E.ctrl];
Y = eb_col(E, E.UR, states); W = eb_col(E, E.EB, states); Ec = eb_col(E, E.EUC, states);
[T, S] = size(Y);
back = nan(T, S);
for t = 13:T, back(t, :) = max(W(t - 12:t - 1, :), [], 1); end
event = W > 0 & back == 0;
never = repmat(ismember(states, E.ctrl), T, 1);
dY = [nan(1, S); diff(Y)]; dE = [nan(1, S); diff(Ec)];
lag = @(M, l) [nan(l, S); M(1:end - l, :)];
sz = mean(W(event));
rows = struct([]);
for h = [0 6 12 24 36]
    lhs = nan(T, S); lhs(2:T - h, :) = Y(2 + h:T, :) - Y(1:T - h - 1, :);
    fwd = nan(T, S);
    for t = 1:T - h, fwd(t, :) = max(W(t:t + h, :), [], 1); end
    clean = back == 0 & fwd == 0;
    X = {lhs, double(event), double(never), double(clean)};
    for l = 1:4, X{end + 1} = lag(dY, l); end %#ok<AGROW>
    for l = 0:4, X{end + 1} = lag(dE, l); end %#ok<AGROW>
    V = cell2mat(cellfun(@(m) m(:), X, 'UniformOutput', false));
    tt = repmat((1:T)', S, 1); ss = kron((1:S)', ones(T, 1));
    ok = all(~isnan(V), 2);
    V = V(ok, :); tt = tt(ok); ss = ss(ok);
    for ctrl = {'never', 'clean'}
        if strcmp(ctrl{1}, 'never'), cv = V(:, 3); else, cv = V(:, 4); end
        sel = V(:, 2) == 1 | (cv == 1 & V(:, 2) == 0);
        y = V(sel, 1); ev = V(sel, 2); Z = V(sel, 5:end); td = tt(sel); sd = ss(sel);
        ud = unique(td);
        Dm = double(td == ud(2:end)');                           % month dummies, first month dropped
        Xr = [ones(numel(y), 1), ev, Z, Dm];
        [b, se] = ols_cluster(y, Xr, sd);
        rr = struct('h', h, 'controls', string(ctrl{1}), 'beta', b(2), 'lo', b(2) - 1.645 * se(2), 'hi', b(2) + 1.645 * se(2), ...
            'beta_per10', b(2) / sz, 'lo_per10', (b(2) - 1.645 * se(2)) / sz, 'hi_per10', (b(2) + 1.645 * se(2)) / sz, ...
            'events', sum(ev), 'control_obs', sum(ev == 0), 'control_states', numel(unique(sd(ev == 0))), ...
            'avg_switch_on_tens_of_weeks', sz);
        if isempty(rows), rows = rr; else, rows(end + 1) = rr; end %#ok<AGROW>
    end
end
R = struct2table(rows); writetable(R, fullfile(out, '7_lpdid.csv')); disp(R);
% table with the CVAR columns of the average (fig_eb_average.csv from eb_exhibits)
fa = fullfile(op, 'fig_eb_average.csv');
if exist(fa, 'file'), F = readtable(fa); else, F = []; fprintf('CVAR columns left empty: run eb_exhibits first\n'); end
lines = {};
for h = [0 6 12 24 36]
    c = {num2str(h)};
    for ctrl = ["never", "clean"]
        z = R(R.h == h & R.controls == ctrl, :);
        c{end + 1} = cell3(z.beta_per10(1), z.lo_per10(1), z.hi_per10(1)); %#ok<AGROW>
    end
    for obj = {'cycle', 'no_controls'}
        if isempty(F), c{end + 1} = ''; else %#ok<AGROW>
            i = find(F.h == h, 1); c{end + 1} = cell3(F.(obj{1})(i), F.([obj{1}, '_lo90'])(i), F.([obj{1}, '_hi90'])(i)); %#ok<AGROW>
        end
    end
    lines{end + 1} = [strjoin(c, ' & '), ' \\']; %#ok<AGROW>
end
ev0 = R.events(R.h == 0 & R.controls == "never");
note = sprintf(['Change in the unemployment rate in percentage points. LP difference-in-differences \\citep{dube2023local}: ', ...
    'response to EB switching on in a state (%d events with no EB in the previous twelve months), with month ', ...
    'fixed effects, four lags of the change in the unemployment rate and the current value and four lags of the ', ...
    'change in the weeks of EUC; controls are the never-triggered states, or every state-month without EB in the ', ...
    'twelve months before the event and up to the horizon. The coefficients are divided by the average size of a ', ...
    'switch-on, %.2f tens of weeks, to express them per ten weeks; 90 percent bands from standard errors ', ...
    'clustered by state. CVAR and VAR without controls: average response to an EB innovation of ten weeks, from ', ...
    'Table \\ref{tab:eb-avg}.'], ev0(1), sz);
t = [{'\begin{table}[htbp]', '\centering', ...
      '\caption{Extended benefits: local-projection difference-in-differences and the CVAR\label{tab:eb-lpdid}}', ...
      '\small', '\begin{tabular}{ccccc}', '\toprule', ...
      ' & \multicolumn{2}{c}{LP difference-in-differences} & \multicolumn{2}{c}{VAR responses} \\', ...
      '\cmidrule(lr){2-3}\cmidrule(lr){4-5}', 'Months & Never-triggered controls & Clean controls & CVAR & VAR without controls \\', ...
      '\midrule'}, lines, {'\bottomrule', '\end{tabular}', ['\par\smallskip\parbox{0.95\textwidth}{\footnotesize ', note, '}'], '\end{table}'}];
fid = fopen(fullfile(op, 'tab_eb_lpdid.tex'), 'w'); fprintf(fid, '%s\n', t{:}); fclose(fid);
fprintf('written results/eb/7_lpdid.csv and tab_eb_lpdid.tex\n');
end

function [b, se] = ols_cluster(y, X, g)
% OLS with cluster-robust standard errors (CRV1, as statsmodels' cov_type='cluster').
Pi = pinv(X); b = Pi * y; u = y - X * b;
Hi = pinv(X' * X);
[gu, ~, gi] = unique(g); G = numel(gu);
meat = zeros(size(X, 2));
for k = 1:G
    s = X(gi == k, :)' * u(gi == k);
    meat = meat + s * s';
end
[N, Kp] = size(X);
V = Hi * meat * Hi * (G / (G - 1)) * ((N - 1) / (N - Kp));
se = sqrt(diag(V));
end

function s = cell3(e, lo, hi)
s = sprintf('%s [%s, %s]', f2(e), f2(lo), f2(hi));
end

function s = f2(x)
if isempty(x) || ~isfinite(x), s = ''; else, s = strrep(sprintf('%.2f', x), '-', '$-$'); end
end
