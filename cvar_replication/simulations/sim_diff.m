function sim_diff(cfg)
%SIM_DIFF  Monte Carlo of the CVAR in changes with imposed weights (Theorem thm:diff; Python
%   mc_diff_branch.py). Monthly DGP, T = 200 after a burn-in of 100:
%     F_t = F_{t-1} + eta_t; Y0_kt = F_t + v_kt, v_kt = 0.5 v_k,t-1 + sigma_v N(0,1);
%     Y1_t = F_t + e_t + sum_{j=0..2} tau_j W_{t-j}, tau = (0.5, 0.3, 0.1);
%     e_t = e_{t-1} + sigma_e N(0,1) (own trend) or 0.8 e_{t-1} + sigma_e N(0,1) (stationary);
%     W_t = 0.7 W_{t-1} + c eta_t + eps_t.
%   True cumulated response Gamma_h = sum_{j<=min(h,2)} tau_j 0.7^(h-j).
%   Designs: own_trend (K = 4, sigma_v = 0.5, sigma_e = 0.3, c = 1), stationary (sigma_e = 0.5),
%   noisy_single (one control, sigma_v = 2), no_reaction (c = 0).
%   Estimators, 4 lags: CVAR in changes with equal weights; VAR without controls in changes; CVAR in
%   levels with estimated weights (q = 4, L = 1) and the joint recentred bootstrap.
%   Outputs: results/simulations/mc_diff/mc_diff.csv, results/paper/tab_mc_diff.tex
if nargin < 1, cfg = config(); end
Rr = cfg.diff_reps; B = cfg.diff_boot; T = 200; p = 4; H = 12; HOR = [0 3 6 12];
DES = {'own_trend', true, 4, 0.5, 0.3, 1, 'Trend of the treated unit'; 'stationary', false, 4, 0.5, 0.5, 1, 'Stationary cycle'; ...
       'noisy_single', true, 1, 2.0, 0.3, 1, 'One noisy control'; 'no_reaction', true, 4, 0.5, 0.3, 0, 'No reaction to the common shock'};
EST = {'cvar_diff', 'CVAR in changes'; 'var_diff', 'VAR without controls'; 'cvar_levels', 'CVAR in levels'};
tau = [0.5 0.3 0.1];
g = zeros(H + 1, 1);
for h = 0:H, for j = 0:min(h, 2), g(h + 1) = g(h + 1) + tau(j + 1) * 0.7^(h - j); end, end
out = fullfile(cfg.results, 'simulations', 'mc_diff'); if ~exist(out, 'dir'), mkdir(out); end
rows = struct([]);
for di = 1:size(DES, 1)
    dd = DES(di, :);
    PT = cell(Rr, 3); LO = cell(Rr, 3); HI = cell(Rr, 3);
    parfor rep = 1:Rr
        s = RandStream('mt19937ar', 'Seed', 100000 * (di - 1) + rep - 1);
        [y, C, W] = simulate(dd, T, s, tau);
        w = ones(dd{3}, 1) / dd{3};
        pr = cell(1, 3); lr = cell(1, 3); hr = cell(1, 3);
        try
            r = cvar.fit(y, C, W, struct('p', p, 'weights', w, 'difference', true));
            bt = cvar.bootstrap(r, B, H, struct('seed', rep - 1, 'no_controls', true));
            [l1, h1] = cvar.bands(bt, 'cycle'); [l2, h2] = cvar.bands(bt, 'no_controls');
            pr(1:2) = {bt.point.cycle(:, 1), bt.point.no_controls(:, 1)}; lr(1:2) = {l1(:, 1), l2(:, 1)}; hr(1:2) = {h1(:, 1), h2(:, 1)};
        catch
        end
        try
            r2 = cvar.fit(y, C, W, struct('p', p, 'q', 4, 'L', 1));
            b2 = cvar.bootstrap(r2, B, H, struct('seed', rep - 1 + 7, 'no_controls', false));
            [l3, h3] = cvar.bands(b2, 'cycle');
            pr{3} = b2.point.cycle(:, 1); lr{3} = l3(:, 1); hr{3} = h3(:, 1);
        catch
        end
        PT(rep, :) = pr; LO(rep, :) = lr; HI(rep, :) = hr;
    end
    for e = 1:3
        ok = ~cellfun(@isempty, PT(:, e));
        if ~any(ok), continue, end
        P = [PT{ok, e}]'; L = [LO{ok, e}]'; U = [HI{ok, e}]';
        for h = HOR
            i = h + 1;
            r = struct('design', string(dd{1}), 'estimator', string(EST{e, 1}), 'h', h, 'true', g(i), 'reps', sum(ok), ...
                'mean', mean(P(:, i)), 'bias', mean(P(:, i)) - g(i), 'rmse', sqrt(mean((P(:, i) - g(i)).^2)), ...
                'coverage', mean(L(:, i) <= g(i) & g(i) <= U(:, i)), 'width', mean(U(:, i) - L(:, i)));
            if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
        end
    end
end
Rt = struct2table(rows); writetable(Rt, fullfile(out, 'mc_diff.csv')); disp(Rt);
f = @(x) strrep(sprintf('%.2f', x), '-', '$-$');
lines = {};
for di = 1:size(DES, 1)
    lines{end + 1} = ['\multicolumn{8}{l}{', DES{di, 7}, '} \\']; %#ok<AGROW>
    for e = 1:3
        z = Rt(Rt.design == DES{di, 1} & Rt.estimator == EST{e, 1}, :);
        if isempty(z), continue, end
        c = [EST(e, 2), arrayfun(@(h) f(z.bias(z.h == h)), [0 6 12], 'UniformOutput', false), ...
             arrayfun(@(h) f(z.coverage(z.h == h)), [0 6 12], 'UniformOutput', false), {f(z.width(z.h == 6))}];
        lines{end + 1} = [strjoin(c, ' & '), ' \\']; %#ok<AGROW>
    end
    lines{end + 1} = '\midrule'; %#ok<AGROW>
end
lines = lines(1:end - 1);
note = sprintf(['%d replications, $T=%d$, %d lags, %d bootstrap draws in each replication. Cumulated response to a unit ', ...
    'innovation in the policy; bias relative to the true response, coverage of 90 percent percentile bands, and mean ', ...
    'width of the band. Data generating process and designs in the text.'], Rr, T, p, B);
t = [{'\begin{table}[htbp]', '\centering', '\caption{The CVAR in changes with imposed weights\label{tab:mc-diff}}', '\small', ...
      '\begin{tabular}{lccccccc}', '\toprule', ' & \multicolumn{3}{c}{Bias} & \multicolumn{3}{c}{Coverage} & Width \\', ...
      '\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}', 'Horizon & 0 & 6 & 12 & 0 & 6 & 12 & 6 \\', '\midrule'}, lines, ...
     {'\bottomrule', '\end{tabular}', ['\par\smallskip\parbox{0.95\textwidth}{\footnotesize ', note, '}'], '\end{table}'}];
op = cfg.paper; if ~exist(op, 'dir'), mkdir(op); end
fid = fopen(fullfile(op, 'tab_mc_diff.tex'), 'w'); fprintf(fid, '%s\n', t{:}); fclose(fid);
fprintf('written to %s and tab_mc_diff.tex\n', out);
end

function [Y1, Y0, W] = simulate(dd, T, s, tau)
burn = 100; n = T + burn; K = dd{3};
eta = randn(s, n, 1); F = cumsum(eta);
z = dd{4} * randn(s, n, K); v = zeros(n, K);
for t = 2:n, v(t, :) = 0.5 * v(t - 1, :) + z(t, :); end
zeta = dd{5} * randn(s, n, 1); e = zeros(n, 1);
for t = 2:n
    if dd{2}, e(t) = e(t - 1) + zeta(t); else, e(t) = 0.8 * e(t - 1) + zeta(t); end
end
ew = randn(s, n, 1); W = zeros(n, 1);
for t = 2:n, W(t) = 0.7 * W(t - 1) + dd{6} * eta(t) + ew(t); end
eff = zeros(n, 1);
for j = 1:3, eff(j:end) = eff(j:end) + tau(j) * W(1:n - j + 1); end
Y1 = F + e + eff; Y0 = F + v;
k = burn + 1:n; Y1 = Y1(k); Y0 = Y0(k, :); W = W(k);
end
