function ak_main(cfg, opt)
%AK_MAIN  The Alaska Permanent Fund Dividend, CVAR in changes with weights estimated in changes
%   (Python ak_1_application.py).
%   Weights: dY1_t = c + b' dY0_t + sum_{j=0..12} k_j' (oil, W)_{t-j} + e_t, b >= 0, with HAC tests
%   of sum(b) = 1 and b = 1/K. VAR in (oil, W, change in the cycle), responses cumulated; p = 12
%   (main) and the Ljung-Box rule. Bands (90 and 68 percent), the same residual draws for every
%   specification:
%     equal      equal weights
%     est_fixed  estimated weights held fixed
%     est_full   weights re-estimated in each draw on a moving-block resample of the rows of the
%                weights regression, then a residual draw of the VAR on those weights (bands quoted)
%     var        VAR without controls
%     controls_response  the VAR with Alaska and the weighted controls (Proposition 1)
%   opt (optional) restricts the run: fields outcomes, sets, samples, pmodes (cell arrays).
%   Outputs (results/alaska): 1_responses.csv, 1_loading.csv, 1_weights.csv
if nargin < 1 || isempty(cfg), cfg = config(); end
if nargin < 2, opt = struct(); end
K = ak_constants();
if ~isfield(opt, 'outcomes'), opt.outcomes = K.outcomes; end
if ~isfield(opt, 'sets'), opt.sets = fieldnames(K.sets)'; end
if ~isfield(opt, 'samples'), opt.samples = K.samples_main(:, 1)'; end
if ~isfield(opt, 'pmodes'), opt.pmodes = {'p12', 'rule'}; end
out = fullfile(cfg.results, 'alaska'); if ~exist(out, 'dir'), mkdir(out); end
D = ak_load(cfg, true);
items = {};
for o = opt.outcomes
    for smp = opt.samples
        last = K.samples_main{strcmp(K.samples_main(:, 1), smp{1}), 2};
        for cs = opt.sets
            for pm = opt.pmodes
                items(end + 1, :) = {o{1}, smp{1}, cs{1}, K.sets.(cs{1}), last, pm{1}}; %#ok<AGROW>
            end
        end
    end
end
n = size(items, 1);
fprintf('%d specifications, B = %d\n', n, cfg.B_app);
R = cell(n, 1); L = cell(n, 1); Wt = cell(n, 1); err = strings(n, 1);
B = cfg.B_app; Bc = min(cfg.B_app, cfg.B_cr); H = cfg.H; ell = cfg.block;
parfor i = 1:n
    try
        [R{i}, L{i}, Wt{i}] = one_spec(D, items(i, :), B, Bc, H, ell, K);
    catch e
        err(i) = sprintf('%s %s %s %s: %s', items{i, 1}, items{i, 2}, items{i, 3}, items{i, 6}, e.message);
    end
end
for i = find(err ~= "")', fprintf('FAILED %s\n', err(i)); end
ok = err == "";
writetable(struct2table([R{ok}]), fullfile(out, '1_responses.csv'));
writetable(struct2table([L{ok}]), fullfile(out, '1_loading.csv'));
writetable(struct2table([Wt{ok}]), fullfile(out, '1_weights.csv'));
fprintf('written to %s\n', out);
end

function [rows, ld, wt] = one_spec(D, it, B, Bc, H, ell, K)
[outcome, sample, cs, cc, last, pmode] = it{:};
A = ak_arrays(D, outcome, cc, last);
base = struct('outcome', string(outcome), 'sample', string(sample), 'controls', string(cs), ...
    'T', numel(A.y), 'start', string(datestr(A.dates(1), 'yyyy-mm')), 'end', string(datestr(A.dates(end), 'yyyy-mm')));
k = numel(cc);
[dy, X, F] = ak_design_changes(A, K.q);
w_hat = ak_wnnls(dy, X, F);
[b1, V1, r2] = ak_wols(dy, X, F, 12);
[s1, se1, p1, pe] = ak_loading_tests(b1, V1);
rows_b = ak_mbb(cvar.rng_stream(22), numel(dy), ell, B);
Wd = zeros(B, k);
for b = 1:B
    rb = rows_b(b, :);
    Wd(b, :) = ak_wnnls(dy(rb), X(rb, :), F(rb, :))';
end
sq = cvar.quantile7(sum(Wd, 2), [0.05 0.95]);
eq = ones(k, 1) / k;
if strcmp(pmode, 'p12')
    p = 12;
else
    p = cvar.lag_rule(ak_fit_changes(A, w_hat, 1).Z, [], [2 3], 12);
end
base.pmode = string(pmode); base.p = p;
r_eq = ak_fit_changes(A, eq, p);
r_w = ak_fit_changes(A, w_hat, p);
nU = size(r_w.U, 1);
idx = randi(cvar.rng_stream(21), nU, B, nU);
keys = {'cycle', 'ice', 'policy'};
Dd = struct();
for sp = {'equal', 'est_fixed', 'est_full'}
    for kk = keys, Dd.(sp{1}).(kk{1}) = zeros(B, H + 1); end
end
Dv = zeros(B, H + 1);
for b = 1:B
    ib = idx(b, :)';
    d1 = cvar.draw_conditional(r_eq, ib, H, false);
    d2 = cvar.draw_conditional(r_w, ib, H, false);
    d3 = cvar.draw_conditional(ak_fit_changes(A, Wd(b, :)', p), ib, H, false);
    for kk = keys
        Dd.equal.(kk{1})(b, :) = d1.(kk{1})(:, 1)';
        Dd.est_fixed.(kk{1})(b, :) = d2.(kk{1})(:, 1)';
        Dd.est_full.(kk{1})(b, :) = d3.(kk{1})(:, 1)';
    end
    Dv(b, :) = cvar.nc_draw(r_w, ib, H)';
end
pe_ = cvar.responses(r_eq, H); pw_ = cvar.responses(r_w, H);
Pt.equal = pe_; Pt.est_fixed = pw_; Pt.est_full = pw_;
rows = struct([]);
for sp = {'equal', 'est_fixed', 'est_full'}
    for kk = keys
        rr = ak_band_rows(base, sp{1}, kk{1}, Pt.(sp{1}).(kk{1})(:, 1), Dd.(sp{1}).(kk{1}), K.horizons);
        rows = [rows, rr]; %#ok<AGROW>
    end
end
rows = [rows, ak_band_rows(base, 'var', 'cycle', pw_.no_controls(:, 1), Dv, K.horizons)];
cr = cvar.controls_response(r_w, H, Bc, 13);
for kk = {'treated', 'controls', 'difference'}
    rows = [rows, ak_band_rows(base, 'controls_response', kk{1}, cr.point.(kk{1})(:, 1), ...
        cr.draws.(kk{1})(:, :, 1), K.horizons)]; %#ok<AGROW>
end
ld = base;
ld.K = k; ld.r2 = r2; ld.sum_nnls = sum(w_hat); ld.sum_nnls_lo90 = sq(1); ld.sum_nnls_hi90 = sq(2);
ld.sum_ols = s1; ld.se_ols = se1; ld.p_sum1 = p1; ld.p_equal = pe;
ld.n_positive = sum(w_hat > 0.005); ld.max_weight = max(w_hat);
wt = struct([]);
for j = 1:k
    w = struct('outcome', string(outcome), 'sample', string(sample), 'controls', string(cs), ...
        'pmode', string(pmode), 'control', string(cc{j}), 'w_nnls', w_hat(j), 'w_ols', b1(j), ...
        'w_ols_se', sqrt(V1(j, j)), 'equal', eq(j));
    if isempty(wt), wt = w; else, wt(end + 1) = w; end %#ok<AGROW>
end
end
