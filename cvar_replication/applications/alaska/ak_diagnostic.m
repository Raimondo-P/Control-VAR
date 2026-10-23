function ak_diagnostic(cfg, opt)
%AK_DIAGNOSTIC  Diagnostic of the weights estimated in changes (Proposition prop:wchanges) for the
%   main design (six donors, pre-2020) (Python ak_6_diagnostic.py).
%   A. Reproduction of the main common movement: principal components of the changes of all 50 other
%      states (constant and lags of the policies projected out); ratio of the loading of the weighted
%      donors to that of Alaska on each component, for estimated and equal weights.
%   B. Adjusted responses: from the VAR with Alaska and the weighted donors, difference minus
%      (1 - r)/r times the response of the donors, draw by draw, r at its estimate.
%   C. Robustness pools: weights estimated in changes from the Census West and from all states.
%   Outputs (results/alaska): 6_diagnostic.csv, 6_responses.csv, 6_weights.csv
if nargin < 1 || isempty(cfg), cfg = config(); end
if nargin < 2, opt = struct(); end
K = ak_constants();
outc = {'epop', 'ur', 'lfpr', 'nonfarm', 'leisure', 'retail'};
if isfield(opt, 'outcomes'), outc = opt.outcomes; end
out = fullfile(cfg.results, 'alaska'); if ~exist(out, 'dir'), mkdir(out); end
D = ak_load(cfg, true);
Da = ak_load_all(cfg, D);
st = ak_states_all();
donors_all = st(~strcmp(st, 'AK'));
pools.west = {'WA', 'OR', 'CA', 'ID', 'NV', 'UT', 'AZ', 'MT', 'WY', 'CO', 'NM', 'HI'};
pools.all = donors_all;
last = datetime(2019, 12, 1); p = 12; H = cfg.H; B = cfg.B_app; Bc = min(cfg.B_app, cfg.B_cr); ell = cfg.block;
n = numel(outc);
DG = cell(n, 1); RS = cell(n, 1); WT = cell(n, 1); err = strings(n, 1);
parfor i = 1:n
    try
        [DG{i}, RS{i}, WT{i}] = one_outcome(Da, outc{i}, donors_all, pools, K, last, p, H, B, Bc, ell);
    catch e
        err(i) = string(outc{i}) + ": " + e.message;
    end
end
for i = find(err ~= "")', fprintf('FAILED %s\n', err(i)); end
ok = err == "";
writetable(struct2table([DG{ok}]), fullfile(out, '6_diagnostic.csv'));
writetable(struct2table([RS{ok}]), fullfile(out, '6_responses.csv'));
writetable(struct2table([WT{ok}]), fullfile(out, '6_weights.csv'));
fprintf('written to %s\n', out);
end

function [dg, rs, wt] = one_outcome(Da, o, donors_all, pools, K, last, p, H, B, Bc, ell)
pool_all = ak_complete(Da, o, donors_all, last);
Aa = ak_arrays(Da, o, pool_all, last);
Xa = ak_partial_out(Aa, Aa.C, K.q);
ya = ak_partial_out(Aa, Aa.y, K.q);
Zs = (Xa - mean(Xa, 1)) ./ std(Xa, 1, 1);
[Uz, Sz, ~] = svd(Zs, 'econ');
sv = diag(Sz);
share = sv.^2 / sum(sv.^2);
PC = Uz(:, 1:3) .* sv(1:3)';
lam_ak = lsqminnorm(PC, ya);
lam_all = lsqminnorm(PC, Xa);
ratio = @(cc, w) (lam_all(:, idx_of(pool_all, cc)) * w(:)) ./ lam_ak;
rs = struct([]); dg = struct([]);
    function add(spec, pt, Dr)
        q9 = cvar.quantile7(Dr, [0.05 0.95]); q6 = cvar.quantile7(Dr, [0.16 0.84]);
        for h = 0:H
            r = struct('outcome', string(o), 'spec', string(spec), 'h', h, 'estimate', pt(h + 1), ...
                'lo90', q9(1, h + 1), 'hi90', q9(2, h + 1), 'lo68', q6(1, h + 1), 'hi68', q6(2, h + 1));
            if isempty(rs), rs = r; else, rs(end + 1) = r; end
        end
    end
    function adddg(wname, k, sw, rat, mult, npos)
        r = struct('outcome', string(o), 'weights', string(wname), 'K', k, 'sum_w', sw, ...
            'share_pc1', share(1), 'share_pc2', share(2), 'share_pc3', share(3), ...
            'ratio_pc1', rat(1), 'ratio_pc2', rat(2), 'ratio_pc3', rat(3), 'multiplier', mult, 'n_positive', npos);
        if isempty(dg), dg = r; else, dg(end + 1) = r; end
    end
% six donors: estimated and equal weights
cc = K.sets.jm_emp(ismember(K.sets.jm_emp, pool_all));
A = ak_arrays(Da, o, cc, last);
[w, r_w, pt_w, D_w, idx] = est_full(A, K, B, ell, p, H);
eq = ones(numel(cc), 1) / numel(cc);
r_eq = ak_fit_changes(A, eq, p);
D_eq = zeros(B, H + 1);
for b = 1:B
    d = cvar.draw_conditional(r_eq, idx(b, :)', H, false);
    D_eq(b, :) = d.cycle(:, 1)';
end
add('six_est', pt_w, D_w);
pe = cvar.responses(r_eq, H);
add('six_equal', pe.cycle(:, 1), D_eq);
wt = struct('outcome', cellstr(repmat(o, numel(cc), 1))', 'state', cc, 'weight', num2cell(w(:)'));
specs = {'est', r_w, w; 'equal', r_eq, eq};
for s = 1:2
    rat = ratio(cc, specs{s, 3});
    r1 = rat(1);
    if r1 > 0.2, m = (1 - r1) / r1; else, m = NaN; end
    cr = cvar.controls_response(specs{s, 2}, H, Bc, 13);
    dpt = cr.point.difference(:, 1); cpt = cr.point.controls(:, 1);
    dD = cr.draws.difference(:, :, 1); cD = cr.draws.controls(:, :, 1);
    add(sprintf('six_%s_controls', specs{s, 1}), cpt, cD);
    add(sprintf('six_%s_joint', specs{s, 1}), dpt, dD);
    if isfinite(m)
        add(sprintf('six_%s_adjusted', specs{s, 1}), dpt - m * cpt, dD - m * cD);
    end
    adddg(specs{s, 1}, numel(cc), sum(specs{s, 3}), rat, m, NaN);
end
% robustness pools
for pn = {'west', 'all'}
    ccp = pools.(pn{1})(ismember(pools.(pn{1}), pool_all));
    Ap = ak_arrays(Da, o, ccp, last);
    [wp, ~, ptp, Dp] = est_full(Ap, K, B, ell, p, H);
    add([pn{1}, '_est'], ptp, Dp);
    adddg([pn{1}, '_est'], numel(ccp), sum(wp), ratio(ccp, wp), NaN, sum(wp > 0.005));
end
for j = 1:numel(wt), wt(j).outcome = string(wt(j).outcome); wt(j).state = string(wt(j).state); end
end

function [w, r, pt, Dr, idx] = est_full(A, K, B, ell, p, H)
% Weights estimated in changes and the bootstrap of ak_main (same seeds).
[dy, X, F] = ak_design_changes(A, K.q);
w = ak_wnnls(dy, X, F);
rows = ak_mbb(cvar.rng_stream(22), numel(dy), ell, B);
r = ak_fit_changes(A, w, p);
nU = size(r.U, 1);
idx = randi(cvar.rng_stream(21), nU, B, nU);
Dr = zeros(B, H + 1);
for b = 1:B
    rb = rows(b, :);
    wb = ak_wnnls(dy(rb), X(rb, :), F(rb, :));
    d = cvar.draw_conditional(ak_fit_changes(A, wb, p), idx(b, :)', H, false);
    Dr(b, :) = d.cycle(:, 1)';
end
pr = cvar.responses(r, H);
pt = pr.cycle(:, 1);
end

function ix = idx_of(list, cc)
[~, ix] = ismember(cc, list);
end
