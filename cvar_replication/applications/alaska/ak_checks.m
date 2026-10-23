function ak_checks(cfg, opt)
%AK_CHECKS  Two checks of the main Alaska specification (pre-2020, six-state donors, weights in
%   changes, oil first, 12 lags) (Python ak_2_checks.py).
%   A. Sensitivity bounds (Proposition 2) for lam = 0, 0.25, 0.5, 1, with breakdown_lambda (the band
%      contains the response of Alaska in the VAR with the controls added) and zero_lambda (the band
%      contains zero). Weights held at their estimates.
%   B. Seasonal adjustment: adjusted series; adjusted plus month dummies; unadjusted plus month
%      dummies; unadjusted plus month dummies times a trend; unadjusted plus month dummies by
%      decade, in the weights regression and in the VAR. Weights held at their estimates.
%   opt.outcomes restricts the outcomes; opt.skip_bounds = true skips part A.
%   Outputs (results/alaska): 2_bounds.csv, 2_seasonal.csv
if nargin < 1 || isempty(cfg), cfg = config(); end
if nargin < 2, opt = struct(); end
K = ak_constants();
if ~isfield(opt, 'outcomes'), opt.outcomes = K.outcomes; end
if ~isfield(opt, 'skip_bounds'), opt.skip_bounds = false; end
out = fullfile(cfg.results, 'alaska'); if ~exist(out, 'dir'), mkdir(out); end
last = datetime(2019, 12, 1); cc = K.sets.jm_emp; p = 12; H = cfg.H;
D = ak_load(cfg, true);
Dn = ak_load_nsa(cfg, D);
no = numel(opt.outcomes);
% A. bounds
if ~opt.skip_bounds
    BD = cell(no, 1); Bc = cfg.B_app;      % as ak_2: the bounds use the full number of draws
    parfor i = 1:no
        A = ak_arrays(D, opt.outcomes{i}, cc, last);
        [dy, X, F] = ak_design_changes(A, K.q);
        r = ak_fit_changes(A, ak_wnnls(dy, X, F), p);
        cr = cvar.controls_response(r, H, Bc, 13);
        bd = cvar.bounds(cr, [0 0.25 0.5 1], [3 6 12 24]);
        Gd = cr.draws.difference(:, :, 1); Sd = abs(cr.draws.controls(:, :, 1));
        grid = round((0:0.01:3) * 100) / 100;
        zl = nan(height(bd), 1);
        for k = 1:height(bd)
            h = bd.h(k); g = Gd(:, h + 1); s = Sd(:, h + 1);
            for l = grid
                if cvar.quantile7(g - l * s, 0.05) <= 0 && 0 <= cvar.quantile7(g + l * s, 0.95)
                    zl(k) = l; break
                end
            end
        end
        bd.zero_lambda = zl;
        bd.outcome = repmat(string(opt.outcomes{i}), height(bd), 1);
        BD{i} = bd;
    end
    writetable(vertcat(BD{:}), fullfile(out, '2_bounds.csv'));
end
% B. seasonal adjustment
versions = {'SA', '', false; 'SA + months', 'fixed', false; 'NSA + months', 'fixed', true; ...
            'NSA + months x trend', 'trend', true; 'NSA + months x decade', 'decade', true};
items = {};
for o = opt.outcomes
    for v = 1:size(versions, 1), items(end + 1, :) = [o, versions(v, :)]; end %#ok<AGROW>
end
n = size(items, 1); SS = cell(n, 1); B = cfg.B_app;
parfor i = 1:n
    [o, version, kind, nsa] = items{i, :};
    if nsa, A = ak_arrays(Dn, o, cc, last); else, A = ak_arrays(D, o, cc, last); end
    M = ak_season_terms(A.dates, kind);
    [dy, X, F] = ak_design_changes(A, K.q);
    if ~isempty(M)       % month terms in the weights regression, on the rows kept there
        [dy, X, F] = design_with(A, K.q, M);
    end
    w = ak_wnnls(dy, X, F);
    r = cvar.fit(A.y, A.C, A.W, struct('p', p, 'other', A.doil, 'var_exog', M, 'weights', w, ...
        'difference', true, 'names', {{'AK'}}));
    nU = size(r.U, 1);
    idx = randi(cvar.rng_stream(21), nU, B, nU);
    Dc = zeros(B, H + 1); Dv = zeros(B, H + 1);
    for b = 1:B
        d = cvar.draw_conditional(r, idx(b, :)', H, false);
        Dc(b, :) = d.cycle(:, 1)';
        Dv(b, :) = cvar.nc_draw(r, idx(b, :)', H)';
    end
    pt = cvar.responses(r, H);
    [ws, ord] = sort(w, 'descend');
    lab = strjoin(arrayfun(@(k) sprintf('%s %.2f', cc{ord(k)}, ws(k)), find(ws > 0.005)', 'UniformOutput', false), ', ');
    rows = struct([]);
    specs = {'cvar', pt.cycle(:, 1), Dc; 'var', pt.no_controls(:, 1), Dv};
    for sp = 1:2
        q = cvar.quantile7(specs{sp, 3}, [0.05 0.95]);
        for h = K.horizons
            rr = struct('outcome', string(o), 'version', string(version), 'spec', string(specs{sp, 1}), ...
                'h', h, 'estimate', specs{sp, 2}(h + 1), 'lo90', q(1, h + 1), 'hi90', q(2, h + 1), ...
                'zero_in_90', q(1, h + 1) <= 0 && 0 <= q(2, h + 1), 'sum_weights', sum(w), ...
                'weights', string(lab), 'start', string(datestr(A.dates(1), 'yyyy-mm')), ...
                'end', string(datestr(A.dates(end), 'yyyy-mm')));
            if isempty(rows), rows = rr; else, rows(end + 1) = rr; end %#ok<AGROW>
        end
    end
    SS{i} = rows;
end
writetable(struct2table([SS{:}]), fullfile(out, '2_seasonal.csv'));
fprintf('written to %s\n', out);
end

function [dy, dX, F] = design_with(A, q, M)
% The regression in changes with seasonal terms (rows 2..T of M, aligned with the changes).
dy = diff(A.y); dX = diff(A.C, 1, 1);
pol = [A.doil, A.W]; pol = pol(2:end, :);
F = [ones(numel(dy), 1), cvar.lagmat(pol, 0:q), M(2:end, :)];
ok = all(~isnan(F), 2) & all(~isnan(dX), 2);
dy = dy(ok); dX = dX(ok, :); F = F(ok, :);
end
