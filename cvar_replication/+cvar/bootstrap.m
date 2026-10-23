function br = bootstrap(res, B, H, bo)
%BOOTSTRAP  Residual bootstrap with percentile bands (Section 4).
%   br = cvar.bootstrap(res, B, H, bo), with options bo:
%   seed          seed of the draws ([] for the global stream)
%   no_controls   also draw the VAR without controls (true)
%   indices       B x (T - p) residual rows for the conditional scheme (joint bootstrap across units)
%   scheme        'auto' (joint for one outcome with weights estimated in levels, conditional
%                 otherwise), 'joint' (Theorem thm:joint) or 'conditional' (Theorem thm:resid; with
%                 estimated weights valid when the first-step term is zero)
%   recentre      shift the joint draws by the estimate minus the bootstrap population value (true)
%   recentre_mult length of the population path, in multiples of the sample (50)
%   br.point and br.draws hold cycle, policy, ice and no_controls; draws are B x (H+1) x m.
if nargin < 2 || isempty(B), B = 999; end
if nargin < 3 || isempty(H), H = 36; end
if nargin < 4, bo = struct(); end
d = struct('seed', [], 'no_controls', true, 'indices', [], 'scheme', 'auto', 'recentre', true, ...
    'recentre_mult', 50);
f = fieldnames(d);
for i = 1:numel(f), if ~isfield(bo, f{i}), bo.(f{i}) = d.(f{i}); end, end
s = cvar.rng_stream(bo.seed);
est = ~[res.units.imposed];
scheme = bo.scheme;
if strcmp(scheme, 'auto')
    if res.m == 1 && est(1) && res.off == 0, scheme = 'joint'; else, scheme = 'conditional'; end
end
n = size(res.U, 1); m = res.m;
dc = zeros(B, H + 1, m); di = zeros(B, H + 1, m); dp = zeros(B, H + 1); dn = zeros(B, H + 1, m);
if strcmp(scheme, 'joint')
    if res.m ~= 1 || ~est(1) || res.off ~= 0
        error('the joint scheme is for a single outcome with estimated weights in levels');
    end
    jt = cvar.jt_fit(res);
    nj = size(jt.U, 1);
    for b = 1:B
        Js = cvar.var_sim(jt.A, jt.U, jt.Jm(1:res.opts.p, :), res.opts.p, size(jt.Jm, 1), randi(s, nj, nj, 1), jt.veJ);
        dd = cvar.joint_from_path(res, jt, Js, H, true);
        dc(b, :, :) = reshape(dd.cycle, 1, H + 1, m); di(b, :, :) = reshape(dd.ice, 1, H + 1, m);
        dp(b, :) = dd.policy';
        if bo.no_controls
            dn(b, :, :) = reshape(cvar.nc_draw(res, randi(s, n, n, 1), H), 1, H + 1, m);
        end
    end
else
    idx = bo.indices;
    if isempty(idx), idx = randi(s, n, B, n); end
    for b = 1:B
        dd = cvar.draw_conditional(res, idx(b, :)', H, bo.no_controls);
        dc(b, :, :) = reshape(dd.cycle, 1, H + 1, m); di(b, :, :) = reshape(dd.ice, 1, H + 1, m);
        dp(b, :) = dd.policy';
        if bo.no_controls, dn(b, :, :) = reshape(dd.no_controls, 1, H + 1, m); end
    end
end
pt = cvar.responses(res, H);
point = struct('cycle', pt.cycle, 'policy', pt.policy, 'ice', pt.ice);
pop = [];
if strcmp(scheme, 'joint') && bo.recentre
    if isempty(bo.seed), ps = []; else, ps = bo.seed + 99991; end
    pop = joint_population(res, jt, H, bo.recentre_mult, ps);
    dc = dc + reshape(point.cycle - pop.cycle, 1, H + 1, m);
    di = di + reshape(point.ice - pop.ice, 1, H + 1, m);
    dp = dp + (point.policy - pop.policy)';
end
draws = struct('cycle', dc, 'policy', dp, 'ice', di);
if bo.no_controls
    point.no_controls = pt.no_controls;
    draws.no_controls = dn;
end
br = struct('point', point, 'draws', draws, 'names', {res.names}, 'scheme', scheme, 'population', pop);
end

function out = joint_population(res, jt, H, mult, seed)
% Bootstrap population value of the responses: the estimator applied to one long path of the
% fitted joint VAR (mult times the sample length), without deterministic regressors and level shifts.
p = res.opts.p;
s = cvar.rng_stream(seed);
n = mult * size(jt.Jm, 1);
A0 = jt.A(1:1 + size(jt.Jm, 2) * p, :);
Js = cvar.var_sim(A0, jt.U, jt.Jm(1:p, :), p, n, randi(s, size(jt.U, 1), n - p, 1), []);
out = cvar.joint_from_path(res, jt, Js, H, false);
end
