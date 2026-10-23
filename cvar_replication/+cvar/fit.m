function res = fit(y, controls, policy, opts)
%FIT  Control VAR estimator (Section 4 of the paper).
%   res = cvar.fit(y, controls, policy, opts)
%   y         T x m treated outcomes in levels
%   controls  T x K controls shared by all outcomes, or a cell array of m matrices
%   policy    T x 1 policy of the treated unit
%   opts      options, see cvar.defaults
%   res holds the data, the weights of each outcome (res.units), the VAR in (R, W, G) (res.Z,
%   res.A, res.U), the VAR without controls (res.nc) and the VAR with the weighted controls added
%   (res.cr, Proposition 1).
if nargin < 4, opts = struct(); end
o = cvar.defaults(opts);
[T, m] = size(y);
if iscell(controls), Ctr = controls; else, Ctr = repmat({controls}, 1, m); end
if numel(Ctr) ~= m, error('one control matrix per outcome'); end
W = policy(:);
if isempty(o.other), R = zeros(T, 0); else, R = reshape(o.other, T, []); end
pol = [R, W];
if iscell(o.breaks), brk = o.breaks; else, brk = repmat({o.breaks}, 1, m); end
if iscell(o.weights), wl = o.weights; else, wl = repmat({o.weights}, 1, m); end
if isscalar(o.difference), dfl = repmat(logical(o.difference), 1, m); else, dfl = logical(o.difference(:)'); end
units = struct('a', {}, 'b', {}, 'shift', {}, 'steps', {}, 'breaks', {}, 'cycle', {}, ...
    'supf_before', {}, 'supf_after', {}, 'rank', {}, 'D1', {}, 'imposed', {});
for j = 1:m
    if ~isempty(wl{j})
        w = wl{j}(:);
        units(j) = struct('a', 0, 'b', w, 'shift', zeros(0, 1), 'steps', zeros(T, 0), 'breaks', [], ...
            'cycle', y(:, j) - Ctr{j} * w, 'supf_before', NaN, 'supf_after', NaN, 'rank', NaN, ...
            'D1', [], 'imposed', true);
        continue
    end
    if dfl(j), error('a differenced outcome needs imposed weights'); end
    [mr, D1] = cvar.rank_step(Ctr{j}, o.rank, o.rank_level, o.rank_lags);
    fw = cvar.fit_weights(y(:, j), Ctr{j}, pol, D1, o.q, o.L, o.constrained, brk{j}, o.max_breaks, o.break_cv);
    fw.rank = mr; fw.D1 = D1; fw.imposed = false;
    units(j) = orderfields(fw, {'a', 'b', 'shift', 'steps', 'breaks', 'cycle', 'supf_before', ...
        'supf_after', 'rank', 'D1', 'imposed'});
end
off = double(any(dfl));
G = cvar.transform([units.cycle], dfl, off);
Z = [R(off + 1:end, :), W(off + 1:end), G];
if isempty(o.var_exog), ve = []; else, ve = reshape(o.var_exog, T, []); ve = ve(off + 1:end, :); end
[A, U] = cvar.var_fit(Z, o.p, ve);
names = o.names;
if isempty(names), names = arrayfun(@(j) sprintf('y%d', j), 1:m, 'UniformOutput', false); end
res = struct('opts', o, 'Y', y, 'controls', {Ctr}, 'R', R, 'W', W, 'pol', pol, 'units', units, ...
    'Z', Z, 'A', A, 'U', U, 'var_exog', ve, 'names', {names}, 'diff', dfl, 'off', off, ...
    'nR', size(R, 2), 'm', m, 'T', T);
% VAR without controls in (R, W, Y)
Zv = [R(off + 1:end, :), W(off + 1:end), cvar.transform(y, dfl, off)];
[Av, Uv] = cvar.var_fit(Zv, o.p, ve);
res.nc = struct('Z', Zv, 'A', Av, 'U', Uv);
% VAR with the weighted controls added, in (R, W, Y, C), C = b'Y0 (Proposition 1)
C = zeros(T, m);
for j = 1:m
    C(:, j) = Ctr{j} * units(j).b + units(j).steps * units(j).shift;
end
Za = [R(off + 1:end, :), W(off + 1:end), cvar.transform(y, dfl, off), cvar.transform(C, dfl, off)];
[Aa, Ua] = cvar.var_fit(Za, o.p, ve);
res.cr = struct('Z', Za, 'A', Aa, 'U', Ua);
end
