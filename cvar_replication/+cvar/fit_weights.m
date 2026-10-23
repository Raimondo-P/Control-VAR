function fw = fit_weights(y, Y0, pol, D1, q, L, constrained, breaks, max_breaks, break_cv, nw)
%FIT_WEIGHTS  Weights with optional level shifts in the counterfactual cycle.
%   breaks: [] (no shifts), 'auto' (sequential sup-Wald tests on the cycle net of the policy
%   terms, adding a step while the statistic exceeds break_cv, at most max_breaks), or a vector of
%   first observations (1-based) of the new levels.
%   fw has fields a, b, shift, steps (T x nb), breaks, cycle, supf_before, supf_after.
if nargin < 5 || isempty(q), q = 12; end
if nargin < 6 || isempty(L), L = 1; end
if nargin < 7 || isempty(constrained), constrained = true; end
if nargin < 8, breaks = []; end
if nargin < 9 || isempty(max_breaks), max_breaks = 2; end
if nargin < 10 || isempty(break_cv), break_cv = 8.85; end
if nargin < 11 || isempty(nw), nw = 12; end
y = y(:);
T = numel(y);
[a, b, sh, st, c] = est([]);
[c0, off] = cvar.net_cycle(c, pol, q);
[sf0, kb] = cvar.supf_mean_break(c0, [], nw);
bd = [];
if ischar(breaks) || isstring(breaks)
    sf = sf0;
    while sf > break_cv && numel(bd) < max_breaks
        bd(end + 1) = kb + off; %#ok<AGROW>
        [a, b, sh, st, c] = est(bd);
        [cn, off] = cvar.net_cycle(c, pol, q);
        [sf, kb] = cvar.supf_mean_break(cn, [], nw);
    end
elseif ~isempty(breaks)
    bd = breaks(:)';
    [a, b, sh, st, c] = est(bd);
end
cn = cvar.net_cycle(c, pol, q);
sf1 = cvar.supf_mean_break(cn, [], nw);
fw = struct('a', a, 'b', b, 'shift', sh, 'steps', st, 'breaks', bd, 'cycle', c, ...
            'supf_before', sf0, 'supf_after', sf1);

    function [a, b, sh, stm, c] = est(bdv)
        if isempty(bdv)
            stx = [];
        else
            stx = zeros(T, numel(bdv));
            for i = 1:numel(bdv)
                stx(:, i) = double((1:T)' >= bdv(i));
            end
        end
        [a, b, sh] = cvar.dols(y, Y0, pol, D1, stx, q, L, constrained);
        if isempty(stx), stm = zeros(T, 0); else, stm = stx; end
        c = y - a - Y0 * b - stm * sh;
    end
end
