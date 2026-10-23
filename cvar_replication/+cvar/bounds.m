function T = bounds(cr, lams, horizons, alpha, grid)
%BOUNDS  Proposition 2 from a controls_response result: bands for the bounds
%   [gamma^G - lam |gamma^C|, gamma^G + lam |gamma^C|] (the alpha/2 quantile of the lower end and
%   the 1 - alpha/2 quantile of the upper end over the draws), and the smallest lam on `grid` at
%   which the band contains the response of the treated outcome in the same VAR. horizons are in
%   periods after the surprise (0 = impact). Returns a table.
if nargin < 2 || isempty(lams), lams = [0, 0.25, 0.5]; end
if nargin < 3 || isempty(horizons), horizons = [6, 12]; end
if nargin < 4 || isempty(alpha), alpha = 0.10; end
if nargin < 5 || isempty(grid), grid = round((0:0.01:3) * 100) / 100; end
G = cr.draws.difference; S = cr.draws.controls; Yd = cr.point.treated;
rows = {};
for j = 1:numel(cr.names)
    for h = horizons(:)'
        g = G(:, h + 1, j); s = abs(S(:, h + 1, j));
        row = struct('outcome', string(cr.names{j}), 'h', h, 'treated_response', Yd(h + 1, j), ...
            'difference', cr.point.difference(h + 1, j), 'controls', cr.point.controls(h + 1, j));
        for l = lams
            [lo, hi] = bd(g, s, l, alpha);
            row.(matlab.lang.makeValidName(sprintf('lam%g_lower', l))) = lo;
            row.(matlab.lang.makeValidName(sprintf('lam%g_upper', l))) = hi;
        end
        target = Yd(h + 1, j);
        row.breakdown_lambda = NaN;
        for l = grid
            [lo, hi] = bd(g, s, l, alpha);
            if lo <= target && target <= hi
                row.breakdown_lambda = l; break
            end
        end
        rows{end + 1} = row; %#ok<AGROW>
    end
end
T = struct2table([rows{:}]);
end

function [lo, hi] = bd(g, s, l, alpha)
lo = cvar.quantile7(g - l * s, alpha / 2);
hi = cvar.quantile7(g + l * s, 1 - alpha / 2);
end
