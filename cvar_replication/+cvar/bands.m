function [lo, hi] = bands(br, key, alpha, method)
%BANDS  Percentile (or Hall) bands from bootstrap draws: br is the output of cvar.bootstrap or
%   cvar.controls_response, key the object ('cycle', 'ice', 'policy', 'no_controls', ...).
if nargin < 3 || isempty(alpha), alpha = 0.10; end
if nargin < 4 || isempty(method), method = 'percentile'; end
q = cvar.quantile7(br.draws.(key), [alpha / 2, 1 - alpha / 2]);
sz = size(br.draws.(key));
lo = reshape(q(1, :), [sz(2:end), 1]); hi = reshape(q(2, :), [sz(2:end), 1]);
if strcmp(method, 'hall')
    pt = br.point.(key);
    l2 = 2 * pt - hi; h2 = 2 * pt - lo; lo = l2; hi = h2;
elseif ~strcmp(method, 'percentile')
    error('method must be ''percentile'' or ''hall''');
end
end
