function q = quantile7(x, p)
%QUANTILE7  Quantiles along the first dimension with linear interpolation (numpy's default,
%   Hyndman and Fan type 7), without the Statistics Toolbox. x is B x ... ; q has size
%   numel(p) x the other dimensions of x.
sz = size(x);
B = sz(1);
xs = sort(reshape(x, B, []), 1);
q = zeros(numel(p), size(xs, 2));
for i = 1:numel(p)
    h = (B - 1) * p(i) + 1;
    lo = floor(h); hi = ceil(h);
    q(i, :) = xs(lo, :) + (h - lo) * (xs(hi, :) - xs(lo, :));
end
q = reshape(q, [numel(p), sz(2:end)]);
end
