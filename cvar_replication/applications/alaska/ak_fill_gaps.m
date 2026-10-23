function X = ak_fill_gaps(X, dates, label)
%AK_FILL_GAPS  Linear interpolation of missing months inside each series, at most two in a row,
%   never at the start or the end (as pandas' interpolate(limit=2, limit_area='inside')): within a
%   longer gap only its first two months are filled.
% Linear interpolation of missing months inside each series, at most two in a row, never at the
% start or the end (as pandas' interpolate(limit=2, limit_area='inside')): within a longer gap only
% its first two months are filled.
new = false(size(X, 1), 1);
for j = 1:size(X, 2)
    x = X(:, j);
    ok = find(~isnan(x));
    if numel(ok) < 2, continue, end
    for g = 1:numel(ok) - 1
        a = ok(g); b = ok(g + 1);
        if b - a > 1
            for t = a + 1:min(b - 1, a + 2)
                x(t) = x(a) + (x(b) - x(a)) * (t - a) / (b - a);
                new(t) = true;
            end
        end
    end
    X(:, j) = x;
end
if any(new)
    fprintf('  %s: interpolated %s\n', label, strjoin(cellstr(datestr(dates(new), 'yyyy-mm')), ' '));
end
end
