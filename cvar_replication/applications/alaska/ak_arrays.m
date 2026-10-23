function A = ak_arrays(D, outcome, cc, last)
%AK_ARRAYS  Alaska, the controls cc (cell of state codes), the dividend and the oil change for one
%   outcome, up to the month `last` (NaT for the whole sample), restricted to the longest block of
%   consecutive complete months. A has fields dates, y, C, W, doil.
Y = D.out.(outcome);
[~, ic] = ismember(cc, D.states);
M = [Y(:, strcmp(D.states, 'AK')), Y(:, ic), D.W, D.doil];
keep = true(size(D.dates));
if ~isnat(last), keep = D.dates <= last; end
M = M(keep, :); dt = D.dates(keep);
ok = all(~isnan(M), 2);
[s, e] = longest_run(ok);
if any(ok) && (e - s + 1) < sum(ok)
    fprintf('  WARNING %s %s: internal gap, kept %s to %s (%d of %d complete months)\n', outcome, ...
        strjoin(cc, ','), datestr(dt(s), 'yyyy-mm'), datestr(dt(e), 'yyyy-mm'), e - s + 1, sum(ok));
end
if isempty(s)
    A = struct('dates', datetime.empty(0, 1), 'y', [], 'C', zeros(0, numel(cc)), 'W', [], 'doil', []);
    return
end
M = M(s:e, :);
A = struct('dates', dt(s:e), 'y', M(:, 1), 'C', M(:, 2:1 + numel(cc)), 'W', M(:, end - 1), 'doil', M(:, end));
end

function [s, e] = longest_run(ok)
s = []; e = []; best = 0; st = [];
ok = [ok(:); false];
for i = 1:numel(ok)
    if ok(i) && isempty(st)
        st = i;
    elseif ~ok(i) && ~isempty(st)
        if i - st > best
            best = i - st; s = st; e = i - 1;
        end
        st = [];
    end
end
end
