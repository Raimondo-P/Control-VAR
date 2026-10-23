function cc = ak_complete(Da, outcome, donors, last)
%AK_COMPLETE  Donors with no missing month over the window where Alaska is observed, up to `last`.
Y = Da.out.(outcome);
keep = Da.dates <= last;
Y = Y(keep, :);
ak = Y(:, strcmp(Da.states, 'AK'));
obs = find(~isnan(ak));
if isempty(obs), cc = {}; return, end
win = Y(obs(1):obs(end), :);
cc = {};
for s = donors(:)'
    j = find(strcmp(Da.states, s{1}), 1);
    if ~isempty(j) && all(~isnan(win(:, j))), cc{end + 1} = s{1}; end %#ok<AGROW>
end
end
