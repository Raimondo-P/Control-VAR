function S = ca_series(D, cc, measure, first)
%CA_SERIES  Seasonally adjusted California and control prices (cents) and the monthly change of the
%   cost measure, on the months where the measure exists; the change at the splice month is zero.
lev = D.(['cost_', measure]);
k = ~isnan(lev);
mon = month(D.dates(k));
S.dates = D.dates(k);
S.y = ca_sa_changes(100 * D.gas_ca(k), {mon});
S.C = zeros(sum(k), numel(cc));
for j = 1:numel(cc), S.C(:, j) = ca_sa_changes(100 * D.(cc{j})(k), {mon}); end
S.lev = lev(k);
S.dose = [0; diff(S.lev)];
if strcmp(measure, 'spliced') && ~isnat(first)
    i = find(S.dates == first, 1);
    if ~isempty(i) && i > 1, S.dose(i) = 0; end
end
S.credit = D.credit(k);
end
