function Ds = ca_sub(D, s0, s1)
%CA_SUB  The panel between the months s0 and s1 (NaT for open ends), as D.loc[s0:s1] in pandas.
k = true(size(D.dates));
if ~isnat(s0), k = k & D.dates >= s0; end
if ~isnat(s1), k = k & D.dates <= s1; end
Ds = D;
for f = fieldnames(D)', Ds.(f{1}) = D.(f{1})(k); end
end
