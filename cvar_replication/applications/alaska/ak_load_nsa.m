function Dn = ak_load_nsa(cfg, D)
%AK_LOAD_NSA  The same outcome panels from the unadjusted series (LAUS LAUST, CES SMU) in
%   applications/data/alaska/raw/bls_nsa.csv, on the dates of D. Missing LAUS months are
%   interpolated as in the adjusted series; the CES series are used as published.
K = ak_constants();
Bt = readtable(fullfile(cfg.data, 'alaska', 'raw', 'bls_nsa.csv'), 'VariableNamingRule', 'preserve');
bd = dateshift(datetime(Bt{:, 1}), 'start', 'month');
[in, ib] = ismember(D.dates, bd);
T = numel(D.dates);
    function v = colx(id)
        v = nan(T, 1);
        if ismember(id, Bt.Properties.VariableNames)
            x = Bt.(id); v(in) = x(ib(in));
        end
    end
out = struct();
for nm = {'epop', 'ur', 'lfpr'}
    M = zeros(T, numel(K.states));
    for s = 1:numel(K.states), M(:, s) = colx(sprintf('LAUST%s00000000000%s', K.fips(K.states{s}), K.laus.(nm{1}))); end
    out.(nm{1}) = ak_fill_gaps(M, D.dates, [nm{1}, ' (NSA)']);
end
for nm = fieldnames(K.ces)'
    M = zeros(T, numel(K.states));
    for s = 1:numel(K.states), M(:, s) = 100 * log(colx(sprintf('SMU%s00000%s01', K.fips(K.states{s}), K.ces.(nm{1})))); end
    out.(nm{1}) = M;
end
Dn = D; Dn.out = out;
end
