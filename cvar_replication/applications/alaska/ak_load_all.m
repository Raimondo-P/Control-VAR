function Da = ak_load_all(cfg, D)
%AK_LOAD_ALL  Outcome panels of all 50 states and the District of Columbia from the files
%   applications/data/alaska/raw/bls_all_*.csv, on the dates of D. If a series appears in more than
%   one file, the copy with the most observations is kept. Missing LAUS months are interpolated as
%   in the main data; the CES series are used as published. Da.states lists the columns.
K = ak_constants();
[st, fips] = ak_states_all();
files = dir(fullfile(cfg.data, 'alaska', 'raw', 'bls_all_*.csv'));
if isempty(files), error('no bls_all_*.csv files in applications/data/alaska/raw'); end
T = numel(D.dates);
best = containers.Map();      % series id -> [file index, observations]
tabs = cell(numel(files), 1); pos = cell(numel(files), 1);
for f = 1:numel(files)
    Bt = readtable(fullfile(files(f).folder, files(f).name), 'VariableNamingRule', 'preserve');
    [in, ib] = ismember(D.dates, dateshift(datetime(Bt{:, 1}), 'start', 'month'));
    tabs{f} = Bt; pos{f} = {in, ib};
    for v = Bt.Properties.VariableNames(2:end)
        nobs = sum(~isnan(Bt.(v{1})));
        if isKey(best, v{1}), prev = best(v{1}); else, prev = [0, -1]; end
        if nobs > prev(2)
            best(v{1}) = [f, nobs];
        end
    end
end
    function x = colx(id)
        x = nan(T, 1);
        if isKey(best, id)
            f = best(id); f = f(1);
            in = pos{f}{1}; ib = pos{f}{2};
            y = tabs{f}.(id); x(in) = y(ib(in));
        end
    end
out = struct();
for nm = {'epop', 'ur', 'lfpr'}
    M = zeros(T, numel(st));
    for s = 1:numel(st), M(:, s) = colx(sprintf('LASST%s00000000000%s', fips(st{s}), K.laus.(nm{1}))); end
    out.(nm{1}) = ak_fill_gaps(M, D.dates, nm{1});
end
for nm = fieldnames(K.ces)'
    M = zeros(T, numel(st));
    for s = 1:numel(st), M(:, s) = 100 * log(colx(sprintf('SMS%s00000%s01', fips(st{s}), K.ces.(nm{1})))); end
    out.(nm{1}) = M;
end
for nm = fieldnames(out)'
    miss = st(all(isnan(out.(nm{1})), 1));
    if ~isempty(miss), fprintf('  %s: no data for %s (left out of the pools)\n', nm{1}, strjoin(miss, ' ')); end
end
Da = D; Da.out = out; Da.states = st;
end
