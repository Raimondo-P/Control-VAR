function get_alaska_data(cfg)
%GET_ALASKA_DATA  Download the Alaska data and overwrite the snapshot in
%   applications/data/alaska/raw (bls.csv, bls_nsa.csv, bls_all_1.csv and fred.csv). The package ships the snapshot used in
%   the paper; run this only to refresh it, since BLS revises LAUS and CES every year.
%   Set the environment variable BLS_API_KEY for the faster limits of the BLS API v2.
if nargin < 1, cfg = config(); end
K = ak_constants();
raw = fullfile(cfg.data, 'alaska', 'raw');
if ~exist(raw, 'dir'), mkdir(raw); end
ids = {};
for s = K.states
    for f = fieldnames(K.laus)', ids{end + 1} = ak_ids(s{1}, 'laus', K.laus.(f{1}), K); end %#ok<AGROW>
end
for s = K.states
    for f = fieldnames(K.ces)', ids{end + 1} = ak_ids(s{1}, 'ces', K.ces.(f{1}), K); end %#ok<AGROW>
end
fprintf('downloading %d BLS series\n', numel(ids));
Tb = bls_fetch(ids, 1976, year(datetime('today')));
writetable(Tb, fullfile(raw, 'bls.csv'));
ids_n = {};
for s = K.states
    for f = {'ur', 'epop', 'lfpr'}, ids_n{end + 1} = sprintf('LAUST%s00000000000%s', K.fips(s{1}), K.laus.(f{1})); end %#ok<AGROW>
end
for s = K.states
    for f = fieldnames(K.ces)', ids_n{end + 1} = sprintf('SMU%s00000%s01', K.fips(s{1}), K.ces.(f{1})); end %#ok<AGROW>
end
fprintf('downloading %d unadjusted BLS series\n', numel(ids_n));
writetable(bls_fetch(ids_n, 1976, year(datetime('today'))), fullfile(raw, 'bls_nsa.csv'));
[sa, fa] = ak_states_all();
ids_a = {};
for s = sa
    for f = {'epop', 'ur', 'lfpr'}, ids_a{end + 1} = sprintf('LASST%s00000000000%s', fa(s{1}), K.laus.(f{1})); end %#ok<AGROW>
end
for s = sa
    for f = fieldnames(K.ces)', ids_a{end + 1} = sprintf('SMS%s00000%s01', fa(s{1}), K.ces.(f{1})); end %#ok<AGROW>
end
fprintf('downloading %d BLS series for all states (donor pools)\n', numel(ids_a));
old_all = dir(fullfile(raw, 'bls_all_*.csv'));
Ta = bls_fetch(ids_a, 1976, year(datetime('today')));
for k = 1:numel(old_all), delete(fullfile(old_all(k).folder, old_all(k).name)); end
writetable(Ta, fullfile(raw, 'bls_all_1.csv'));
fprintf('downloading FRED: WTISPLC, CPIAUCSL\n');
F1 = fred('WTISPLC'); F2 = fred('CPIAUCSL');
Tf = outerjoin(F1, F2, 'Keys', 'date', 'MergeKeys', true);
writetable(Tf, fullfile(raw, 'fred.csv'));
fprintf('written to %s\n', raw);
end

function Tb = bls_fetch(ids, y0, y1)
% BLS public API v2, in chunks of series and of years.
key = getenv('BLS_API_KEY');
if isempty(key), per = 25; span = 10; else, per = 50; span = 20; end
opts = weboptions('MediaType', 'application/json', 'Timeout', 180);
dates = datetime.empty(0, 1); sid = {}; val = [];
for i = 1:per:numel(ids)
    chunk = ids(i:min(i + per - 1, numel(ids)));
    for a = y0:span:y1
        b = min(a + span - 1, y1);
        payload = struct('seriesid', {chunk}, 'startyear', num2str(a), 'endyear', num2str(b));
        if ~isempty(key), payload.registrationkey = key; end
        res = webwrite('https://api.bls.gov/publicAPI/v2/timeseries/data/', payload, opts);
        if ~strcmp(res.status, 'REQUEST_SUCCEEDED')
            error('BLS %d-%d: %s. Run again later; nothing was written.', a, b, res.status);
        end
        S = res.Results.series;
        for k = 1:numel(S)
            dd = S(k).data;
            if iscell(dd), dd = [dd{:}]; end
            for r = 1:numel(dd)
                pr = dd(r).period;
                if startsWith(pr, 'M') && ~strcmp(pr, 'M13')
                    dates(end + 1, 1) = datetime(str2double(dd(r).year), str2double(pr(2:end)), 1); %#ok<AGROW>
                    sid{end + 1, 1} = S(k).seriesID; %#ok<AGROW>
                    val(end + 1, 1) = str2double(dd(r).value); %#ok<AGROW>
                end
            end
        end
        pause(1);
    end
end
L = table(dates, sid, val, 'VariableNames', {'date', 'id', 'value'});
Tb = unstack(L, 'value', 'id', 'AggregationFunction', @(x) x(end), 'VariableNamingRule', 'preserve');
Tb = sortrows(Tb, 'date');
% every requested series must cover every year, or the snapshot would be incomplete
yrs = unique(year(Tb.date));
if any(~ismember(y0:y1 - 1, yrs))
    error('the BLS download is missing some years; run again later, nothing was written');
end
end

function T = fred(sid)
fn = [tempname, '.csv'];
websave(fn, ['https://fred.stlouisfed.org/graph/fredgraph.csv?id=', sid]);
X = readtable(fn, 'TreatAsMissing', '.', 'VariableNamingRule', 'preserve');
delete(fn);
T = table(X{:, 1}, X{:, 2}, 'VariableNames', {'date', sid});
end
