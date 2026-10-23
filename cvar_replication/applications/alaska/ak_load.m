function D = ak_load(cfg, use_extra)
%AK_LOAD  Outcome panels, the dividend and the oil change from the snapshot in
%   applications/data/alaska/raw (bls.csv, fred.csv) and applications/data/alaska/pfd_amounts.csv.
%   D.dates (monthly), D.out.(outcome) T x 7 (columns in the order of ak_constants().states),
%   D.W (the dividend paid in the month, thousands of 2024 dollars per person), D.doil
%   (100 x change in log WTI).
if nargin < 2, use_extra = true; end
K = ak_constants();
raw = fullfile(cfg.data, 'alaska', 'raw');
Bt = readtable(fullfile(raw, 'bls.csv'), 'VariableNamingRule', 'preserve');
Ft = readtable(fullfile(raw, 'fred.csv'), 'VariableNamingRule', 'preserve');
bd = month_start(Bt{:, 1}); fd = month_start(Ft{:, 1});
dates = union(bd, fd);
T = numel(dates);
[~, ib] = ismember(bd, dates); [~, iff] = ismember(fd, dates);
colB = @(id) align(Bt, id, ib, T);
out = struct();
for nm = {'epop', 'ur', 'lfpr'}
    M = zeros(T, numel(K.states));
    for s = 1:numel(K.states), M(:, s) = colB(ak_ids(K.states{s}, 'laus', K.laus.(nm{1}), K)); end
    out.(nm{1}) = M;
end
implied = zeros(T, numel(K.states));
for s = 1:numel(K.states)
    implied(:, s) = 100 * colB(ak_ids(K.states{s}, 'laus', K.laus.emp, K)) ./ colB(ak_ids(K.states{s}, 'laus', K.laus.pop, K));
end
miss = isnan(out.epop); out.epop(miss) = implied(miss);
fprintf('check: largest gap between the LAUS ratio and employment/population x 100: %.3f\n', ...
    max(abs(out.epop(:) - implied(:)), [], 'omitnan'));
for nm = fieldnames(K.ces)'
    M = zeros(T, numel(K.states));
    for s = 1:numel(K.states), M(:, s) = 100 * log(colB(ak_ids(K.states{s}, 'ces', K.ces.(nm{1}), K))); end
    out.(nm{1}) = M;
end
for nm = fieldnames(out)'
    out.(nm{1}) = ak_fill_gaps(out.(nm{1}), dates, nm{1});
end
wti = nan(T, 1); cpi = nan(T, 1);
wti(iff) = Ft.WTISPLC; cpi(iff) = Ft.CPIAUCSL;
cpi = ak_fill_gaps(cpi, dates, 'CPI');
in24 = year(dates) == 2024 & ~isnan(cpi);
if any(in24), base = mean(cpi(in24)); else, base = cpi(find(~isnan(cpi), 1, 'last')); end
doil = [NaN; 100 * diff(log(wti))];
P = readtable(fullfile(cfg.data, 'alaska', 'pfd_amounts.csv'), 'CommentStyle', '#');
W = zeros(T, 1); W(~ismember(dates, fd)) = NaN;
lost = [];
for i = 1:height(P)
    t = datetime(P.year(i), P.pay_month(i), 1);
    k = find(dates == t, 1);
    if ~isempty(k) && ~isnan(cpi(k))
        amt = P.dividend(i) + use_extra * P.extra(i);
        W(k) = amt * base / cpi(k) / 1000;
    else
        lost(end + 1) = P.year(i); %#ok<AGROW>
    end
end
if ~isempty(lost)
    fprintf('  WARNING: dividend not placed (month outside the data or CPI missing) for %s\n', mat2str(lost));
end
D = struct('dates', dates, 'out', out, 'W', W, 'doil', doil, 'states', {K.states});
fprintf('dividend months: %d, from %s to %s\n', sum(W > 0), datestr(min(dates(W > 0)), 'yyyy-mm'), ...
    datestr(max(dates(W > 0)), 'yyyy-mm'));
end

function d = month_start(x)
if ~isdatetime(x), x = datetime(x); end
d = dateshift(x, 'start', 'month');
d.Format = 'yyyy-MM-dd';
end

function v = align(Bt, id, ib, T)
v = nan(T, 1);
if ismember(id, Bt.Properties.VariableNames)
    v(ib) = Bt.(id);
end
end
