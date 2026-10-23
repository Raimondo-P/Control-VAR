function [D, first] = ca_load(cfg, require_cec)
%CA_LOAD  Monthly panel of retail gasoline prices (USD per gallon, monthly means of the EIA weekly
%   series in applications/data/california/ca_raw), the LCFS credit price
%   (lcfs_credit_transfers.xlsx, sheet Monthly), and the LCFS cost per gallon in cents: cost_const
%   (constant factor), cost_tv (CARB formula), cost_cec (refiners, CEC: cec_lcfs_monthly.csv) and
%   cost_spliced (formula before the first CEC month, CEC from then on). D is a timetable-like
%   struct with field dates; first is the first CEC month.
if nargin < 2, require_cec = true; end
K = ca_constants();
ca = fullfile(cfg.data, 'california');
S = struct();
for g = K.gas
    T = readtable(fullfile(ca, 'ca_raw', [g{1}, '.csv']), 'VariableNamingRule', 'preserve');
    d = dateshift(datetime(T{:, 1}), 'start', 'month'); v = T{:, 2};
    ok = ~isnan(v); d = d(ok); v = v(ok);
    [u, ~, ig] = unique(d);
    S.(g{1}) = {u, accumarray(ig, v, [], @mean)};
end
X = readtable(fullfile(ca, 'lcfs_credit_transfers.xlsx'), 'Sheet', 'Monthly', 'VariableNamingRule', 'preserve');
m = X.month;
if isnumeric(m), m = datetime(m, 'ConvertFrom', 'excel'); else, m = datetime(m); end
m = dateshift(m, 'start', 'month'); cp = X.avg_price_usd_per_credit;
if ~isnumeric(cp), cp = str2double(string(cp)); end
okc = ~isnan(cp) & ~isnat(m); m = m(okc); cp = cp(okc);       % groupby().last() skips missing values
[u, ~, ig] = unique(m);
last = accumarray(ig, (1:numel(cp))', [], @max);
cr = cp(last);
cd = (u(1):calmonths(1):u(end))';
crm = nan(numel(cd), 1); [~, j] = ismember(u, cd); crm(j) = cr;
crm = interp_pandas(crm);
% join on the months where every series is available
dates = cd;
for g = K.gas, dates = intersect(dates, S.(g{1}){1}); end
D = struct('dates', dates);
for g = K.gas
    [~, j] = ismember(dates, S.(g{1}){1}); D.(g{1}) = S.(g{1}){2}(j);
end
[~, j] = ismember(dates, cd); D.credit = crm(j);
ok = ~isnan(D.credit) & all(cell2mat(cellfun(@(g) ~isnan(D.(g)), K.gas, 'UniformOutput', false)), 2);
for f = fieldnames(D)', D.(f{1}) = D.(f{1})(ok); end
D.cost_const = 100 * K.const_t_per_gal * D.credit;
D.cost_tv = 100 * D.credit .* arrayfun(@(y) ca_t_per_gal(y, K), year(D.dates));
f = fullfile(ca, 'cec_lcfs_monthly.csv');
first = NaT;
if ~exist(f, 'file')
    if require_cec, error('applications/data/california/cec_lcfs_monthly.csv not found'); end
    return
end
C = readtable(f);
cdts = dateshift(datetime(C.date), 'start', 'month');
first = min(cdts);
D.cost_cec = nan(numel(D.dates), 1);
[in, j] = ismember(D.dates, cdts); D.cost_cec(in) = C.lcfs_cents(j(in));
D.cost_spliced = D.cost_cec; pre = D.dates < first; D.cost_spliced(pre) = D.cost_tv(pre);
end

function x = interp_pandas(x)
% pandas' Series.interpolate() (linear, forward): interior gaps linear, trailing gaps take the last
% value, leading gaps stay missing.
ok = find(~isnan(x));
if numel(ok) < 2, return, end
t = (1:numel(x))';
in = t > ok(1) & t < ok(end) & isnan(x);
x(in) = interp1(ok, x(ok), t(in), 'linear');
x(t > ok(end)) = x(ok(end));
end
