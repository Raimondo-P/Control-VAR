function M = ak_season_terms(dates, kind)
%AK_SEASON_TERMS  Deterministic seasonal terms for the weights regression and the VAR.
%   'fixed'  11 month dummies; 'trend' month dummies and month dummies times a linear trend;
%   'decade' month dummies separately in each decade, plus decade dummies; '' none.
if isempty(kind), M = []; return, end
mo = month(dates);
D = zeros(numel(dates), 11);
for k = 1:11, D(:, k) = double(mo == k); end
switch kind
    case 'fixed'
        M = D;
    case 'trend'
        t = linspace(0, 1, numel(dates))';
        M = [D, D .* t];
    case 'decade'
        dec = floor(year(dates) / 10);
        ds = unique(dec);
        M = zeros(numel(dates), 0);
        for d = ds', M = [M, D .* (dec == d)]; end %#ok<AGROW>
        for d = ds(2:end)', M = [M, double(dec == d)]; end %#ok<AGROW>
    otherwise
        error('unknown seasonal terms %s', kind);
end
end
