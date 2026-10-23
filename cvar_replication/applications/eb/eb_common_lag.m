function [p, sts] = eb_common_lag(E, out)
%EB_COMMON_LAG  One lag order for all states and both models: the larger of the median rule orders
%   of the CVAR and of the VAR (truncated to an integer, as int() in Python).
L = readtable(fullfile(out, '3_lags.csv'), 'TextType', 'string');
sts = E.elig(ismember(E.elig, cellstr(L.st)));
k = ismember(cellstr(L.st), sts);
p = floor(max(median(L.p_cvar(k)), median(L.p_var(k))));
end
