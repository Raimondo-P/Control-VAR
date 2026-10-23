function o = defaults(o)
%DEFAULTS  Fill the options of cvar.fit with their default values.
%   p            lag length of the VAR in (other policies, policy, cycles); 12 for monthly data
%   q            current value and q lags of every policy in the weights regression
%   L            leads and lags of the changes in the trend directions of the controls
%   constrained  non-negative weights
%   rank         'johansen', 'bic' or the number of cointegrating relations among the controls
%   rank_level   level of the trace test
%   rank_lags    'aic' or the number of lagged differences of the Johansen regression
%   breaks       [] , 'auto' or a vector of first observations (1-based) of new levels; with
%                several outcomes, a cell array with one entry per outcome
%   max_breaks, break_cv  for breaks = 'auto'
%   other        T x nr other policies, ordered before the policy ([] if absent)
%   var_exog     T x ne deterministic regressors of the VAR, held fixed in the bootstrap
%   weights      [] (estimate), a K x 1 vector of imposed weights, or a cell array per outcome
%                (a vector, or [] to estimate that outcome's weights)
%   difference   false, true or a logical vector per outcome: the outcome enters the VAR as the
%                change of its cycle and its responses are cumulated (needs imposed weights)
%   names, control_names, dates  labels
if nargin < 1 || isempty(o), o = struct(); end
d = struct('p', 12, 'q', 12, 'L', 1, 'constrained', true, 'rank', 'johansen', ...
    'rank_level', 0.05, 'rank_lags', 'aic', 'breaks', [], 'max_breaks', 2, 'break_cv', 8.85, ...
    'other', [], 'var_exog', [], 'weights', [], 'difference', false, 'names', {{}}, ...
    'control_names', {{}}, 'dates', []);
f = fieldnames(d);
for i = 1:numel(f)
    if ~isfield(o, f{i})
        o.(f{i}) = d.(f{i});
    end
end
end
