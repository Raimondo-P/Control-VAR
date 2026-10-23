function eb_average(cfg)
%EB_AVERAGE  Average response across the EB states with the joint bootstrap (the same months drawn
%   for every state), one common lag order. Requires eb_lags.
%   Outputs: results/eb/5_average.csv, 5_states_common_lag.csv
if nargin < 1, cfg = config(); end
E = eb_load(cfg); out = eb_out(cfg);
[p, sts] = eb_common_lag(E, out);
fprintf('common lag order: %d\n', p);
rs = cellfun(@(s) eb_fit_diff(E, s, p), sts, 'UniformOutput', false);
[per, avg] = cvar.joint_bootstrap(rs, cfg.B_app, cfg.H, 21);
ta = eb_br_table(avg, [0 6 12 24 36]); ta.p = repmat(p, height(ta), 1);
writetable(ta, fullfile(out, '5_average.csv'));
tp = cellfun(@(b) eb_br_table(b, [0 6 12 24 36]), per, 'UniformOutput', false);
writetable(vertcat(tp{:}), fullfile(out, '5_states_common_lag.csv'));
disp(ta);
end
