function run_eb(cfg)
%RUN_EB  The extended benefits application in order.
if nargin < 1, cfg = config(); end
eb_unit_roots(cfg); eb_design_tests(cfg); eb_lags(cfg); eb_states(cfg); eb_average(cfg);
eb_exhibits(cfg); eb_lpdid(cfg);
end
