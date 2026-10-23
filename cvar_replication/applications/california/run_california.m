function run_california(cfg)
%RUN_CALIFORNIA  The California application in order.
if nargin < 1, cfg = config(); end
ca_cec_validation(cfg); ca_robustness(cfg); ca_a4(cfg); ca_exhibits(cfg);
end
