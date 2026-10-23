function run_alaska(cfg)
%RUN_ALASKA  The whole Alaska application in order: screening, main estimates, checks, diagnostic,
%   exhibits. The draws come from config.m.
if nargin < 1, cfg = config(); end
ak_screen(cfg);
ak_main(cfg);
ak_checks(cfg);
ak_diagnostic(cfg);
ak_exhibits(cfg);
end
