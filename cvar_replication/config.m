function cfg = config()
%CONFIG  Settings of the replication package: numbers of draws, seeds and folders. Raise the
%   numbers of draws here; every script reads them from this file.
root = fileparts(mfilename('fullpath'));
cfg = struct();
cfg.root    = root;
cfg.data    = fullfile(root, 'applications', 'data');
cfg.results = fullfile(root, 'results');
cfg.paper   = fullfile(root, 'results', 'paper');      % tables and figures read by the LaTeX files
% applications
cfg.B_app   = 999;     % bootstrap draws of the responses
cfg.B_test  = 999;     % bootstrap draws of the test of the design
cfg.B_cr    = 499;     % bootstrap draws of the response of the controls and the bounds
cfg.block   = 12;      % block length of the moving-block bootstrap of the weights in changes
cfg.H       = 36;      % largest horizon of the responses (months)
% simulations
cfg.sim_reps = 500;    % replications of the estimator and bootstrap simulations (sim1, lags, grid, joint)
cfg.sim_boot = 499;    % bootstrap draws in each of those replications
cfg.diff_reps = 500;   % replications of the simulation of the CVAR in changes
cfg.diff_boot = 199;   % bootstrap draws in each of those replications
cfg.test_reps = 2000;  % replications of the simulations of the test of the design
cfg.test_boot = 199;   % sieve-bootstrap draws in each of those replications
end
