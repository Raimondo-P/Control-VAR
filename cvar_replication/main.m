%MAIN  Runs the applications of the paper in order. The numbers of draws are in config.m.
%   Run from this folder. The simulations of the supplement are added in the next stage.
root = fileparts(mfilename('fullpath'));
addpath(root); addpath(genpath(fullfile(root, 'applications')));
cfg = config();
run_alaska(cfg);
run_eb(cfg);
run_california(cfg);
