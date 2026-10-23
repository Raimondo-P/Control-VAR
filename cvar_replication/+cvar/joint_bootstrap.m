function [per, avg] = joint_bootstrap(results, B, H, seed, no_controls)
%JOINT_BOOTSTRAP  Several treated units that share controls: every draw uses the same residual
%   rows (time periods) for all units, so the bands of the average across units reflect the
%   dependence across units. results is a cell array of cvar.fit outputs with the same sample and
%   lag length. Returns the per-unit bootstrap results and the result for the average.
if nargin < 2 || isempty(B), B = 999; end
if nargin < 3 || isempty(H), H = 36; end
if nargin < 4, seed = []; end
if nargin < 5 || isempty(no_controls), no_controls = true; end
n = check(results);
s = cvar.rng_stream(seed);
idx = randi(s, n, B, n);
per = cell(size(results));
for i = 1:numel(results)
    per{i} = cvar.bootstrap(results{i}, B, H, struct('no_controls', no_controls, 'indices', idx, ...
        'scheme', 'conditional'));
end
keys = fieldnames(per{1}.point);
for k = 1:numel(keys)
    pk = 0; dk = 0;
    for i = 1:numel(per)
        pk = pk + mean(reshape(per{i}.point.(keys{k}), H + 1, []), 2);
        dk = dk + mean(reshape(per{i}.draws.(keys{k}), B, H + 1, []), 3);
    end
    pt.(keys{k}) = pk / numel(per);
    dr.(keys{k}) = dk / numel(per);
end
avg = struct('point', pt, 'draws', dr, 'names', {{'Average'}}, 'scheme', 'conditional', 'population', []);
end

function n = check(results)
ns = cellfun(@(r) size(r.U, 1), results);
ps = cellfun(@(r) r.opts.p, results);
if numel(unique(ns)) ~= 1 || numel(unique(ps)) ~= 1
    error('the joint bootstrap requires the same sample and lag length for every unit');
end
n = ns(1);
end
