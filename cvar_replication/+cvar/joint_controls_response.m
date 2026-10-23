function [per, avg] = joint_controls_response(results, B, H, seed)
%JOINT_CONTROLS_RESPONSE  Proposition 1 for every unit with common residual draws, and the
%   average across units.
if nargin < 2 || isempty(B), B = 499; end
if nargin < 3 || isempty(H), H = 36; end
if nargin < 4, seed = []; end
ns = cellfun(@(r) size(r.cr.U, 1), results);
if numel(unique(ns)) ~= 1, error('same sample required for every unit'); end
s = cvar.rng_stream(seed);
idx = randi(s, ns(1), B, ns(1));
per = cell(size(results));
for i = 1:numel(results)
    per{i} = cvar.controls_response(results{i}, H, B, [], idx);
end
keys = fieldnames(per{1}.point);
for k = 1:numel(keys)
    pk = 0; dk = 0;
    for i = 1:numel(per)
        pk = pk + mean(per{i}.point.(keys{k}), 2);
        dk = dk + mean(per{i}.draws.(keys{k}), 3);
    end
    pt.(keys{k}) = pk / numel(per);
    dr.(keys{k}) = dk / numel(per);
end
avg = struct('point', pt, 'draws', dr, 'names', {{'Average'}});
end
