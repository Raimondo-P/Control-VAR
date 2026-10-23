function T = eb_br_table(br, horizons, alpha)
%EB_BR_TABLE  Long table of point estimates and percentile bands at selected horizons (the
%   table() method of the Python bootstrap result).
if nargin < 3, alpha = 0.10; end
rows = struct([]);
for k = fieldnames(br.point)'
    pt = br.point.(k{1}); [lo, hi] = cvar.bands(br, k{1}, alpha);
    pt = reshape(pt, size(pt, 1), []); lo = reshape(lo, size(pt)); hi = reshape(hi, size(pt));
    for j = 1:size(pt, 2)
        if strcmp(k{1}, 'policy'), nm = "policy"; else, nm = string(br.names{j}); end
        for h = horizons
            r = struct('object', string(k{1}), 'outcome', nm, 'h', h, 'estimate', pt(h + 1, j), ...
                'lower', lo(h + 1, j), 'upper', hi(h + 1, j));
            if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
        end
    end
end
T = struct2table(rows);
end
