function rows = ak_band_rows(base, spec, obj, pt, draws, hor)
%AK_BAND_ROWS  Rows with the estimate and the 90 and 68 percent percentile bands at the horizons hor.
q = cvar.quantile7(draws, [0.05 0.95 0.16 0.84]);
rows = struct([]);
for h = hor
    r = base;
    r.spec = string(spec); r.object = string(obj); r.h = h; r.estimate = pt(h + 1);
    r.lo90 = q(1, h + 1); r.hi90 = q(2, h + 1); r.lo68 = q(3, h + 1); r.hi68 = q(4, h + 1);
    r.zero_in_90 = r.lo90 <= 0 && 0 <= r.hi90; r.zero_in_68 = r.lo68 <= 0 && 0 <= r.hi68;
    if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
end
end
