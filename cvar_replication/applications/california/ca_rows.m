function rows = ca_rows(tag, est, extra, hor)
%CA_ROWS  Long rows (one per horizon) with both models' estimates, bands and width ratio.
rows = struct([]);
for h = hor
    r = extra; r.spec = string(tag); r.h = h;
    for nm = {'cvar', 'var'}
        e = est.(nm{1}); i = h + 1;
        r.(nm{1}) = e.pt(i); r.([nm{1}, '_lo']) = e.lo(i); r.([nm{1}, '_hi']) = e.hi(i);
        r.([nm{1}, '_hall_lo']) = e.hlo(i); r.([nm{1}, '_hall_hi']) = e.hhi(i);
        r.([nm{1}, '_lo68']) = e.lo68(i); r.([nm{1}, '_hi68']) = e.hi68(i);
    end
    r.width_ratio_boot = (r.cvar_hi - r.cvar_lo) / (r.var_hi - r.var_lo);
    r.cvar_sig_boot = r.cvar_lo > 0 || r.cvar_hi < 0; r.var_sig_boot = r.var_lo > 0 || r.var_hi < 0;
    r.cvar_sig68 = r.cvar_lo68 > 0 || r.cvar_hi68 < 0; r.var_sig68 = r.var_lo68 > 0 || r.var_hi68 < 0;
    if isempty(rows), rows = r; else, rows(end + 1) = r; end %#ok<AGROW>
end
end
