function [rc, rv] = ca_fits(S, pc, pv, ve, branch)
%CA_FITS  The CVAR (branch 'differenced': equal weights in changes; 'levels': estimated weights,
%   q = 12) and the VAR without controls (from the differenced fit with pv lags).
w = ones(size(S.C, 2), 1) / size(S.C, 2);
if strcmp(branch, 'levels')
    rc = cvar.fit(S.y, S.C, S.dose, struct('p', pc, 'q', 12, 'var_exog', ve));
else
    rc = cvar.fit(S.y, S.C, S.dose, struct('p', pc, 'var_exog', ve, 'weights', w, 'difference', true));
end
rv = cvar.fit(S.y, S.C, S.dose, struct('p', pv, 'var_exog', ve, 'weights', w, 'difference', true));
end
