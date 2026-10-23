function r = ak_fit_changes(A, w, p)
%AK_FIT_CHANGES  CVAR in changes with imposed weights w: VAR in (oil, W, change in the cycle).
r = cvar.fit(A.y, A.C, A.W, struct('p', p, 'other', A.doil, 'weights', w(:), 'difference', true, ...
    'names', {{'AK'}}));
end
