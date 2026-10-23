function r = eb_fit_diff(E, s, p)
%EB_FIT_DIFF  Differenced branch for state s: equal imposed weights on its controls, cycle in
%   changes, EUC ordered first, p lags.
cc = eb_controls_of(E, s);
r = cvar.fit(eb_col(E, E.UR, s), eb_col(E, E.UR, cc), eb_col(E, E.EB, s), struct('p', p, ...
    'other', eb_col(E, E.EUC, s), 'weights', ones(numel(cc), 1) / numel(cc), 'difference', true, 'names', {{s}}));
end
