function b = ak_wnnls(dy, X, F)
%AK_WNNLS  Non-negative weights in changes, after partialling out the policy terms.
P = pinv(F);
b = lsqnonneg(X - F * (P * X), dy - F * (P * dy));
end
