function [s, se, p_sum1, p_equal] = ak_loading_tests(b, V)
%AK_LOADING_TESTS  Sum of the weights with its standard error, the p-value of sum(b) = 1 and the
%   p-value of b = 1/K (Wald).
K = numel(b);
one = ones(K, 1);
s = sum(b); se = sqrt(one' * V * one);
d = b - 1 / K;
Wd = d' * pinv(V) * d;
p_sum1 = 2 * cvar.norm_sf(abs((s - 1) / se));
p_equal = cvar.chi2_sf(Wd, K);
end
