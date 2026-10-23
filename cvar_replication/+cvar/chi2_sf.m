function s = chi2_sf(x, k)
%CHI2_SF  Survival function of the chi-square distribution with k degrees of freedom.
s = gammainc(x / 2, k / 2, 'upper');
end
