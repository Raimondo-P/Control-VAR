function s = norm_sf(z)
%NORM_SF  Survival function of the standard normal distribution.
s = 0.5 * erfc(z / sqrt(2));
end
