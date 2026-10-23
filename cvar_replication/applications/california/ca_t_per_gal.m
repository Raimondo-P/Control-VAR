function t = ca_t_per_gal(yr, K)
%CA_T_PER_GAL  LCFS deficit per retail gallon of E10 implied by CARB's formula with the year's benchmark.
y = min(max(yr, 2014), 2026);
t = max(K.ci_carbob(y) - K.ci_std(y), 0) * K.ed_carbob * K.share_carbob / 1e6;
end
