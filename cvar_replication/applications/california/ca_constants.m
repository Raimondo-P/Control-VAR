function K = ca_constants()
%CA_CONSTANTS  Design of the California application: CARB benchmarks (check against 17 CCR 95484),
%   energy density, blend share, constant factor, control sets and region names.
yrs = 2014:2026;
std = [97.96 97.96 96.50 95.02 93.55 93.23 91.98 90.74 89.50 88.25 87.01 85.77 84.52];
K.ci_std = containers.Map(num2cell(yrs), num2cell(std));
K.ci_carbob = containers.Map(num2cell(yrs), num2cell(99.78 * (yrs <= 2018) + 100.82 * (yrs > 2018)));
K.ed_carbob = 119.53;            % MJ per gallon of CARBOB
K.share_carbob = 0.90;           % E10
K.const_t_per_gal = 0.0018;
K.gas = {'gas_ca', 'gas_gulf', 'gas_r10', 'gas_r20', 'gas_r40', 'gas_tx', 'gas_fl'};
K.sets = struct('gulf', {{'gas_gulf'}}, 'padd1_4', {{'gas_r10', 'gas_r20', 'gas_gulf', 'gas_r40'}}, 'tx_fl', {{'gas_tx', 'gas_fl'}});
K.setlab = struct('gulf', 'Gulf Coast (PADD 3)', 'padd1_4', 'PADDs 1 to 4', 'tx_fl', 'Texas and Florida');
K.region = {'gas_r10', 'East Coast'; 'gas_r20', 'Midwest'; 'gas_gulf', 'Gulf Coast'; 'gas_r40', 'Rocky Mountain'};
K.H = 24;
K.horizons = [0 1 2 3 6 12 24];
end
