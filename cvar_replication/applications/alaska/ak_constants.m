function K = ak_constants()
%AK_CONSTANTS  Design of the Alaska application, fixed in advance (Jones and Marinescu 2022,
%   Kueng 2018): states, control sets, BLS series, outcomes and samples.
K.states = {'AK', 'WA', 'UT', 'WY', 'NV', 'MT', 'MN'};
K.fips   = containers.Map({'AK', 'WA', 'UT', 'WY', 'NV', 'MT', 'MN'}, {'02', '53', '49', '56', '32', '30', '27'});
K.sets   = struct('wa', {{'WA'}}, 'jm_emp', {{'UT', 'WY', 'WA', 'NV', 'MT', 'MN'}}, 'jm_pt', {{'NV', 'WY'}});
K.laus   = struct('ur', '03', 'emp', '05', 'epop', '07', 'lfpr', '08', 'pop', '09');
K.ces    = struct('nonfarm', '00000000', 'retail', '42000000', 'leisure', '70000000');
K.outcomes = {'epop', 'ur', 'lfpr', 'nonfarm', 'retail', 'leisure'};
% sample name and last month (NaT: to the end of the data)
K.samples_screen = {'full', NaT; 'pre2020', datetime(2019, 12, 1); 'pre2016', datetime(2015, 12, 1)};
K.samples_main   = {'full', NaT; 'pre2020', datetime(2019, 12, 1); 'pre2016', datetime(2015, 12, 1); ...
                    'pre2008', datetime(2007, 12, 1)};
K.q = 12;                 % current value and lags of the policies in the weights regressions
K.horizons = [0 1 2 3 6 12 24 36];
end
