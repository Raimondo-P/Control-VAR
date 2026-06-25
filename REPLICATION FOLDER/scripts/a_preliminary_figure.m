%% Housekeeping
clear
clc

% =========================
% DATA PATHS
% =========================
data_US = xlsread('../data/raw/replication_data.xlsx');
load('../data/processed/GERMANY_DISASTERS.mat')

data_EU_uncertainty_indicators = xlsread('../data/raw/Uncertainty_indicators_MU1_MU2.xlsx');
data_EU_macro_indicators = xlsread('../data/raw/Input_Matlab_VAR_Data.xlsx');

INDPRO_DATA = readtable('../data/raw/fredgraph.xlsx');

% =========================
% VARIABLES
% =========================
cost_of_dis = data_US(199:432,6)./1000;
deaths_of_dis = data_US(199:432,7);

[Y,M] = meshgrid(1996:2015, 1:12);
ss = datenum([Y(:), M(:), ones(numel(Y),1)]);
ss(1:6)=[];
calend = datestr(ss);

dates = 1996 + 7/12 : 1/12 : 2015+12/12;

counter_ku = cost_of_dis;
for j = 1:6
    [~,ku(j)] = max(abs(counter_ku));
    counter_ku(ku(j)) = NaN;
    calend(ku(j),:)
end
ku = sort(ku);

%% US
fin_unc_US = data_US(199:432,2); % NOTE: if on windows move this to fin_unc_US = data_US(199:432,1)
mac_unc_US = data_US(199:432,3); % NOTE: if on windows move this to mac_unc_US = data_US(199:432,2)
ind_pro_US = table2array(INDPRO_DATA(932:1165,2));

%% EU uncertainty
mac_unc_GER = data_EU_uncertainty_indicators(:,2);
mac_unc_ES  = data_EU_uncertainty_indicators(:,6);
mac_unc_FR  = data_EU_uncertainty_indicators(:,10);
mac_unc_IT  = data_EU_uncertainty_indicators(:,14);

%% Germany
GER_INDPRO = table2array(INDPRO_DATA(932:1165,3));
ind_pro_control = GER_INDPRO;
mac_unc_control = mac_unc_GER;

%% EU macro extraction
EU_indicators = '../data/raw/Input_Matlab_VAR_Data.xlsx';
[~,sheet_names_EU] = xlsfinfo(EU_indicators);

sheet_names_to_extract = {'raw_DE_96', 'raw_ES_96', 'raw_FR_96', 'raw_IT_96'};
time_data = [];
ip_data = [];

for i = 1:length(sheet_names_EU)
    if any(strcmp(sheet_names_EU{i}, sheet_names_to_extract))
        [~,~,raw] = xlsread(EU_indicators, sheet_names_EU{i});
        data = raw(2:end,:);
        time = cell2mat(data(43:end,1));
        ip = cell2mat(data(43:end,5));

        time_data = [time_data time];
        ip_data = [ip_data ip];
    end
end

%% Create output folder (important)
if ~exist('../output/figures', 'dir')
    mkdir('../output/figures');
end

dates = 1996 + 7/12 : 1/12 : 2015+12/12;

%% FIGURE 1
figure(1)

subplot(3,1,1)
plot(dates,log(ind_pro_US),'k-','LineWidth',1.2)
hold on
plot(dates,log(ind_pro_control),'k--','LineWidth',1.2)
xlim([dates(1), dates(end)])
legend('US','DEU','Location','SouthEast')
title('Industrial Production')
set(gca,'Box','off')

subplot(3,1,2)
plot(dates, mac_unc_US,'k-','LineWidth',1.2)
hold on
plot(dates, mac_unc_control,'k--','LineWidth',1.2)
xlim([dates(1), dates(end)])
title('Macroeconomic Uncertainty')
set(gca,'Box','off')

subplot(3,1,3)
plot(dates,cost_of_dis,'k-','LineWidth',1.2)
hold on
xline(dates(63), '--k', {'9/11'});
xline(dates(99), '--k', {'Ivan'});
xline(dates(110), '--k', {'Katrina'});
xline(dates(147), '--k', {'Hanna'});
xxx = xline(dates(196), '--k', {'Sandy'});
xxx.LabelHorizontalAlignment = 'left';

xlim([dates(1), dates(end)])
title('Cost of Disasters')
set(gca,'Box','off')

%% FIGURE 2
figure(2)

plot(dates,cost_of_dis,'k-','LineWidth',1.2)
hold on
xline(dates(63), '--k', {'9/11'});
xline(dates(99), '--k', {'Ivan'});
xline(dates(110), '--k', {'Katrina'});
xline(dates(147), '--k', {'Hanna'});
xxx = xline(dates(196), '--k', {'Sandy'});
xxx.LabelHorizontalAlignment = 'left';

xlim([dates(1), dates(end)])
title('Cost of Disasters')
set(gca,'Box','off')


%% FIGURE 3
figure(3)

subplot(2,1,1)
plot(dates,log(ind_pro_US),'k-','LineWidth',1.2)
hold on
plot(dates,log(ind_pro_control),'k--','LineWidth',1.2)
xlim([dates(1), dates(end)])
legend('US','DEU','Location','SouthEast')
title('Industrial Production')
set(gca,'Box','off')

subplot(2,1,2)
plot(dates, mac_unc_US,'k-','LineWidth',1.2)
hold on
plot(dates, mac_unc_control,'k--','LineWidth',1.2)
xlim([dates(1), dates(end)])
title('Macroeconomic Uncertainty')
set(gca,'Box','off')

%% Save figures
if ~exist('../output/figures', 'dir')
    mkdir('../output/figures');
end

figure(1)
print('../output/figures/Figure_preliminary_1.pdf', '-dpdf')

figure(2)
print('../output/figures/Figure_preliminary_2.pdf', '-dpdf')

figure(3)
print('../output/figures/Figure_preliminary_3.pdf', '-dpdf')