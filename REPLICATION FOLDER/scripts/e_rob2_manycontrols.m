clear
clc
restoredefaultpath

%% add paths
addpath('../CVAR_toolbox')
addpath('../CVAR_toolbox/core')
addpath('../CVAR_toolbox/diagnostics')
addpath('../CVAR_toolbox/identification')
addpath('../CVAR_toolbox/plotting')

addpath('../auxiliary data transformation')

%% load data 
load('../data/processed/GERMANY_DISASTERS.mat')
load('../data/processed/data_for_VAR.mat')

data_US = readtable('../data/raw/replication_data.xlsx');

cost_of_dis = table2array(data_US(199:432,6))./1000;
deaths_of_dis = table2array(data_US(199:432,7));

INDPRO_DATA = readtable('../data/raw/fredgraph.xlsx');

US_INDPRO  = table2array(INDPRO_DATA(931:1164,2));
GER_INDPRO = table2array(INDPRO_DATA(931:1164,3));
ITA_INDPRO = table2array(INDPRO_DATA(931:1164,4));
FRA_INDPRO = table2array(INDPRO_DATA(931:1164,5));
ESP_INDPRO = table2array(INDPRO_DATA(931:1164,6));

%% data setup
y_t = [ log(US_INDPRO) , mac_unc_US ];

y_c = [ log(GER_INDPRO), log(ITA_INDPRO), log(FRA_INDPRO), log(ESP_INDPRO), ...
        mac_unc_GER, mac_unc_IT, mac_unc_FR, mac_unc_ES ];

D_t = cost_of_dis;

%% discard moments in which germany is also perturbed
[a1,a2] = maxk(costs_GER,2);
D_c = zeros(length(y_t),1);
D_c(a2) = a1;

%% VAR parameters
o = 3;
r = 6;
T = length(D_t);

D_t = zeros(T,1);
D_t(cost_of_dis > quantile(cost_of_dis,.95)) = 1;

horizons = 20;

colnames = [{'Disasters '}, {'IP '},{'MU '},{'IP DEU'}, {'IP ITA'}, ...
            {'IP FRA'}, {'IP ESP'},{'MU DEU'}, {'MU ITA'}, ...
            {'MU FRA'}, {'MU ESP'}];

other = zeros(T,1);
other(140:157) = 1;

foldername = '../output/diagnostics';

%% estimation
[IRbarChol,VARoptChol,IRinfChol,IRsupChol,rsugg,LR_trace] = ...
    SVARCholDiD(D_t,y_t,y_c,r,horizons,colnames,other,foldername);

%% plot IRFs
IRbar_sum = IRbarChol;
IRinf_sum = IRinfChol;
IRsup_sum = IRsupChol;

VARoptChol.vnames_mod = [{'Disasters'}, ...
                         {'IP (Control VAR many controls)'}, ...
                         {'MU (Control VAR many controls)'}];

SwatheOpt = PlotSwatheOption;
SwatheOpt.marker = '*';
SwatheOpt.trans = 1;

figure()
tiledlayout(3,1,'TileSpacing','Compact','Padding','Compact')

nexttile
for ii = 1:3
    PlotSwathe(IRbarChol(:,ii,1), ...
        [IRinfChol(:,ii,1) IRsupChol(:,ii,1)],SwatheOpt); hold on;

    plot(1:20,IRbarChol(:,ii,1),'--k','LineWidth',2,'Marker',SwatheOpt.marker);
    plot(zeros(1,20),'--k','LineWidth',0.5);

    xlim([1 20]);

    if ii == 1
        title('Natural disasters (Dummy)');
        nexttile
    elseif ii == 2
        title('Industrial production (Control VAR with many controls)');
        nexttile
    elseif ii == 3
        title('Macroeconomic uncertainty (Control VAR with many controls)');
        xlabel('Month')
    end

    set(gca,'Layer','top');
end

if ~exist('../output/figures', 'dir')
    mkdir('../output/figures');
end
print('../output/figures/Figure_IRF_rob2_manycontrols.pdf', '-dpdf')
close