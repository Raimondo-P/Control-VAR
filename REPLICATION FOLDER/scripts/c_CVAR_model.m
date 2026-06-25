clear
clc
restoredefaultpath
%% add paths
addpath('../CVAR_toolbox')
addpath('../CVAR_toolbox/core')
addpath('../CVAR_toolbox/diagnostics')
addpath('../CVAR_toolbox/identification')
addpath('../CVAR_toolbox/plotting')
%% load data 
load('../data/processed/GERMANY_DISASTERS.mat')
load('../data/processed/data_for_VAR.mat')
data_US = xlsread('../data/raw/replication_data.xlsx');
cost_of_dis = data_US(199:432,6)./1000; % mac 5, windows 6
deaths_of_dis = data_US(199:432,7); % mac 6, windows 7
INDPRO_DATA = readtable('../data/raw/fredgraph.xlsx');
US_INDPRO = table2array(INDPRO_DATA(931:1164,2)); % mac 1, windows 2
GER_INDPRO = table2array(INDPRO_DATA(931:1164,3)); % mac 2, windows 3
%%
y_t = [ log(US_INDPRO) , (mac_unc_US) ];
y_c = [ log(GER_INDPRO) , (mac_unc_GER) ];
D_t = cost_of_dis;
%% discard moments in which germany is also perturbated by natural disasters
[a1,a2] = maxk(costs_GER,2);
D_c = zeros(length(y_t),1);
D_c(a2) = a1;
%% VAR paramters set
o = 3;
r = 6;
T = length(D_t);
D_t = zeros(T,1);
D_t(cost_of_dis>quantile(cost_of_dis,.95),:) = 1;

horizons = 20;
colnames = [{'Disasters '}, {'IP '},{'MU '},{'IP DEU'},{'MU DEU'}];
other = zeros(T,1); %breaks in y_t(:,1)
other(140:157) = 1; %great financial crisis dummy
foldername = '../output/diagnostics';
%%      
[IRbarChol,VARoptChol,IRinfChol,IRsupChol,rsugg,LR_trace]=SVARCholDiD(D_t,y_t,y_c,r,horizons,colnames,other,foldername);
%% figure with just the IRF
IRbar_sum = IRbarChol;
IRinf_sum = IRinfChol;
IRsup_sum = IRsupChol;
VARoptChol.vnames_mod = [{'Disasters'}, ... 
                            {'IP (Coint)'}, ...
                            {'MU (Coint)'}];
                 
VARirplot_2(IRbar_sum,VARoptChol,IRinf_sum,IRsup_sum)

if ~exist('../output/figures', 'dir')
    mkdir('../output/figures');
end
print('../output/figures/Figure_IRF_CVAR.pdf', '-dpdf')
close