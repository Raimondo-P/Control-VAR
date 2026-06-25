clear
clc

addpath('../CVAR_toolbox')
addpath('../CVAR_toolbox/core')
addpath('../CVAR_toolbox/diagnostics')
addpath('../CVAR_toolbox/identification')
addpath('../CVAR_toolbox/plotting')

U_data = importdata('../data/raw/replication_data.xlsx');
U_data = U_data.data.Uncertainty;

%% US 
dates = 1980 + 1/12 : 1/12 : 2020+02/12;
U_data = U_data(1:end-2,:);

Uf = U_data(:,2);
Um = U_data(:,3);
ip = U_data(:,4);
boarding_plane = (U_data(:,5));

noaa_cost = U_data(:,6)/1000;
noaa_death = U_data(:,7)/1000;

%% rework noaa cost
noaa_cost_quant = quantile(noaa_cost,.95);
noaa_cost_dummy = zeros(482,1);
noaa_cost_dummy(noaa_cost >= noaa_cost_quant) = 1;

ind = isnan(Um) == 0;

%% prepare
ydata = [noaa_cost_dummy(ind) ip(ind) Um(ind)];
r = 6;
horizons = 20;

colnames = [{'Disasters dummy'}, ...
            {'IP'}, ...
            {'MU'}];

z = mwdetrend(ydata(:,2),r);
X(:,2) = z.chat;

z = mwdetrend(ydata(:,1),r);
X(:,1) = noaa_cost_dummy;

z = mwdetrend(ydata(:,3),r);
X(:,3) = z.chat;

tb_stat_LMN = [adftest(X(:,1)) adftest(X(:,2)) adftest(X(:,3))];

%% VAR
[VAR, VARopt_MLN_dummy] = VARmodel_O(X,r);
VARopt_MLN_dummy.vnames = colnames;

[TABLE, beta] = VARprint(VAR,VARopt_MLN_dummy,2);

VARopt_MLN_dummy.nsteps = horizons;
VARopt_MLN_dummy.ident = 'short';
VARopt_MLN_dummy.FigSize = [26,12];

[IRF, VAR] = VARir(VAR,VARopt_MLN_dummy);

[IRinf_LMN_dummy,IRsup_LMN_dummy,IRmed_LMN_dummy,IRbar_LMN_dummy] = VARirband(VAR,VARopt_MLN_dummy);

SwatheOpt = PlotSwatheOption;
SwatheOpt.marker = '*';
SwatheOpt.trans = 1;

figure()
tiledlayout(3,1,'TileSpacing','Compact','Padding','Compact')

nexttile
for ii = 1:3
    PlotSwathe(IRbar_LMN_dummy(:,ii,1), ...
        [IRinf_LMN_dummy(:,ii,1) IRsup_LMN_dummy(:,ii,1)],SwatheOpt); hold on;

    plot(1:20,IRbar_LMN_dummy(:,ii,1),'--k','LineWidth',2,'Marker',SwatheOpt.marker);
    plot(zeros(1,20),'--k','LineWidth',0.5);

    xlim([1 20]);

    if ii == 1
        title('Natural disasters (Dummy)');
        nexttile
    elseif ii == 2
        title('Industrial production');
        nexttile
    elseif ii == 3
        title('Macroeconomic uncertainty');
        xlabel('Month')
    end

    set(gca,'Layer','top');
end

if ~exist('../output/figures', 'dir')
    mkdir('../output/figures');
end
print('../output/figures/Figure_IRF_rob1_dummy.pdf', '-dpdf')
close