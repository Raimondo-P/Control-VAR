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

ind = isnan(Um) == 0;

%% prepare
ydata = [noaa_cost(ind) ip(ind) Um(ind)];
r = 6;
horizons = 20;

colnames = [{'Disasters (LMN)'}, ...
            {'IP (LMN)'}, ...
            {'MU (LMN)'}];

z = mwdetrend(ydata(:,2),r);
X(:,2) = z.chat;

z = mwdetrend(ydata(:,1),r);
X(:,1) = z.chat;

z = mwdetrend(ydata(:,3),r);
X(:,3) = z.chat;

tb_stat_LMN = [adftest(X(:,1)) adftest(X(:,2)) adftest(X(:,3))];

%% VAR detrended
[VAR, VARopt] = VARmodel_O(X,r);
VARopt.vnames = colnames;

[TABLE, beta] = VARprint(VAR,VARopt,2);

VARopt.nsteps = horizons;
VARopt.ident = 'short';
VARopt.FigSize = [26,12];

[IRF, VAR] = VARir(VAR,VARopt);

[IRinf_LMN_detrended,IRsup_LMN_detrended,IRmed_LMN_detrended,IRbar_LMN_detrended] = VARirband(VAR,VARopt);

VARirplot(IRbar_LMN_detrended,VARopt,IRinf_LMN_detrended,IRsup_LMN_detrended);

VARopt_MLN_detrended = VARopt;

% print('../output/figures/Figure_IRF_detrended.pdf', '-dpdf')
% close

%% VAR trend
[VAR, VARopt] = VARmodel_O(ydata,r);
VARopt.vnames = colnames;

[TABLE, beta] = VARprint(VAR,VARopt,2);

VARopt.nsteps = horizons;
VARopt.ident = 'short';
VARopt.FigSize = [26,12];

[IRF, VAR] = VARir(VAR,VARopt);

[IRinf_LMN_trend,IRsup_LMN_trend,IRmed_LMN,IRbar_LMN_trend] = VARirband(VAR,VARopt);

VARopt_MLN_trend = VARopt;

VARopt_MLN_detrended.vnames_mod = [{'Disasters'}, ...
                                  {'IP (Filtered)'}, ...
                                  {'MU (Filtered)'}];

SwatheOpt = PlotSwatheOption;
SwatheOpt.marker = '*';
SwatheOpt.trans = 1;

tiledlayout(3,1,'TileSpacing','Compact','Padding','Compact')

nexttile
for ii = 1:3
    PlotSwathe(IRbar_LMN_detrended(:,ii,1), ...
        [IRinf_LMN_detrended(:,ii,1) IRsup_LMN_detrended(:,ii,1)],SwatheOpt); hold on;
    plot(1:20,IRbar_LMN_detrended(:,ii,1),'--k','LineWidth',2,'Marker',SwatheOpt.marker);
    plot(zeros(1,20),'--k','LineWidth',0.5);
    xlim([1 20]);

    if ii == 1
        title('Natural disasters (Original)');
        nexttile
    elseif ii == 2
        title('Industrial production (Original)');
        nexttile
    elseif ii == 3
        title('Macroeconomic uncertainty (Original)');
        xlabel('Month')
    end

    set(gca,'Layer','top');
end

print('../output/figures/Figure_IRF_detrended.pdf', '-dpdf')
close

VARopt_MLN_trend.vnames_mod = [{'Disasters'}, ...
                              {'IP (Filtered)'}, ...
                              {'MU (Filtered)'}];

tiledlayout(3,1,'TileSpacing','Compact','Padding','Compact')

nexttile
for ii = 1:3
    PlotSwathe(IRbar_LMN_trend(:,ii,1), ...
        [IRinf_LMN_trend(:,ii,1) IRsup_LMN_trend(:,ii,1)],SwatheOpt); hold on;
    plot(1:20,IRbar_LMN_trend(:,ii,1),'--k','LineWidth',2,'Marker',SwatheOpt.marker);
    plot(zeros(1,20),'--k','LineWidth',0.5);
    xlim([1 20]);

    if ii == 1
        title('Natural disasters (Unfiltered)');
        nexttile
    elseif ii == 2
        title('Industrial production (Unfiltered)');
        nexttile
    elseif ii == 3
        title('Macroeconomic uncertainty (Unfiltered)');
        xlabel('Month')
    end

    set(gca,'Layer','top');
end

if ~exist('../output/figures', 'dir')
    mkdir('../output/figures');
end
print('../output/figures/Figure_IRF_trend.pdf', '-dpdf')
close