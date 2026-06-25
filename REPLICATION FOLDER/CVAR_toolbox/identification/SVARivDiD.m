function[IRbar,VARopt,IRinf,IRsup,Z_t,statistics_values]=SVARivDiD(D_t,D_c,y_t,y_c,k,r,horizons,colnames)
%========================================================================
% Perform vector autogressive (VAR) estimation with OLS 
%========================================================================
% []=SVARCholDiD(D_t,y_t,y_c,Z_t,r,horizons,colnames)
% -----------------------------------------------------------------------
% INPUT
%	- D_t: exogenous event (T x 1)
%	- y_t: treated series ordered as Cholesky (T x n)
%	- y_c: control series ordered as Cholesky (T x n)
%	- k: starting periods of intervention (sigma x 1)
%	- r: lag length
%	- horizons: IRF horizons
%	- colnames: names of the columns
% -----------------------------------------------------------------------
% OUTPUT
% -----------------------------------------------------------------------
% =======================================================================
% Modification to Cesa-Bianchi
% Raimondo Pala
% raimondopala@gmail.com
% May 2023. 
% -----------------------------------------------------------------------

%% find cointegrating vector
T = length(D_t);
[zeta,p,Y_tilde,D_tilde,Z_t]=cointegration_relation(y_t,y_c,D_t,D_c,T,k,r);

%% Test cointegration
[statistics_values,accept_10]=test_trends_DiD(y_t,y_c,Y_tilde,Z_t)
ydata = [D_tilde Y_tilde];

%% run VAR by Ambrogio Cesa Bianchi
[VAR, VARopt] = VARmodel(ydata,r);
VARopt.vnames = colnames;
[TABLE, beta] = VARprint(VAR,VARopt,2);


%% COMPUTE IR AND VD
%==========================================================================
% Set options some options for IRF calculation
VARopt.nsteps = horizons;
VARopt.ident = 'short';
VARopt.FigSize = [26,12];
VARopt.IV = Z_t;
% Compute IRF
[IRF, VAR] = VARir(VAR,VARopt);
% Compute error bands
[IRinf,IRsup,IRmed,IRbar] = VARirband(VAR,VARopt);
% Plot
VARirplot(IRbar,VARopt,IRinf,IRsup);
end