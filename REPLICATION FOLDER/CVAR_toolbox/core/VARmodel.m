function [VAR, VARopt,LR_trace_table] = VARmodel(Y_t,D_t,nlag,const,other,nlag_ex,ncoint,foldername)
%========================================================================
% Perform vector autogressive (VAR) estimation with OLS 
%========================================================================
% [VAR, VARopt] = VARmodel(ENDO,nlag,const,EXOG,nlag_ex)
% -----------------------------------------------------------------------
% INPUT
%	- ENDO: an (nobs x nvar) matrix of y-vectors
%	- nlag: lag length
% -----------------------------------------------------------------------
% OPTIONAL INPUT
%	- const: 0 no constant; 1 constant; 2 constant and trend; 3 constant 
%       and trend^2 [dflt = 0]
%	- EXOG: optional matrix of variables (nobs x nvar_ex)
%	- nlag_ex: number of lags for exogeonus variables [dflt = 0]
%	- COINT: optional matrix of cointegration variables (nobs x nvar_ex)
%	- nlag_coint: number of lags for cointegration variables [dflt = 0]
% -----------------------------------------------------------------------
% OUTPUT
%   - VAR: structure including VAR estimation results
%   - VARopt: structure including VAR options (see VARoption)
% -----------------------------------------------------------------------
% EXAMPLE
%   - See VARToolbox_Code.m in "../Primer/"
% =======================================================================
% VAR Toolbox 3.0
% Ambrogio Cesa-Bianchi
% ambrogiocesabianchi@gmail.com
% March 2012. Updated November 2020
% -----------------------------------------------------------------------


%% Check inputs
% -----------------------------------------------------------------------
[alpha,beta,GAMMA,Lam,SIGMAml,U,U2,LR_trace,LR_max]=VECMmlrestr(Y_t,2,ncoint,other);

crits_vals = [282.45; 236.54; 196.37; 159.48; 126.58; 97.18; 71.86; 49.65; 32.00; 17.85; 7.52];
if size(LR_trace,1)<=11
    critical_values = crits_vals(end-length(LR_trace)+1:end);
else
    critical_values = [crits_vals ; zeros(length(LR_trace)-11,1)];
end
labels = {'LR Trace Test', 'Critical Values'};
LR_trace_table = table(round(LR_trace,2), critical_values, 'VariableNames', labels);
output_filename = fullfile(foldername, 'LR_trace.xlsx');
sheet_name = 'Sheet1';
writetable(LR_trace_table, output_filename, 'Sheet', sheet_name);

labels = {'LR Trace Test', 'Critical Values'};
LR_max_table = table(round(LR_max',2), critical_values, 'VariableNames', labels);
output_filename = fullfile(foldername,'LR_max.xlsx');
sheet_name = 'Sheet2';
writetable(LR_max_table, output_filename, 'Sheet', sheet_name);


ENDO = U2'; 
ENDO(:,1) = D_t(3:end);
% ENDO = Ytot(2:end,:)
[nobs, nvar] = size(ENDO);
EXOG = other(3:end,:);

% Create VARopt and update it
VARopt = VARoption;
VAR.ENDO = ENDO;
VAR.nlag = nlag;

% Check if ther are constant, trend, both, or none
if ~exist('const','var')
    const = 1;
end
VAR.const = const;

% Check if there are exogenous variables 
if isempty(EXOG)~=1
    [nobs2, nvar_ex] = size(EXOG);
    % Check that ENDO and EXOG are conformable
    if (nobs2 ~= nobs)
        error('var: nobs in EXOG-matrix not the same as y-matrix');
    end
    clear nobs2
    % Check if there is lag order of EXOG, otherwise set it to 0
    if ~exist('nlag_ex','var')
        nlag_ex = 0;
    end
    VAR.EXOG = EXOG;
else
    nvar_ex = 0;
    nlag_ex = 0;
    VAR.EXOG = [];
end


%% Save some parameters and create data matrices
% -----------------------------------------------------------------------
    nobse         = nobs - max(nlag,nlag_ex);
    VAR.nobs      = nobse;
    VAR.nvar      = nvar;
    VAR.nvar_ex   = nvar_ex;    
    VAR.nlag      = nlag;
    VAR.nlag_ex   = nlag_ex;
    ncoeff        = nvar*nlag; 
    VAR.ncoeff    = ncoeff;
    ncoeff_ex     = nvar_ex*(nlag_ex+1);
    ntotcoeff     = ncoeff + ncoeff_ex + const;
    VAR.ntotcoeff = ntotcoeff;
    VAR.const     = const;

% Create independent vector and lagged dependent matrix
[Y, X] = VARmakexy(ENDO,nlag,const);

% Create (lagged) exogenous matrix
if nvar_ex>0
    X_EX  = VARmakelags(EXOG,nlag_ex);
    if nlag == nlag_ex
        X = [X X_EX];
    elseif nlag > nlag_ex
        diff = nlag - nlag_ex;
        X_EX = X_EX(diff+1:end,:);
        X = [X X_EX];
    elseif nlag < nlag_ex
        diff = nlag_ex - nlag;
        Y = Y(diff+1:end,:);
        X = [X(diff+1:end,:) X_EX];
    end
end

%% OLS estimation equation by equation
% -----------------------------------------------------------------------

for i = 1:size(ENDO,2)
    Yvec = ENDO(nlag+1:end,i);
    OLSout = OLSmodel(Yvec,X,0);
    aux = ['eq' num2str(i)];
    eval( ['VAR.' aux '.beta  = OLSout.beta;'] );  % bhats
    eval( ['VAR.' aux '.tstat = OLSout.tstat;'] ); % t-stats
    eval( ['VAR.' aux '.bstd  = OLSout.bstd;'] );  % beta std error
    % compute t-probs
    tstat = zeros(ncoeff,1);
    tstat = OLSout.tstat;
    tout = tdis_prb(tstat,nobse-ncoeff);
    eval( ['VAR.' aux '.tprob = tout;'] );        % t-probs
    eval( ['VAR.' aux '.resid = OLSout.resid;'] );% resids 
    eval( ['VAR.' aux '.yhat  = OLSout.yhat;'] ); % yhats
    eval( ['VAR.' aux '.y     = Yvec;'] );        % actual y
    eval( ['VAR.' aux '.rsqr  = OLSout.rsqr;'] ); % r-squared
    eval( ['VAR.' aux '.rbar  = OLSout.rbar;'] ); % r-adjusted
    eval( ['VAR.' aux '.sige  = OLSout.sige;'] ); % standard error
    eval( ['VAR.' aux '.dw    = OLSout.dw;'] );   % DW
end 

%% Compute the matrix of coefficients & VCV
% -----------------------------------------------------------------------
Ft = (X'*X)\(X'*Y);
VAR.Ft = Ft;
F = Ft';
VAR.F = Ft';
SIGMA = (1/(nobse-ntotcoeff))*(Y-X*Ft)'*(Y-X*Ft); % adjusted for # of estimated coeff per equation
VAR.sigma = SIGMA;
VAR.resid = Y - X*Ft;
VAR.X = X;
VAR.Y = Y;
if nvar_ex > 0
    VAR.X_EX = X_EX;
end


%% Companion matrix of F and max eigenvalue
% -----------------------------------------------------------------------
Fcomp = [F(:,1+const:nvar*nlag+const); eye(nvar*(nlag-1)) zeros(nvar*(nlag-1),nvar)];
VAR.Fcomp = Fcomp;
VAR.maxEig = max(abs(eig(Fcomp)));


%% Initialize other results
% -----------------------------------------------------------------------
VAR.B   = [];   % structural impact matrix (need identification: see VARir/VARvd/VARhd)
VAR.Biv = [];   % first columns of structural impact matrix (need "iv" identification: see VARir/VARvd/VARhd)
VAR.PSI = [];   % Wold multipliers (computed only with VARir/VARvd/VARhd)
VAR.Fp  = [];   % Recursive F by lag (useful to compute MA representation)
VAR.IV  = [];   % External instruments for identification
