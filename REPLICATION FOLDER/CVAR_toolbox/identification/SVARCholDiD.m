function[IRbar,VARopt,IRinf,IRsup,rsugg,LR_trace]=SVARCholDiD(D_t,y_t,y_c,r,horizons,colnames,other,foldername)
%========================================================================
% Perform vector autogressive (VAR) estimation with OLS 
%========================================================================
% []=SVARCholDiD(D_t,y_t,y_c,Z_t,r,horizons,colnames)
% -----------------------------------------------------------------------
% INPUT
%	- D_t: exogenous event (Tx1)
%	- y_t: treated series ordered as Cholesky (Txn)
%	- y_c: control series ordered as Cholesky (Txn)
%	- r: lag length
%	- horizons: IRF horizons
%	- colnames: names of the columns
%   - k: initial treatment periods (sx1)
% -----------------------------------------------------------------------
% OUTPUT
% -----------------------------------------------------------------------
% =======================================================================
% Modification to Cesa-Bianchi
% Raimondo Pala
% raimondopala@gmail.com
% May 2023. 
% -----------------------------------------------------------------------

% Y_t = [D_t(2:end,:) y_t(2:end,:)-y_t(1:end-1,:) y_c(2:end,:)-y_c(1:end-1,:)]; 
Y_t = [D_t y_t y_c];
[rsugg]=check_lags(Y_t,r,foldername,other);
ncoint = size(Y_t,2)-1;
const = 1;
%% run VAR by Ambrogio Cesa Bianchi
nlag = rsugg; 
[VAR, VARopt,LR_trace] = VARmodel(Y_t,D_t,rsugg,const,other,nlag,ncoint,foldername);
VARopt.vnames = colnames;
if isempty(other)~=1
    for i = 1:size(other, 2)
        VARopt.vnames_ex{i} = ['Name_' num2str(i)];
    end
end
% [TABLE, beta] = VARprint(VAR,VARopt,2);


%% COMPUTE IR AND VD
%==========================================================================
% Set options some options for IRF calculation
VARopt.nsteps = horizons;
VARopt.ident = 'short';
VARopt.FigSize = [26,12];
VARopt.pctg = 95;
% Compute IRF
[IRF, VAR] = VARir(VAR,VARopt);
% Compute error bands
[IRinf,IRsup,IRmed,IRbar] = VARirband(VAR,VARopt);
% Plot
VARirplot(IRbar,VARopt,IRinf,IRsup);
end