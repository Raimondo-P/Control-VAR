function [VAR, VARopt, LR_trace] = SVARCholDiD_estimate(D_t, y_t, y_c, rsugg, colnames, foldername, other, ncoint, beta_fix, const, D_exog, include_D_lag)
%========================================================================
% SVARCholDiD_estimate — VAR/VECM estimation
%
%   D_exog = 1 : D_t is exogenous; gamma0 = OLS coeff on D_t in y_t eqn.
%   D_exog = 0 : [D_t, y_t] estimated jointly. D_t ordered first in
%                Cholesky so y_t shocks cannot move D_t on impact (DiD
%                exclusion). gamma0 = P(2,1)/P(1,1).
%
%   beta_fix   : impose cointegrating vector (e.g. [1;-1]); skip Johansen.
%   include_D_lag : add D_{t-1} as regressor (D_exog=1 only).
%========================================================================

if nargin < 7  || isempty(other);         other         = []; end
if nargin < 8  || isempty(ncoint);        ncoint        = []; end
if nargin < 9  || isempty(beta_fix);      beta_fix      = []; end
if nargin < 10 || isempty(const);         const         = 0;  end
if nargin < 11 || isempty(D_exog);        D_exog        = 1;  end
if nargin < 12 || isempty(include_D_lag); include_D_lag = 0;  end

[T, n_yt] = size(y_t);
[~, n_yc] = size(y_c);
p         = rsugg;
LR_trace  = [];

fprintf('\n--- SVARCholDiD_estimate  T=%d  p=%d  D_exog=%d ---\n', T, p, D_exog);

%% Cointegrating vector
YC_t  = [y_t, y_c];
nvarC = size(YC_t, 2);

if ~isempty(beta_fix)
    beta_coint = beta_fix(:);
else
    if isempty(ncoint); ncoint = nvarC - 1; end
    if isempty(other)
        [alpha, beta, GAMMA, ~, ~, ~, ~, LR_trace_raw, ~] = VECMmlrestr(YC_t, p, ncoint);
    else
        [alpha, beta, GAMMA, ~, ~, ~, ~, LR_trace_raw, ~] = VECMmlrestr(YC_t, p, ncoint, other);
    end
    beta_coint = beta(1:nvarC, 1:ncoint);
    fprintf('  beta: '); fprintf('%.4f  ', beta_coint(:,1)'); fprintf('\n');
    crits = [282.45;236.54;196.37;159.48;126.58;97.18;71.86;49.65;32.00;17.85;7.52];
    n  = length(LR_trace_raw);
    cv = crits(max(1,end-n+1):end);
    if length(cv) < n; cv = [zeros(n-length(cv),1); cv]; end
    LR_trace = table(round(LR_trace_raw(:),2), cv(end-n+1:end), ...
        'VariableNames', {'LR_Trace','Critical_Values'});
    disp(LR_trace);
end

%% EC term: purge D_t (and D_{t-1}) so ec_lag is orthogonal to treatment
D_lag    = [0; D_t(1:end-1)];
ec_raw   = YC_t * beta_coint;
ec_clean = zeros(size(ec_raw));
for k = 1:size(ec_raw, 2)
    if include_D_lag
        Xp = [D_t, D_lag, ones(T,1)];
    else
        Xp = [D_t, ones(T,1)];
    end
    ec_clean(:,k) = ec_raw(:,k) - Xp*(Xp\ec_raw(:,k));
end
ec_clean    = ec_clean - mean(ec_clean);
ec_lag      = ec_clean(p:T-1, :);
ncoint_used = size(ec_lag, 2);

fprintf('  corr(ec_lag, ytlag1) = %.4f', corr(ec_lag(:,1), y_t(p:T-1,1)));
if abs(corr(ec_lag(:,1), y_t(p:T-1,1))) > 0.9
    fprintf('  WARNING: near-collinear');
end
fprintf('\n');

%% Build regressor matrix
nobs  = T - p;
noth  = size(other, 2);
ncst  = const > 0;

if D_exog
    n_D_cols    = 1 + include_D_lag;
    ncols_endo  = n_yt * p;
    ncols_exog  = n_D_cols + ncoint_used + noth + ncst;
    ncols_total = ncols_endo + ncols_exog;
    X = zeros(nobs, ncols_total);
    for lag = 1:p
        X(:, (lag-1)*n_yt+(1:n_yt)) = y_t(p+1-lag:T-lag, :);
    end
    col_D = ncols_endo + 1;
    X(:, col_D) = D_t(p+1:T);
    if include_D_lag
        col_D1 = ncols_endo + 2;
        X(:, col_D1) = D_lag(p+1:T);
    else
        col_D1 = [];
    end
    X(:, ncols_endo+n_D_cols+(1:ncoint_used)) = ec_lag;
    if noth > 0; X(:, ncols_endo+n_D_cols+ncoint_used+(1:noth)) = other(p+1:T,:); end
    if ncst;     X(:, end) = 1; end
    Y = y_t(p+1:T, :);
else
    n_endo      = 1 + n_yt;
    Y_endo_full = [D_t, y_t];
    ncols_endo  = n_endo * p;
    ncols_exog  = ncoint_used + noth + ncst;
    ncols_total = ncols_endo + ncols_exog;
    col_D  = [];
    col_D1 = [];
    X = zeros(nobs, ncols_total);
    for lag = 1:p
        X(:, (lag-1)*n_endo+(1:n_endo)) = Y_endo_full(p+1-lag:T-lag, :);
    end
    X(:, ncols_endo+(1:ncoint_used)) = ec_lag;
    if noth > 0; X(:, ncols_endo+ncoint_used+(1:noth)) = other(p+1:T,:); end
    if ncst;     X(:, end) = 1; end
    Y = Y_endo_full(p+1:T, :);
end

%% OLS
B     = (X'*X) \ (X'*Y);
Resid = Y - X*B;
SIGMA = (Resid'*Resid) / (nobs - ncols_total);

if D_exog
    gamma_hat     = B(col_D, :);
    gamma_hat_lag = [];
    if include_D_lag; gamma_hat_lag = B(col_D1, :); end
    P_chol  = [];
    d_unit  = [];
    Resid_y = Resid;
    SIGMA_y = SIGMA;
else
    [P_chol, ok] = chol(SIGMA, 'lower');
    if ok ~= 0
        [Vv,Dd] = eig((SIGMA+SIGMA')/2);
        SIGMA   = Vv*diag(max(diag(Dd),1e-10))*Vv';
        P_chol  = chol(SIGMA, 'lower');
    end
    d_unit        = P_chol(:,1) / P_chol(1,1);
    gamma_hat     = d_unit(2:end)';
    gamma_hat_lag = [];
    Resid_y       = Resid(:, 2:end);
    SIGMA_y       = SIGMA(2:end, 2:end);
    fprintf('  gamma0_hat = %.4f  (Cholesky P(2,1)/P(1,1))\n', gamma_hat(1));
end

fprintf('  sigma_u = %.4f   maxEig = ', sqrt(SIGMA_y(1,1)));

%% Companion matrix
if D_exog
    n_endo = n_yt;
else
    n_endo = 1 + n_yt;
end
F_endo = B(1:ncols_endo, :)';
if p > 1
    Fcomp = [F_endo; eye(n_endo*(p-1)), zeros(n_endo*(p-1), n_endo)];
else
    Fcomp = F_endo;
end
maxEig = max(abs(eig(Fcomp)));
fprintf('%.5f\n', maxEig);

%% Pack
VAR.B_ols         = B;
VAR.Resid         = Resid_y;
VAR.Resid_full    = Resid;
VAR.SIGMA         = SIGMA_y;
VAR.SIGMA_full    = SIGMA;
VAR.gamma_hat     = gamma_hat;
VAR.gamma_hat_lag = gamma_hat_lag;
VAR.include_D_lag = include_D_lag;
VAR.P_chol        = P_chol;
VAR.d_unit        = d_unit;
VAR.X             = X;
VAR.Y             = Y;
VAR.nobs          = nobs;
VAR.ncols         = ncols_total;
VAR.ncols_endo    = ncols_endo;
VAR.nlag          = p;
VAR.p             = p;
VAR.nvar          = n_yt;
VAR.n_yt          = n_yt;
VAR.n_yc          = n_yc;
VAR.n_endo        = n_endo;
VAR.col_D         = col_D;
VAR.col_D1        = col_D1;
VAR.D_exog        = D_exog;
VAR.Fcomp         = Fcomp;
VAR.maxEig        = maxEig;
VAR.beta_coint    = beta_coint;
VAR.ec            = ec_clean;
VAR.ec_lag        = ec_lag;
VAR.D_t           = D_t;
VAR.D_lag         = D_lag;
VAR.y_t           = y_t;
VAR.y_c           = y_c;
VAR.other         = other;
VAR.const         = const;
VAR.F             = B';
VAR.Ft            = B;
VAR.sigma         = SIGMA_y;
VAR.ENDO_mat      = Y;
VAR.nvar_ex       = ncols_exog;
VAR.nlag_ex       = 0;
VAR.ncoeff        = ncols_endo;
VAR.ntotcoeff     = ncols_total;
VAR.resid         = Resid_y;
VAR.B             = [];
VAR.Biv           = [];
VAR.PSI           = [];
VAR.Fp            = [];
VAR.IV            = [];

VARopt        = VARoption;
VARopt.vnames = colnames;
VARopt.snames = colnames;
end