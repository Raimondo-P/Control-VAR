function [IRbar, VARopt, IRinf, IRsup, rsugg, LR_trace, VAR] = SVARCholDiD(D_t, y_t, y_c, r, horizons, colnames, other, foldername, ncoint, beta_fix, const, D_exog, include_D_lag)
%========================================================================
% SVARCholDiD — Full pipeline wrapper
%========================================================================
% EXAMPLES
%
%   % Fully automatic:
%   SVARCholDiD(D_t, y_t, y_c, r_max, horizons, cols, [], 'out');
%
%   % Known rank and coint vector, no D lag:
%   SVARCholDiD(D_t, y_t, y_c, 2, horizons, cols, [], 'out', 1, [1 -1]);
%
%   % Known rank and coint vector, with D lag:
%   SVARCholDiD(D_t, y_t, y_c, 2, horizons, cols, [], 'out', 1, [1 -1], 0, 1, 1);
%========================================================================

%% Defaults
if nargin < 9  || isempty(ncoint);        ncoint        = []; end
if nargin < 10 || isempty(beta_fix);      beta_fix      = []; end
if nargin < 11 || isempty(const);         const         = 0;  end
if nargin < 12 || isempty(D_exog);        D_exog        = 1;  end
if nargin < 13 || isempty(include_D_lag); include_D_lag = 0;  end

%% Step 1: Lag selection
if ~isempty(beta_fix)
    rsugg = r;
else
    rsugg = SVARCholDiD_lags(D_t, y_t, y_c, r, foldername, other);
end

%% Step 2: Estimation
[VAR, VARopt, LR_trace] = SVARCholDiD_estimate(D_t, y_t, y_c, rsugg, colnames, foldername, other, ncoint, beta_fix, const, D_exog, include_D_lag);

%% Step 3: IRF
[IRbar, IRinf, IRsup, VAR, VARopt] = SVARCholDiD_irf(VAR, VARopt, horizons);
end