function [rsugg] = SVARCholDiD_lags(D_t, y_t, y_c, r, foldername, other)
%========================================================================
% STEP 1: Lag selection for SVARCholDiD
%========================================================================
% Runs check_lags on the full system [D_t, y_t, y_c] and returns the
% suggested lag order. Call this before SVARCholDiD_estimate.
% Skip this step entirely by passing rsugg directly to SVARCholDiD_estimate.
%
% INPUT
%    - D_t       : exogenous event (Tx1)
%    - y_t       : treated series (Txn)
%    - y_c       : control series (Txn), [] if not used
%    - r         : max lag length to consider
%    - foldername: output folder for lag selection tables
%    - other     : exogenous variables ([] if none)  [optional]
%
% OUTPUT
%    - rsugg     : suggested lag order (scalar)
%========================================================================
% Raimondo Pala — raimondopala@gmail.com
%========================================================================

if nargin < 6; other = []; end

if isempty(y_c)
    Y_t = [D_t y_t];
else
    Y_t = [D_t y_t y_c];
end

if isempty(other)
    rsugg = check_lags(Y_t, r, foldername);
else
    rsugg = check_lags(Y_t, r, foldername, other);
end

end