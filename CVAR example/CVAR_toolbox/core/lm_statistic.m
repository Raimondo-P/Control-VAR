function[LM,crit,is_sign,tab_sig]=lm_statistic(y,x,z)
%========================================================================
% Test for common cycles
%========================================================================
% []=test_cycles(y_t,y_c,T)
% -----------------------------------------------------------------------
% INPUT
%	- y_t: outcome variable California (T x 1)
%	- y_c: control series ordered as Cholesky (T x n)
%   - T: vector of number of observations (1x1)
% -----------------------------------------------------------------------
% OUTPUT
%   - LM: value of the LM statistic
%   - crit: associated critical values
%   - tab_sig: table where the first column is the LM statistic, the second
%   is alpha, the third is whether it accepts
% -----------------------------------------------------------------------
%   - comment: lm test statistics for the test gamma = 0 in
%   y = beta * x + gamma * z + e
% =======================================================================
% Raimondo Pala
% raimondopala@gmail.com
% May 2023. 
% -----------------------------------------------------------------------
if isempty(x)
    Mx = eye(length(y));
else
    I = eye(length(x));
    Mx = I - x*(x'*x)^-1*x';
end
gamma_hat = (z'*Mx*z)^-1*z'*Mx*y;
V_gamma = var(y)*(z'*Mx*z)^-1;
LM = y'*Mx*z*(z'*Mx*z)^-1*z'*Mx*y/var(y);
df = size(x,2)+size(z,2);
crit = [chi2inv(.90,df) chi2inv(.95,df) chi2inv(.99,df)];
is_sign(1:3) = LM > crit;
alpha = [.90 .95 .99];
tab_sig = table(repmat(LM,3,1), alpha', is_sign', 'VariableNames', {'LM value','significance values', 'LM accept'});
end