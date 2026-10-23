function [m, D1, Hc] = sim_rank_step(Y0)
%SIM_RANK_STEP  Rank step of the simulations: Johansen trace test at 5 percent with the number of
%   lagged differences set by AIC in levels (at most 6 lags) minus one, at least one. Returns the
%   number of cointegrating relations m, the trend directions D1 and the cointegrating vectors Hc.
K = size(Y0, 2);
k = max(sim_ic_lags(Y0, 6) - 1, 1);
jo = cvar.johansen(Y0, k);
m = find(jo.lr1 < jo.cvt(:, 2), 1) - 1;
if isempty(m), m = K; end
if m == 0
    D1 = eye(K); Hc = zeros(K, 0);
else
    Hc = jo.evec(:, 1:m); D1 = null(Hc');
end
end
