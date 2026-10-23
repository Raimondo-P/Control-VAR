function tr = sim_truth(d, T_long, p_long, seed)
%SIM_TRUTH  Targets of a design from one long simulation: the pseudo-true weights (dynamic least
%   squares with six leads and lags of the trend directions), the pseudo-true response of the cycle
%   gamma_h = E[G*_{t+h} u_t]/E[u_t^2] and the causal effect theta_h = E[Eff_{t+h} u_t]/E[u_t^2],
%   with u_t the innovation of W on R_t and p_long lags of (R, W, G*).
if nargin < 2 || isempty(T_long), T_long = 100000; end
if nargin < 3 || isempty(p_long), p_long = 24; end
if nargin < 4 || isempty(seed), seed = 987654; end
HM = 24;
S = sim_simulate(d, T_long, RandStream('mt19937ar', 'Seed', seed));
Lam = sim_design_matrices(d.K, d.r, d.boundary);
if d.r == d.K, D1 = eye(d.K); else, D1 = null(null(Lam')'); end
[a, b] = cvar.dols(S.Y1, S.Y0, [S.R, S.W], D1, [], 12, 6, d.constrained);
G = S.Y1 - a - S.Y0 * b;
Z = [S.R, S.W, G];
X = [ones(T_long - p_long, 1), S.R(p_long + 1:end)];
for j = 1:p_long, X = [X, Z(p_long - j + 1:T_long - j, :)]; end %#ok<AGROW>
w = S.W(p_long + 1:end);
u = w - X * (X \ w);
Gs = G(p_long + 1:end); Es = S.Eff(p_long + 1:end);
n = numel(u) - HM; uu = u(1:n)' * u(1:n);
gam = zeros(HM + 1, 1); the = zeros(HM + 1, 1);
for h = 0:HM
    gam(h + 1) = Gs(h + 1:h + n)' * u(1:n) / uu;
    the(h + 1) = Es(h + 1:h + n)' * u(1:n) / uu;
end
tr = struct('gamma', gam, 'theta', the, 'a_star', a, 'b_star', b);
end
