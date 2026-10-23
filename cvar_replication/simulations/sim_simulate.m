function S = sim_simulate(d, T, s, burn)
%SIM_SIMULATE  The data generating process of the estimator simulations (Section S.4):
%   dF_t = 0.3 dF_{t-1} + eta_t (r common trends); Y0_t = Lam F_t + v_t, v AR(1) 0.5;
%   Y1_t(0) = lam'F_t + e_t, e AR(1) 0.5, lam = Lam' b_true;
%   R_t = 0.7 R_{t-1} + c_eta sum(eta_t)/sqrt(r) + eps_R;
%   W_t = 0.5 W_{t-1} + c_eta sum(eta_t)/sqrt(r) + 0.3 R_t + feedback Gtrue_{t-1} + eps_W;
%   Eff_t = 0.8 Eff_{t-1} + theta0 W_t; Y1_t = Y1_t(0) + Eff_t; Gtrue_t = Y1_t - b_true'Y0_t.
%   s is a RandStream.
if nargin < 4, burn = 300; end
K = d.K; r = d.r;
[Lam, b_true, lam] = sim_design_matrices(K, r, d.boundary);
n = T + burn;
eta = randn(s, n, r); ev = randn(s, n, K); ee = randn(s, n, 1); eR = randn(s, n, 1); eW = randn(s, n, 1);
dF = zeros(n, r); F = zeros(n, r); v = zeros(n, K); e = zeros(n, 1);
R = zeros(n, 1); W = zeros(n, 1); Eff = zeros(n, 1); Gt = zeros(n, 1);
common = sum(eta, 2) / sqrt(r);
for t = 2:n
    dF(t, :) = 0.3 * dF(t - 1, :) + eta(t, :);
    F(t, :) = F(t - 1, :) + dF(t, :);
    v(t, :) = 0.5 * v(t - 1, :) + ev(t, :);
    e(t) = 0.5 * e(t - 1) + ee(t);
    R(t) = 0.7 * R(t - 1) + d.c_eta * common(t) + eR(t);
    W(t) = 0.5 * W(t - 1) + d.c_eta * common(t) + 0.3 * R(t) + d.feedback * Gt(t - 1) + eW(t);
    Eff(t) = 0.8 * Eff(t - 1) + d.theta0 * W(t);
    y0 = Lam * F(t, :)' + v(t, :)';
    y1 = lam' * F(t, :)' + e(t) + Eff(t);
    Gt(t) = y1 - b_true' * y0;
end
Y0 = F * Lam' + v; Y1 = F * lam + e + Eff;
k = burn + 1:n;
S = struct('Y1', Y1(k), 'Y0', Y0(k, :), 'R', R(k), 'W', W(k), 'Eff', Eff(k));
end
