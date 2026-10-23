function [Lam, b_true, lam] = sim_design_matrices(K, r, boundary)
%SIM_DESIGN_MATRICES  Loadings of the controls, true weights and loadings of the treated unit.
if r == K
    Lam = eye(K) + 0.3;
else
    Lam = [eye(r); ones(K - r, r) / r] + 0.3;
end
b_true = ones(K, 1) / K;
if boundary
    b_true(end) = 0; b_true(1:end - 1) = 1 / (K - 1);
end
lam = Lam' * b_true;
end
