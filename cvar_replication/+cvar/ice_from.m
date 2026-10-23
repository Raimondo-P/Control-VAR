function tau = ice_from(gamma, psi)
%ICE_FROM  Impulse causal effects of Theorem 4(a).
%   tau = cvar.ice_from(gamma, psi): tau_0 = gamma_0 and tau_h = gamma_h - sum_{j<h} tau_j psi_{h-j},
%   column by column for a matrix gamma ((H+1) x m); psi is (H+1) x 1.
tau = zeros(size(gamma));
tau(1, :) = gamma(1, :);
for h = 1:size(gamma, 1) - 1
    acc = zeros(1, size(gamma, 2));
    for j = 0:h - 1
        acc = acc + tau(j + 1, :) * psi(h - j + 1);
    end
    tau(h + 1, :) = gamma(h + 1, :) - acc;
end
end
