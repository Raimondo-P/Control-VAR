function [dy, dX, F] = ak_design_changes(A, q)
%AK_DESIGN_CHANGES  The regression in changes of the weights:
%   dY1_t = c + b' dY0_t + sum_{j=0..q} k_j' (oil, W)_{t-j} + e_t.
dy = diff(A.y);
dX = diff(A.C, 1, 1);
pol = [A.doil, A.W];
pol = pol(2:end, :);
F = [ones(numel(dy), 1), cvar.lagmat(pol, 0:q)];
ok = all(~isnan(F), 2) & all(~isnan(dX), 2);
dy = dy(ok); dX = dX(ok, :); F = F(ok, :);
end
