function Z = var_sim(A, U, Z0, p, T, idx, exog)
%VAR_SIM  Rebuild a VAR path from initial values and resampled, centred residuals.
%   Z = cvar.var_sim(A, U, Z0, p, T, idx, exog): Z0 is p x n, idx (length T - p) are the residual
%   rows used (1-based), exog is T x ne or empty (its coefficients are the last rows of A).
if nargin < 7, exog = []; end
n = size(U, 2);
E = U(idx, :) - mean(U, 1);
Al = cell(p, 1);
for j = 1:p
    Al{j} = A(1 + (j - 1) * n + 1:1 + j * n, :);
end
c = A(1, :);
Ex = [];
if ~isempty(exog)
    Ex = reshape(exog, T, []) * A(1 + n * p + 1:end, :);
end
Z = zeros(T, n);
Z(1:p, :) = Z0;
for t = p + 1:T
    z = c + E(t - p, :);
    for j = 1:p
        z = z + Z(t - j, :) * Al{j};
    end
    if ~isempty(Ex)
        z = z + Ex(t, :);
    end
    Z(t, :) = z;
end
end
