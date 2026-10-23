function R = irf(A, U, p, shock, H)
%IRF  Recursive (Cholesky) impulse responses at horizons 0..H.
%   R = cvar.irf(A, U, p, shock, H) gives the responses of all variables to the innovation of
%   variable `shock` (1-based), normalised to a unit impact on that variable. R is (H+1) x n.
n = size(U, 2);
S = chol(cov(U), 'lower');
s = S(:, shock) / S(shock, shock);
Al = cell(p, 1);
for j = 1:p
    Al{j} = A(1 + (j - 1) * n + 1:1 + j * n, :)';
end
R = zeros(H + 1, n);
R(1, :) = s';
for h = 1:H
    r = zeros(n, 1);
    for j = 1:min(p, h)
        r = r + Al{j} * R(h - j + 1, :)';
    end
    R(h + 1, :) = r';
end
end
