function [best, arg] = supf_mean_break(c, trim, nw)
%SUPF_MEAN_BREAK  Andrews (1993) sup-Wald statistic for one break in the mean, Newey-West variance
%   with nw lags (5 percent critical value 8.85 with 15 percent trimming). arg is the first
%   observation (1-based) of the second regime.
if nargin < 2 || isempty(trim), trim = 0.15; end
if nargin < 3 || isempty(nw), nw = 12; end
c = c(:);
T = numel(c);
best = 0; arg = [];
for k0 = floor(trim * T):floor((1 - trim) * T) - 1
    X = [ones(T, 1), double((0:T - 1)' >= k0)];
    b = lsqminnorm(X, c);
    u = c - X * b;
    Q = X' * X / T;
    gm = X .* u;
    S = gm' * gm / T;
    for l = 1:nw
        G = gm(l + 1:end, :)' * gm(1:end - l, :) / T;
        S = S + (1 - l / (nw + 1)) * (G + G');
    end
    Qi = inv(Q);
    V = Qi * S * Qi / T;
    w = b(2)^2 / V(2, 2);
    if w > best
        best = w; arg = k0 + 1;
    end
end
end
