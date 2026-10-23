function out = lagmat(x, lags)
%LAGMAT  Lags (positive), current value (0) and leads (negative) of x, padded with NaN.
%   out = cvar.lagmat(x, lags) stacks, for each element j of lags, x shifted by j periods.
T = size(x, 1);
x = reshape(x, T, []);
out = zeros(T, 0);
for j = lags(:)'
    z = nan(size(x));
    if j > 0
        z(j + 1:T, :) = x(1:T - j, :);
    elseif j == 0
        z = x;
    else
        z(1:T + j, :) = x(1 - j:T, :);
    end
    out = [out, z]; %#ok<AGROW>
end
end
