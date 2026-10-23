function pv = ljungbox_p(x, m, df_adjust)
%LJUNGBOX_P  p-value of the Ljung-Box test of no autocorrelation up to lag m.
if nargin < 3, df_adjust = 0; end
x = x(:) - mean(x);
n = numel(x);
d = x' * x;
q = 0;
for k = 1:m
    q = q + (x(k + 1:end)' * x(1:end - k) / d)^2 / (n - k);
end
q = n * (n + 2) * q;
pv = cvar.chi2_sf(q, max(m - df_adjust, 1));
end
