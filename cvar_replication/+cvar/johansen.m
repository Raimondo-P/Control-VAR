function jo = johansen(Y, k)
%JOHANSEN  Johansen trace test with a constant (det_order = 0) and k lagged differences,
%   following statsmodels' coint_johansen step by step so that the eigenvalues, the trace
%   statistics and the critical values coincide with the Python version.
%   jo.eig      eigenvalues, in decreasing order
%   jo.evec     eigenvectors in the same order (columns)
%   jo.lr1      trace statistics for ranks 0..K-1
%   jo.cvt      critical values (90, 95, 99 percent), MacKinnon, Haug and Michelis (1999)
[T, K] = size(Y);
endog = Y - mean(Y, 1);                       % detrend of order 0
dx = diff(endog, 1, 1);
dxk = dx(k + 1:end, :);
Zl = zeros(size(dxk, 1), 0);
for j = 1:k
    Zl = [Zl, dx(k + 1 - j:end - j, :)]; %#ok<AGROW>
end
if ~isempty(Zl), Zl = Zl - mean(Zl, 1); end
dxk = dxk - mean(dxk, 1);
r0t = presid(dxk, Zl);
lx = endog(2:T - k, :);
lx = lx - mean(lx, 1);
rkt = presid(lx, Zl);
t = size(rkt, 1);
skk = rkt' * rkt / t;
sk0 = rkt' * r0t / t;
s00 = r0t' * r0t / t;
sig = sk0 * (s00 \ sk0');
[du, au] = eig(skk \ sig);
au = real(diag(au)); du = real(du);
temp = inv(chol(du' * skk * du, 'lower'));
dt = du * temp;
[a, order] = sort(au, 'descend');
d = dt(:, order);
nz = find(d(:) ~= 0, 1);
if ~isempty(nz), d = d * sign(d(nz)); end
lr1 = zeros(K, 1);
cvt = zeros(K, 3);
for i = 1:K
    lr1(i) = -t * sum(log(1 - a(i:end)));
    cvt(i, :) = trace_cv(K - i + 1);
end
jo = struct('eig', a, 'evec', d, 'lr1', lr1, 'cvt', cvt);
end

function r = presid(y, x)
if isempty(x)
    r = y;
else
    r = y - x * (pinv(x) * y);
end
end

function cv = trace_cv(n)
% Trace critical values with a constant (statsmodels c_sjt(n, 0)), 90, 95, 99 percent.
tab = [2.7055 3.8415 6.6349; 13.4294 15.4943 19.9349; 27.0669 29.7961 35.4628; ...
       44.4929 47.8545 54.6815; 65.8202 69.8189 77.8202; 91.1090 95.7542 104.9637; ...
       120.3673 125.6185 135.9825; 153.6341 159.5290 171.0905; 190.8714 197.3772 210.0366; ...
       232.1030 239.2468 253.2526; 277.3740 285.1402 300.2821; 326.5354 334.9795 351.2150];
if n >= 1 && n <= 12
    cv = tab(n, :);
else
    cv = nan(1, 3);
end
end
