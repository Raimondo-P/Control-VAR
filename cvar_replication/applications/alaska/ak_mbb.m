function rows = ak_mbb(s, M, ell, B)
%AK_MBB  Moving-block bootstrap rows (1-based): B resamples of M rows in blocks of length ell.
nb = ceil(M / ell);
st = randi(s, M - ell + 1, B, nb) - 1;
rows = zeros(B, M);
for b = 1:B
    R = st(b, :)' + (0:ell - 1);
    r = reshape(R', [], 1);
    rows(b, :) = r(1:M)' + 1;
end
end
