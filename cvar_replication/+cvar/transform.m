function out = transform(M, dfl, off)
%TRANSFORM  Differenced columns become changes, the others stay in levels; the first `off` rows drop.
out = zeros(size(M, 1) - off, size(M, 2));
for j = 1:size(M, 2)
    if dfl(j)
        out(:, j) = diff(M(:, j));
    else
        out(:, j) = M(off + 1:end, j);
    end
end
end
