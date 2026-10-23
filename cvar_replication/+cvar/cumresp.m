function Rb = cumresp(Rb, dfl)
%CUMRESP  Cumulate the responses of the differenced columns (level responses of the cycle).
for j = 1:numel(dfl)
    if dfl(j)
        Rb(:, j) = cumsum(Rb(:, j));
    end
end
end
