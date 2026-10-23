function x = eb_col(E, M, s)
%EB_COL  Column of the panel M (UR, EB or EUC) for the state or states s.
[~, j] = ismember(s, E.states);
x = M(:, j);
end
