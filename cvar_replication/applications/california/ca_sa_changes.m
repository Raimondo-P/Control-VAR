function z = ca_sa_changes(x, groups)
%CA_SA_CHANGES  Remove calendar effects from the changes of a level series: for each grouping
%   (e.g. calendar months), the group means of the changes minus the overall mean; re-cumulate.
x = x(:); dx = diff(x); adj = zeros(size(dx));
for k = 1:numel(groups)
    g = groups{k}(:); g = g(2:end);
    s = dx - adj;
    [u, ~, ig] = unique(g);
    eff = accumarray(ig, s, [numel(u), 1], @mean) - mean(s);
    adj = adj + eff(ig);
end
z = [x(1); x(1) + cumsum(dx - adj)];
end
