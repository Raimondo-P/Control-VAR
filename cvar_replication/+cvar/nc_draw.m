function g = nc_draw(res, idx, H)
%NC_DRAW  One residual-bootstrap draw of the response of the VAR without controls.
p = res.opts.p; nR = res.nR; T = size(res.Z, 1);
Zvs = cvar.var_sim(res.nc.A, res.nc.U, res.nc.Z(1:p, :), p, T, idx, res.var_exog);
[A3, U3] = cvar.var_fit(Zvs, p, res.var_exog);
R3 = cvar.irf(A3, U3, p, nR + 1, H);
g = cvar.cumresp(R3(:, nR + 2:end), res.diff);
end
