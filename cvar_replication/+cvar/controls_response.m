function cr = controls_response(res, H, B, seed, indices)
%CONTROLS_RESPONSE  Proposition 1: recursive VAR in (R, W, Y, C) with C = b'Y0 the weighted
%   controls. Returns the responses of Y (the VAR with the controls added), of C (the bias from the
%   common trends) and of G = Y - C, with their bootstrap draws (weights held at their estimates).
if nargin < 2 || isempty(H), H = 36; end
if nargin < 3 || isempty(B), B = 499; end
if nargin < 4, seed = []; end
if nargin < 5, indices = []; end
p = res.opts.p; m = res.m; nR = res.nR;
Za = res.cr.Z; A = res.cr.A; U = res.cr.U;
Rr = cvar.irf(A, U, p, nR + 1, H);
pt.treated = cvar.cumresp(Rr(:, nR + 2:nR + 1 + m), res.diff);
pt.controls = cvar.cumresp(Rr(:, nR + 2 + m:end), res.diff);
pt.difference = pt.treated - pt.controls;
n = size(U, 1);
if isempty(indices)
    s = cvar.rng_stream(seed);
    indices = randi(s, n, B, n);
end
B = size(indices, 1);
dt = zeros(B, H + 1, m); dc = zeros(B, H + 1, m);
for b = 1:B
    Zs = cvar.var_sim(A, U, Za(1:p, :), p, size(Za, 1), indices(b, :)', res.var_exog);
    [A2, U2] = cvar.var_fit(Zs, p, res.var_exog);
    R2 = cvar.irf(A2, U2, p, nR + 1, H);
    dt(b, :, :) = reshape(cvar.cumresp(R2(:, nR + 2:nR + 1 + m), res.diff), 1, H + 1, m);
    dc(b, :, :) = reshape(cvar.cumresp(R2(:, nR + 2 + m:end), res.diff), 1, H + 1, m);
end
draws = struct('treated', dt, 'controls', dc, 'difference', dt - dc);
cr = struct('point', pt, 'draws', draws, 'names', {res.names});
end
