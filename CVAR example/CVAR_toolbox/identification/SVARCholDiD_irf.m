function [IRbar, IRinf, IRsup, VAR, VARopt] = SVARCholDiD_irf(VAR, VARopt, horizons, pctg, ~, plot_irf)
%========================================================================
% STEP 3: IRF and bootstrap bands for SVARCholDiD
%========================================================================
% D_exog == 1  (legacy / local-projection mode):
%   h=0   : gamma0 (OLS coefficient on contemporaneous D_t)
%   h=1   : gamma1 (if include_D_lag) + AR companion propagation
%   h>1   : purely AR companion propagation of the y_t block
%
% D_exog == 0  (Cholesky-SVAR mode, D_t endogenous, ordered first):
%   The structural shock is a unit innovation to D_t's own equation
%   (Cholesky-orthogonalized so y_t shocks cannot move D_t on impact).
%   It is propagated through the FULL joint [D_t,y_t] companion matrix
%   at every horizon, so D_t's own persistence and its feedback into
%   y_t are captured throughout, not just at h=0/1.
%========================================================================

%% Defaults
if nargin < 4 || isempty(pctg);     pctg     = 95; end
if nargin < 6 || isempty(plot_irf); plot_irf = 1;  end

nboot  = 100;
alpha  = (100 - pctg) / 2 / 100;
H      = horizons;
n_yt   = VAR.n_yt;
p      = VAR.p;
X      = VAR.X;
B      = VAR.B_ols;
Fcomp  = VAR.Fcomp;
nobs   = VAR.nobs;
D_exog = VAR.D_exog;

if isfield(VAR,'Resid_full') && ~isempty(VAR.Resid_full)
    Resid_full = VAR.Resid_full;
else
    Resid_full = VAR.Resid;
end

%% IRF propagation helper for a single block of size n with p lags
    function IR = compute_irf(g0, g1, Fc, n, h)
        % g0 : (1 x n) impact vector
        % g1 : (1 x n) extra lagged-D effect at h=1, [] if not used
        % IR : (h x n)
        IR = zeros(h, n);
        IR(1, :) = g0;
        for hh = 2:h
            resp         = zeros(n*p, 1);
            resp(1:n)    = IR(hh-1, :)';
            IR(hh, :)    = (Fc(1:n, :) * resp)';
            if hh == 2 && ~isempty(g1)
                IR(hh, :) = IR(hh, :) + g1;
            end
        end
    end

if D_exog
    %% ===================== LEGACY MODE (D_t exogenous) ================
    g1 = VAR.gamma_hat_lag;
    IRbar = compute_irf(VAR.gamma_hat, g1, Fcomp, n_yt, H);

    IR_boot  = zeros(nboot, H, n_yt);
    resid_dm = Resid_full - mean(Resid_full);

    for b = 1:nboot
        idx    = randi(nobs, nobs, 1);
        Y_boot = X * B + resid_dm(idx, :);
        B_boot = (X'*X) \ (X'*Y_boot);

        F_boot = B_boot(1:VAR.ncols_endo, :)';
        if p > 1
            Fc_boot = [F_boot; eye(n_yt*(p-1)), zeros(n_yt*(p-1), n_yt)];
        else
            Fc_boot = F_boot;
        end

        g0_b = B_boot(VAR.col_D, :);
        if ~isempty(VAR.col_D1)
            g1_b = B_boot(VAR.col_D1, :);
        else
            g1_b = [];
        end

        IR_boot(b,:,:) = compute_irf(g0_b, g1_b, Fc_boot, n_yt, H);
    end

    IRbar_D = [];   % not applicable in legacy mode

else
    %% ===================== CHOLESKY MODE (D_t endogenous) =============
    n_endo     = VAR.n_endo;        % = 1 + n_yt
    ncols_endo = VAR.ncols_endo;    % = n_endo * p

    %% Point estimate: propagate the unit-D-shock through the full system
    d_unit  = VAR.d_unit;            % (n_endo x 1), d_unit(1) = 1
    IR_full = zeros(H, n_endo);
    IR_full(1, :) = d_unit';
    for hh = 2:H
        resp           = zeros(n_endo*p, 1);
        resp(1:n_endo) = IR_full(hh-1, :)';
        IR_full(hh, :) = (Fcomp(1:n_endo, :) * resp)';
    end
    IRbar_D = IR_full(:, 1);        % D_t's own impulse response (diagnostic)
    IRbar   = IR_full(:, 2:end);    % y_t response -- this is the function's main output

    %% Bootstrap: resample joint residuals, re-Cholesky-factorize each draw
    IR_boot  = zeros(nboot, H, n_yt);
    resid_dm = Resid_full - mean(Resid_full);

    for b = 1:nboot
        idx    = randi(nobs, nobs, 1);
        Y_boot = X * B + resid_dm(idx, :);
        B_boot = (X'*X) \ (X'*Y_boot);
        R_boot = Y_boot - X*B_boot;
        SIG_b  = (R_boot'*R_boot) / (nobs - VAR.ntotcoeff);

        F_boot = B_boot(1:ncols_endo, :)';
        if p > 1
            Fc_boot = [F_boot; eye(n_endo*(p-1)), zeros(n_endo*(p-1), n_endo)];
        else
            Fc_boot = F_boot;
        end

        [P_b, flag_b] = chol(SIG_b, 'lower');
        if flag_b ~= 0
            [Vv, Dd] = eig((SIG_b + SIG_b')/2);
            Dd       = diag(max(diag(Dd), 1e-10));
            P_b      = chol(Vv*Dd*Vv', 'lower');
        end
        d_b = P_b(:,1) / P_b(1,1);

        IRb_full = zeros(H, n_endo);
        IRb_full(1,:) = d_b';
        for hh = 2:H
            resp           = zeros(n_endo*p, 1);
            resp(1:n_endo) = IRb_full(hh-1,:)';
            IRb_full(hh,:) = (Fc_boot(1:n_endo,:) * resp)';
        end
        IR_boot(b,:,:) = IRb_full(:, 2:end);
    end
end

IRinf = squeeze(quantile(IR_boot, alpha,   1));
IRsup = squeeze(quantile(IR_boot, 1-alpha, 1));

if n_yt == 1
    IRinf = IRinf(:);
    IRsup = IRsup(:);
end

fprintf('  IRF h=0..4: '); fprintf('%.4f  ', IRbar(1:min(5,H),1)); fprintf('\n');
fprintf('  95%% band h=0: [%.4f, %.4f]\n', IRinf(1), IRsup(1));
if ~D_exog
    fprintf('  [Cholesky mode] D_t own IRF h=0..4: '); fprintf('%.4f  ', IRbar_D(1:min(5,H))); fprintf('\n');
end

%% Plot
if plot_irf && n_yt <= 6
    h_ax = 0:H-1;
    figure('Name','CVAR IRF','NumberTitle','off', ...
           'Position',[100 100 300*n_yt 350]);
    for k = 1:n_yt
        subplot(1, n_yt, k);
        fill([h_ax fliplr(h_ax)],[IRsup(:,k)' fliplr(IRinf(:,k)')], ...
            [0.7 1 0.8],'EdgeColor','none','FaceAlpha',0.6); hold on;
        plot(h_ax, IRbar(:,k), 'r-', 'LineWidth', 2);
        yline(0,'Color',[0.6 0.6 0.6]);
        title(['D\_t \rightarrow ' VARopt.vnames{k+1}]);
        xlabel('Horizon'); ylabel('Response');
    end
    sgtitle('CVAR — IRF of D\_t');
end

end