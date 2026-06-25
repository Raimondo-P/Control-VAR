%==========================================================================
% example_single.m  —  Single dataset: CVAR vs VAR-diff
% Pala (2025), arXiv:2510.23762
%==========================================================================
clear; clc; close all;

addpath('CVAR_toolbox')
addpath('CVAR_toolbox/identification')
addpath('CVAR_toolbox/core')
addpath('CVAR_toolbox/diagnostics')
addpath('CVAR_toolbox/plotting')

rng(2506);

T       = 600;
gamma0  = -1.0;
p_D     = 0.10;
sig_eta = 1.0;
sig_u   = 0.05;   % small idiosyncratic noise -> clean signal
sig_w   = 0.05;
p_lag   = 2;
H       = 20;
burn    = 200;
N       = burn + T;

colnames = {'D_t','y_t','y_c'};
folder   = pwd;

tau = cumsum(sig_eta * randn(N,1));
D   = rand(N,1) < p_D;
y_t = tau + gamma0*D + sig_u*randn(N,1);
y_c = tau +            sig_w*randn(N,1);

D_t = D(burn+1:end);
y_t = y_t(burn+1:end);
y_c = y_c(burn+1:end);

fprintf('T=%d  treated=%d (%.1f%%)  corr(y_t,y_c)=%.3f\n\n', ...
    T, sum(D_t), 100*mean(D_t), corr(y_t,y_c));

%% VAR-diff
% dy_t(t) = y_t(t+1)-y_t(t) is driven by D_t(t+1), hence the +1 offset
dy   = diff(y_t);
nobs = T - 1 - p_lag;
Yd   = dy(p_lag+1:T-1);
Dt_d = D_t(p_lag+2:T);
Xd   = zeros(nobs, p_lag+1);
for j = 1:p_lag; Xd(:,j) = dy(p_lag+1-j:T-1-j); end
Xd(:,end) = Dt_d;
Bd = (Xd'*Xd)\(Xd'*Yd);
Fd = Bd(1:p_lag)';

irf_d = zeros(H,1); irf_d(1) = Bd(end);
for h = 2:H
    r = zeros(p_lag,1);
    r(1:min(h-1,p_lag)) = irf_d(max(1,h-p_lag):h-1);
    irf_d(h) = Fd*r;
end
IRF_vd = cumsum(irf_d);

Res = Yd - Xd*Bd; Res = Res - mean(Res);
nboot = 500;
boot_vd = zeros(nboot,H);
for b = 1:nboot
    idx = randi(nobs,nobs,1);
    Bb  = (Xd'*Xd)\(Xd'*(Xd*Bd + Res(idx)));
    Fb  = Bb(1:p_lag)';
    ib  = zeros(H,1); ib(1) = Bb(end);
    for h = 2:H
        r = zeros(p_lag,1);
        r(1:min(h-1,p_lag)) = ib(max(1,h-p_lag):h-1);
        ib(h) = Fb*r;
    end
    boot_vd(b,:) = cumsum(ib)';
end
IRF_vd_lo = quantile(boot_vd,0.025)';
IRF_vd_hi = quantile(boot_vd,0.975)';

%% CVAR
VAR_cvar = SVARCholDiD_estimate(D_t, y_t, y_c, p_lag, colnames, folder, ...
    [], [], [1;-1], 0, 0, 0);
[IRF_cv, IRF_cv_lo, IRF_cv_hi] = SVARCholDiD_irf(VAR_cvar, VARoption, H, 95, [], 0);

true_irf = [gamma0; zeros(H-1,1)];

fprintf('  %-6s %12s %12s %12s\n','h','True ATT','VAR-diff','CVAR');
for h = [1,2,5,10,H]
    fprintf('  %-6d %12.4f %12.4f %12.4f\n', h-1, true_irf(h), IRF_vd(h), IRF_cv(h));
end

h_ax = (0:H-1)';
figure('Position',[80 80 900 460]); hold on;
fill([h_ax;flipud(h_ax)],[IRF_vd_hi;flipud(IRF_vd_lo)],[1 .75 .75],'EdgeColor','none','FaceAlpha',.5,'HandleVisibility','off');
fill([h_ax;flipud(h_ax)],[IRF_cv_hi;flipud(IRF_cv_lo)],[.75 .88 1],'EdgeColor','none','FaceAlpha',.5,'HandleVisibility','off');
plot(h_ax, true_irf,'--','Color',[.1 .65 .3],'LineWidth',2.5,'DisplayName','True ATT');
plot(h_ax, IRF_vd,  '-', 'Color',[.85 .3 .3],'LineWidth',2,  'DisplayName','VAR-diff (no control)');
plot(h_ax, IRF_cv,  '-', 'Color',[.2 .5 .8], 'LineWidth',2,  'DisplayName','CVAR (beta=[1,-1])');
yline(0,'Color',[.5 .5 .5],'HandleVisibility','off');
legend('FontSize',10); grid on; box on; xlim([0 H-1]);
xlabel('Horizon h'); ylabel('Response of y_t');
title(sprintf('VAR-diff vs CVAR  (T=%d, \\gamma_0=%.1f, rng=1)', T, gamma0));