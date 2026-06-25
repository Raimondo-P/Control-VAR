%==========================================================================
% example_mc.m  —  Monte Carlo: CVAR vs VAR-diff
% Pala (2025), arXiv:2510.23762
%
% DGP:  y_t = tau_t + gamma0*D_t + u_t,  y_c = tau_t + w_t
%       D_t ~ Bernoulli(p_D),  y_t - y_c ~ I(0),  beta=[1,-1]
%==========================================================================
clear; clc; close all;

addpath('CVAR_toolbox')
addpath('CVAR_toolbox/identification')
addpath('CVAR_toolbox/core')
addpath('CVAR_toolbox/diagnostics')
addpath('CVAR_toolbox/plotting')

rng(42);

T       = 600;
gamma0  = -1.0;
p_D     = 0.10;
sig_eta = 1.0;
sig_u   = 0.5;
sig_w   = 0.5;
p_lag   = 2;
H       = 20;
nsim    = 2000;
burn    = 200;
N       = burn + T;
MAX_EIG = 1.15;

colnames = {'D_t','y_t','y_c'};
folder   = pwd;

g0_vd      = nan(nsim,1);
g0_cvar    = nan(nsim,1);
IRF_vd_all = nan(nsim,H);
IRF_cv_all = nan(nsim,H);

t0 = tic;
fprintf('Running %d MC draws...\n', nsim);

for s = 1:nsim

    tau = cumsum(sig_eta * randn(N,1));
    D   = rand(N,1) < p_D;
    y_t = tau + gamma0*D + sig_u*randn(N,1);
    y_c = tau +            sig_w*randn(N,1);
    D_t = D(burn+1:end);
    y_t = y_t(burn+1:end);
    y_c = y_c(burn+1:end);

    try
        dy   = diff(y_t);
        nobs = T - 1 - p_lag;
        Yd   = dy(p_lag+1:T-1);
        Dt_d = D_t(p_lag+2:T);
        Xd   = zeros(nobs, p_lag+1);
        for j = 1:p_lag; Xd(:,j) = dy(p_lag+1-j:T-1-j); end
        Xd(:,end) = Dt_d;
        Bd  = (Xd'*Xd)\(Xd'*Yd);
        Fd  = Bd(1:p_lag)';
        Fcd = [Fd; eye(p_lag-1), zeros(p_lag-1,1)];
        if max(abs(eig(Fcd))) <= 1.02 && isfinite(Bd(end))
            g0_vd(s) = Bd(end);
            irf = zeros(H,1); irf(1) = Bd(end);
            for h = 2:H
                r = zeros(p_lag,1);
                r(1:min(h-1,p_lag)) = irf(max(1,h-p_lag):h-1);
                irf(h) = Fd*r;
            end
            IRF_vd_all(s,:) = cumsum(irf)';
        end
    catch; end

    try
        evalc(['VAR = SVARCholDiD_estimate(D_t,y_t,y_c,p_lag,colnames,folder,[],[],[1;-1],0,0,0);']);
        g = VAR.gamma_hat(1);
        if isfinite(VAR.maxEig) && VAR.maxEig <= MAX_EIG && isfinite(g)
            g0_cvar(s) = g;
            evalc('[irf,~,~,~,~] = SVARCholDiD_irf(VAR,VARoption,H,95,[],0);');
            IRF_cv_all(s,:) = irf(:,1)';
        end
    catch; end

    if mod(s,200)==0
        fprintf('  %d/%d  (%.1fs)\n', s, nsim, toc(t0));
    end
end

fprintf('\n%s\n', repmat('=',1,68));
fprintf('  MC BIAS  (gamma0=%.2f, nsim=%d, T=%d)\n', gamma0, nsim, T);
fprintf('%s\n', repmat('=',1,68));
fprintf('  %-25s %6s %8s %8s %8s %8s\n','Estimator','valid','mean','bias','std','RMSE');
fprintf('  %s\n', repmat('-',1,64));
for m = 1:2
    x = {g0_vd, g0_cvar}; x = x{m}; x = x(~isnan(x));
    lbl = {'VAR-diff','CVAR [1,-1]'};
    fprintf('  %-25s %6d %8.4f %8.4f %8.4f %8.4f\n', lbl{m}, numel(x), ...
        mean(x), mean(x)-gamma0, std(x), sqrt(mean((x-gamma0).^2)));
end
fprintf('%s\n\n', repmat('=',1,68));

h_ax        = (0:H-1)';
irf_vd_mean = nanmean(IRF_vd_all)';
irf_cv_mean = nanmean(IRF_cv_all)';
irf_vd_lo   = quantile(IRF_vd_all,0.05)';
irf_vd_hi   = quantile(IRF_vd_all,0.95)';
irf_cv_lo   = quantile(IRF_cv_all,0.05)';
irf_cv_hi   = quantile(IRF_cv_all,0.95)';
true_irf    = [gamma0; zeros(H-1,1)];

figure('Position',[80 80 1200 460]);

subplot(1,3,[1 2]); hold on;
fill([h_ax;flipud(h_ax)],[irf_vd_hi;flipud(irf_vd_lo)],[1 .75 .75],'EdgeColor','none','FaceAlpha',.5,'HandleVisibility','off');
fill([h_ax;flipud(h_ax)],[irf_cv_hi;flipud(irf_cv_lo)],[.75 .88 1],'EdgeColor','none','FaceAlpha',.5,'HandleVisibility','off');
plot(h_ax, true_irf,    '--','Color',[.1 .65 .3],'LineWidth',2.5,'DisplayName','True ATT');
plot(h_ax, irf_vd_mean, '-', 'Color',[.85 .3 .3],'LineWidth',2,  'DisplayName','VAR-diff');
plot(h_ax, irf_cv_mean, '-', 'Color',[.2 .5 .8], 'LineWidth',2,  'DisplayName','CVAR');
yline(0,'Color',[.5 .5 .5],'HandleVisibility','off');
legend('FontSize',9); grid on; box on; xlim([0 H-1]);
xlabel('Horizon h'); ylabel('Response');
title(sprintf('MC-mean IRF  (nsim=%d, T=%d)', nsim, T));

subplot(1,3,3); hold on;
boxplot([g0_vd; g0_cvar],[ones(nsim,1);2*ones(nsim,1)],'Labels',{'VAR-diff','CVAR'},'Symbol','');
yline(gamma0,'k--','LineWidth',2);
ylabel('\gamma_0  (h=0)');
title('Impact effect');
grid on; box on; xtickangle(15);

sgtitle('Control VAR','FontWeight','bold');