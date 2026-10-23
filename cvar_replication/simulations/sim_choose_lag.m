function p = sim_choose_lag(Z, rule)
%SIM_CHOOSE_LAG  'aic'; 'paper': the smallest of the BIC, HQ and AIC orders whose cycle residuals
%   pass Ljung-Box(12) at 5 percent, else AIC; 'lbup': the smallest order passing Ljung-Box, else 12;
%   or a fixed order (number or digit string).
if isnumeric(rule), p = rule; return, end
if all(isstrprop(rule, 'digit')), p = str2double(rule); return, end
lb = @(q) cvar.ljungbox_p(cycle_resid(Z, q), 12);
switch rule
    case 'aic'
        p = sim_ic_lags(Z);
    case 'paper'
        [a, h, b] = sim_ic_lags(Z);
        for q = unique([a, h, b])
            if lb(q) > 0.05, p = q; return, end
        end
        p = a;
    case 'lbup'
        for q = 1:12
            if lb(q) > 0.05, p = q; return, end
        end
        p = 12;
    otherwise
        error('unknown lag rule %s', rule);
end
end

function u = cycle_resid(Z, p)
[~, U] = cvar.var_fit(Z, p); u = U(:, end);
end
