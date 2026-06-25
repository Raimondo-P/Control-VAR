function result = bgtest_VAR(Y, r, other)
if nargin >2
    [alpha,beta,GAMMA,Lam,SIGMAml,U,U2,LR_trace,LR_max]=VECMgls(Y,1,size(Y,2)-1,other);
    resi = U';
    [n, k] = size(other);
    m = r;
    Z = lagmatrix(resi,1);
    for i = 2: r
        Z = [Z lagmatrix(resi,i)];
    end
    na = isnan(Z(:,end));
    X2 = other(~na, :);
    Z2 = Z(~na, :);
    resinona = resi(~na,:);
    n = size(X2, 1);
    for j = 1:size(resinona,2)
        auxfit = fitlm([ones(n,1) X2 Z2],resinona(:,j), 'Intercept', false);
        cf = table2array(auxfit.Coefficients(:,1));
        vc = auxfit.CoefficientCovariance;
        dfe = auxfit.DFE;
        df = struct();
        % bg(j) = n *(sum(auxfit.Fitted(r+1:end,1).^2) / sum(resinona(:,j).^2));
         bg(j) = auxfit.Rsquared.Ordinary;

    end
    bgtot = sum(bg);
    critical = [chi2inv(.90,(size(Y,2)+size(other,2))*r); chi2inv(.95,(size(Y,2)+size(other,2))*r); chi2inv(.99,(size(Y,2)+size(other,2))*r)];
    df.df = m;
    result = struct('statistic', bgtot, 'parameter', df, 'method', ['Breusch-Godfrey test for serial correlation of order up to 1'], 'critical', critical, 'coefficients', cf, 'vcov', vc);
    result.dfe = dfe;
   
else
    [alpha,beta,GAMMA,Lam,SIGMAml,U,U2,LR_trace,LR_max]=VECMgls(Y,1,size(Y,2)-1);
    resi = U';
    m = r;
    Z = lagmatrix(resi,1);
    for i = 2: r
        Z = [Z lagmatrix(resi,i)];
    end
    na = isnan(Z(:,end));
    Z2 = Z(~na, :);
    resinona = resi(~na,:);
    for j = 1:size(resinona,2)
        auxfit = fitlm([ones(length(Z2),1) Z2],resinona(:,j), 'Intercept', false);
        cf = table2array(auxfit.Coefficients(:,1));
        vc = auxfit.CoefficientCovariance;
        dfe = auxfit.DFE;
        df = struct();
        % bg(j) = n *(sum(auxfit.Fitted(r+1:end,1).^2) / sum(resinona(:,j).^2));
         bg(j) = auxfit.Rsquared.Ordinary;
    end
    bgtot = sum(bg);
    critical = [chi2inv(.90,(size(Y,2)*r)); chi2inv(.95,size(Y,2)*r); chi2inv(.99,(size(Y,2))*r)];
    df.df = m;
    result = struct('statistic', bgtot, 'parameter', df, 'method', ['Breusch-Godfrey test for serial correlation of order up to 1'], 'critical', critical, 'coefficients', cf, 'vcov', vc);
    result.dfe = dfe;
end
