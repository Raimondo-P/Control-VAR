function result = bgtest(X, Y, lags)
    [n, k] = size(X);
    m = lags;
    fit = fitlm([ones(n,1) X],Y, 'Intercept', false);
    resi = table2array(fit.Residuals(:,1));
    Z = lagmatrix(resi,1);
    for i = 2: lags
        Z = [Z lagmatrix(resi,i)];
    end
    na = isnan(Z(:,end));
    X2 = X(~na, :);
    Z2 = Z(~na, :);
    resinona = resi(~na,:);
    n = size(X2, 1);
    auxfit = fitlm([ones(n,1) X2 Z2],resinona, 'Intercept', false);
    cf = table2array(auxfit.Coefficients(:,1));
    vc = auxfit.CoefficientCovariance;
    dfe = auxfit.DFE;
    df = struct();
    bg = n *(sum(auxfit.Fitted(lags+1:end,1).^2) / sum(resinona.^2));
    p_val = 1 - chi2cdf(bg, m);
    df.df = m;
    result = struct('statistic', bg, 'parameter', df, 'method', ['Breusch-Godfrey test for serial correlation of order up to 1'], 'pvalue', p_val, 'coefficients', cf, 'vcov', vc);
    result.dfe = dfe;
   
end
