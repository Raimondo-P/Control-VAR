function result = adfTest(x)
%========================================================================
% Perorm ADF test (see ADF)
%========================================================================
% result = adfTest(x)
% -----------------------------------------------------------------------
% INPUT
%	- x: series to test (T x 1)
% -----------------------------------------------------------------------
% OUTPUT
% -----------------------------------------------------------------------
% =======================================================================
% Raimondo Pala
% raimondopala@gmail.com
% May 2023. 
% -----------------------------------------------------------------------
    x = double(x);
    if length(x) >10
    [accepts,PVAL,STAT,CRIT]=adftest(x,'Model','TS');
%     [accepts,PVAL,STAT,CRIT]=adftest(x);
    tablep = [0.01, 0.025, 0.05, 0.1];
    table = [-4.0008, -3.6911, -3.4316, -3.1391];
    result.statistic = STAT;
    result.pValue = PVAL;
    result.table = table;
    result.true = [ tablep ; STAT<table];
    result.interp = {'If true == 0, no trend'};
    result = struct(result);
    else
        disp(['We cannot conduct an ADF test for t<10, cell will be empty there'])
        result.statistic = NaN;
        result.pValue = NaN;
        result.table = NaN(1,4);
        result.true = NaN(2,4);
        result.interp = {''};
        result = struct(result);
    end
end