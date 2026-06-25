function[rsugg]=check_lags(Y_t,r,foldername,Q_tilde)
%========================================================================
% Find vector of weights (w_i)
%========================================================================
% [rsugg]=check_lags(Y_tilde,Q_tilde)
% -----------------------------------------------------------------------
% INPUT
%   - Y_tilde: [D_t \Delta y_t ] (Tx(dd+nn))
%	- Q-tilde: [\Delta y_c \tilde{y}] (Tx(dd*2))
%   - r: maximum lags to be considered (1x1)

% --------------------------------------
% OUTPUT
%   - rsugg: suggested lags
% -----------------------------------------------------------------------
% =======================================================================
% Raimondo Pala
% raimondopala@gmail.com
% May 2023. 
% -----------------------------------------------------------------------
if nargin<4
    % BIC part
    s = size(Y_t,2);
    for i = 1:r
        Mdl(i) = varm(s,i);
        EstMdl(i) = estimate(Mdl(i),Y_t-lagmatrix(Y_t,1));
        Results{i} = summarize(EstMdl(i));
        bics(i) = Results{i}.BIC;
    end
    [~,rsugg] = min(bics);
    disp(['BIC suggests using ' num2str(rsugg) ' lags'])


    % Breush Godfrey part
    for i = 1:r % i varies according to lags included
        res = bgtest_VAR(Y_t,i);
        bg(i) = res.statistic; %rows are variables, j are lags
        cr(i,:) = res.critical;
        bgsig(i) = res.statistic>cr(i,2); % is statistic greater than pvalue
    end
    i = 1;
    while i<r
        if bgsig(i)==1
            disp(['Breush-Godfrey test suggests a lag of ' num2str(i) ])
            break
        else
            i = i+1;
        end
    end


    

else
    % BIC part
    s1 = size(Y_t,2);
    s2 = size(Q_tilde,2);
    s = s1+s2;
    for i = 1:r
        Mdl(i) = varm(s,i);
        EstMdl(i) = estimate(Mdl(i),[Y_t Q_tilde]-lagmatrix([Y_t Q_tilde],1));
        Results{i} = summarize(EstMdl(i));
        bics(i) = Results{i}.BIC;
    end
    [~,rsugg] = min(bics);
    disp(['BIC suggests using ' num2str(rsugg) ' lags'])
    % Breush Godfrey part
    for i = 1:r % i varies according to lags included
        res = bgtest_VAR(Y_t,i,Q_tilde);
        bg(i) = res.statistic; %rows are variables, j are lags
        cr(i,:) = res.critical;
        bgsig(i) = res.statistic>cr(i,2); % is statistic greater than pvalue
    end
    i =1;
    while i<r
        if bgsig(i)==1
            disp(['Breush-Godfrey test suggests a lag of ' num2str(i) ])
            break
        else
            i = i+1;
        end
    end
end

% BIC part
headers = cell(1, length(bics));
for i = 1:length(bics)
    headers{i} = sprintf('%d lag', i);
end
to_tab = [ headers; num2cell(round(bics,2))]';
filename = fullfile(foldername, 'BIC_lags.xlsx');
% writecell(to_tab, filename);

% Breush Godfrey part 
bgtab = [bg ; cr']';
filePath = fullfile(foldername, 'BreushGodfrey.xlsx');
variableLabels = {'Test', 'critical alpha = .9', 'critical alpha = .95', 'critical alpha = .99'};
lagLabels = cellstr(strcat('Lag ', num2str((1:size(bgtab, 1))')));
dataTable = array2table(round(bgtab,2), 'RowNames', lagLabels, 'VariableNames', variableLabels);
% writetable(dataTable, filePath, 'WriteRowNames', true, 'WriteVariableNames', true);

