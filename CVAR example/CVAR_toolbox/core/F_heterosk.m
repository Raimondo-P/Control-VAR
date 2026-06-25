function[Fstat]=F_heterosk(D,Z)
[b,~,r]=regress(D,Z); % first stage
varb=length(D)*mean((Z.*r).^2)/sum(Z.^2)^2;  % heteroskedasticity-robust asymptotic variance
Fstat=b^2/varb; % F-statistic
end