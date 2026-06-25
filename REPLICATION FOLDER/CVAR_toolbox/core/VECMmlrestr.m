function [alpha,beta,GAMMA,Lam,SIGMAml,U,U2,LR_trace,LR_max]=VECMmlrestr(yyy,r,p,other)
%========================================================================
% Calculates VECM using ml
%========================================================================
% [alpha,beta,GAMMA,Lam,SIGMAml]=VECMmlrestr(y,p,r)
% -----------------------------------------------------------------------
% INPUT
%	- y: [y_t y_c], matrix containing treated and control series ordered
%	appropriately to causal effects (N-1 x T )
%	- r: lag length
%	- p: cointegration order
% -----------------------------------------------------------------------
% OUTPUT
%	- alpha: intercept of cointegration relation (N-1 x p)
%   - beta: cointegrating vector containing intercept ( N x p) 
%   - GAMMA = lags coefficients with lags in second dimension (Nx1) is lag(1) ((N-1) x ((N-1)x3) )
%   - Lam
%   - SIGMAml = covariance matrix of errors((N-1) x (N-1))
% -----------------------------------------------------------------------
% =======================================================================
% Raimondo Pala
% raimondopala@gmail.com
% May 2023. 
% -----------------------------------------------------------------------
[t,q]=size(yyy);
ydif=dif(yyy);
yyy=yyy';
ydif=ydif';
DY=ydif(:,r:t-1);	
if nargin<4
    X=ones(1,t-r);
else
    X= [ones(1,t-r) ; other(1:t-r,:)'];
end
for i=1:r-1
 	X=[X; ydif(:,r-i:t-1-i)];
end;

Y=yyy(:,r:t-1);
Y=[Y; ones(1,t-r)];
X= [X(2:end,:)]; 

R0=DY-(DY*X'/(X*X'))*X;
R1=Y-(Y*X'/(X*X'))*X;
S00=R0*R0'/(t-r);
S11=R1*R1'/(t-r);
S01=R0*R1'/(t-r);

% Compute ML estimates
%
iS11sq=inv(sqrtm(S11));
[B,Lam]=eig(iS11sq*S01'*inv(S00)*S01*iS11sq);
lam=diag(Lam);
[lamsort,index]=sort(lam,'descend');
Lam=diag(lamsort);
B=B(:,index); 
beta=iS11sq*B(:,1:p); % this needs to be restricted in all values that are not i column with i*2 row
alpha=S01*beta*inv(beta'*S11*beta);
X = X(1:end-1,:);
GAMMA=(DY-alpha*beta'*Y)*X'/(X*X');

U=DY-alpha*beta'*Y-GAMMA*X;
U2=DY-alpha*beta'*Y;
SIGMAml=U*U'/(t-r);


% LR test statistics
Lam=eye(length(Lam))-Lam;
for j = 1:size(Lam,2)
    lambda(j) = Lam(j,j);
end
lambda=-(t-p)*log(lambda);
ja = linspace(length(Lam),1,length(Lam));
for j = 1 : length(lambda)
    LR(j) =  sum(lambda(ja(j):length(lambda)));
end
LR_trace = flip(LR)';
LR_max=lambda;




