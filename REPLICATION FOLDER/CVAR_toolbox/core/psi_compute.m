function [psi] = psi_compute(T,q);
% Compute PSI Matrix for Cosine Transformation
% Inputs"
% T = time series length
% q = number of cosine transforms
% Output:
% psi = Txq matrix
% Note, Scale normalization has psi'*psi = (1/T)*I(q) .. dct is divided by T

  fvec = pi*(1:1:q);
  tvec = (0.5:1:T)'/T;
  psi = (sqrt(2)/T)*cos(tvec*fvec);
  
 % fvec_2T = fvec/(2*T);
 % ljt = (sin(fvec_2T))./fvec_2T;
 % psi = psi.*repmat(ljt,T,1);
  
end

