function p = sieve_order(T)
%SIEVE_ORDER  Order of the sieve autoregression in the bootstrap of the design test (Section 5):
%   p_T = ceil(16 (T/200)^(1/3)).
p = ceil(16 * (T / 200)^(1 / 3));
end
