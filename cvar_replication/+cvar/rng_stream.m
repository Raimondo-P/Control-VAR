function s = rng_stream(seed)
%RNG_STREAM  A Mersenne Twister stream with the given seed, or the global stream when seed is empty.
if nargin < 1 || isempty(seed)
    s = RandStream.getGlobalStream();
else
    s = RandStream('mt19937ar', 'Seed', seed);
end
end
