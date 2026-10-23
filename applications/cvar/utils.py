"""Small helpers shared by the modules."""
import math
import numpy as np


def lagmat(x, lags):
    """Stack lags (positive), the current value (0) and leads (negative) of x, padding with NaN."""
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    x = x.reshape(T, -1)
    out = []
    for j in lags:
        z = np.full(x.shape, np.nan)
        if j > 0:
            z[j:] = x[:T - j]
        elif j == 0:
            z[:] = x
        else:
            z[:T + j] = x[-j:]
        out.append(z)
    return np.hstack(out) if out else np.zeros((T, 0))


def lag_rule(T):
    """Order of the sieve autoregression in the bootstrap of the design test (Section 5):
    p_T = ceil(16 (T/200)^(1/3)), 16 lags at T = 200 and 21 at T = 400."""
    return int(math.ceil(16 * (T / 200) ** (1 / 3)))


def as2d(x, T=None):
    """Return x as a float (T, k) array; None becomes a (T, 0) array."""
    if x is None:
        if T is None:
            raise ValueError("T is required to build an empty block")
        return np.zeros((T, 0))
    x = np.asarray(x, dtype=float)
    return x.reshape(x.shape[0], -1)


def band(draws, alpha=0.10, point=None, method="percentile"):
    """Bands from bootstrap draws along axis 0.
    percentile: the alpha/2 and 1 - alpha/2 quantiles of the draws (the bands of the paper).
    hall: 2 * point minus the upper and lower quantiles (requires point)."""
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    if method == "percentile":
        return lo, hi
    if method == "hall":
        if point is None:
            raise ValueError("method='hall' needs the point estimate")
        return 2 * point - hi, 2 * point - lo
    raise ValueError("method must be 'percentile' or 'hall'")
