"""First steps of the estimator (Section 4): rank of the controls, dynamic least squares weights,
cycle, and level shifts in the counterfactual cycle."""
import numpy as np
from scipy.optimize import nnls
from scipy.linalg import null_space
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order

from .utils import lagmat

_JOH_COL = {0.10: 0, 0.05: 1, 0.01: 2}


def rank_step(Y0, method="johansen", level=0.05, k_ar_diff="aic", maxlags=12):
    """Number of cointegrating relations among the controls and the trend directions D1.

    Returns (m, D1): m cointegrating relations (so r = K - m common trends) and a K x (K - m)
    matrix spanning the orthogonal complement of the estimated cointegrating vectors
    (D1 = I_K when m = 0).

    method: 'johansen' (trace test at `level`, 0.10, 0.05 or 0.01), 'bic' (information
    criterion on the Johansen eigenvalues with penalty log(T), consistent for the rank as in
    Cheng and Phillips 2009), or an integer m fixed by the user.
    k_ar_diff: lags in differences of the Johansen regression, 'aic' (at least 1) or an integer.
    """
    Y0 = np.asarray(Y0, dtype=float)
    K = Y0.shape[1]
    if K == 1:
        return 0, np.eye(1)
    if k_ar_diff == "aic":
        k = max(select_order(Y0, maxlags=maxlags, deterministic="ci").aic, 1)
    else:
        k = int(k_ar_diff)
    jo = coint_johansen(Y0, 0, k)
    if isinstance(method, (int, np.integer)):
        m = int(method)
    elif method == "johansen":
        col = _JOH_COL[level]
        m = next((i for i in range(K) if jo.lr1[i] < jo.cvt[i, col]), K)
    elif method == "bic":
        Teff = Y0.shape[0] - k - 1
        lam = np.clip(np.sort(jo.eig)[::-1], 0, 1 - 1e-12)
        ic = [Teff * np.sum(np.log(1 - lam[mm:])) + np.log(Teff) * (2 * K * mm - mm ** 2)
              for mm in range(K + 1)]
        m = int(np.argmin(ic))
    else:
        raise ValueError("method must be 'johansen', 'bic' or an integer")
    if m >= K:
        # every control stationary: no trend directions, the leads and lags drop out
        return K, np.zeros((K, 0))
    D1 = np.eye(K) if m == 0 else null_space(jo.evec[:, :m].T)
    return m, D1


def dols(y, Y0, pol, D1, exog=None, q=12, L=1, constrained=True):
    """Dynamic least squares weights, regression (eq:dols) of the paper:
        y_t = a + b'Y0_t + sum_{j=0..q} k_j' pol_{t-j} + sum_{j=-L..L} d_j' dX1_{t+j} + g' exog_t + e_t,
    with dX1 = D1' dY0 the changes in the trend directions of the controls, and b >= 0 when
    constrained (non-negative least squares after partialling out the other regressors).
    Returns (a, b, g) with g the coefficients on exog (empty when exog is None)."""
    y = np.asarray(y, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    T, K = Y0.shape
    parts = [np.ones((T, 1))]
    if pol is not None and np.asarray(pol).size:
        parts.append(lagmat(pol, range(0, q + 1)))
    if D1.shape[1] > 0:
        dX1 = np.vstack([np.full((1, D1.shape[1]), np.nan), np.diff(Y0, axis=0) @ D1])
        parts.append(lagmat(dX1, range(-L, L + 1)))
    ne = 0
    if exog is not None and np.asarray(exog).size:
        ex = np.asarray(exog, dtype=float).reshape(T, -1)
        ne = ex.shape[1]
        parts.append(ex)
    F = np.hstack(parts)
    ok = ~np.isnan(F).any(1)
    yy, X, F = y[ok], Y0[ok], F[ok]
    if constrained:
        P = np.linalg.pinv(F)
        b, _ = nnls(X - F @ (P @ X), yy - F @ (P @ yy))
        g = P @ (yy - X @ b)
    else:
        c = np.linalg.lstsq(np.hstack([F, X]), yy, rcond=None)[0]
        g, b = c[:F.shape[1]], c[F.shape[1]:]
    return g[0], b, (g[-ne:] if ne else np.zeros(0))


def supf_mean_break(c, trim=0.15, nw=12):
    """Andrews (1993) sup-Wald statistic for one break in the mean, Newey-West variance with
    `nw` lags. The 5 percent critical value for one restriction and 15 percent trimming is 8.85.
    Returns (statistic, index of the break)."""
    c = np.asarray(c, dtype=float)
    T = len(c)
    best, arg = 0.0, None
    for k in range(int(trim * T), int((1 - trim) * T)):
        X = np.column_stack([np.ones(T), (np.arange(T) >= k).astype(float)])
        b = np.linalg.lstsq(X, c, rcond=None)[0]
        u = c - X @ b
        Q = X.T @ X / T
        g = X * u[:, None]
        S = g.T @ g / T
        for l in range(1, nw + 1):
            G = g[l:].T @ g[:-l] / T
            S += (1 - l / (nw + 1)) * (G + G.T)
        Qi = np.linalg.inv(Q)
        V = Qi @ S @ Qi / T
        w = b[1] ** 2 / V[1, 1]
        if w > best:
            best, arg = w, k
    return best, arg


def net_cycle(c, pol, q=12):
    """The cycle net of the current value and q lags of the policies, an estimate of the
    counterfactual cycle (Assumption 4). Returns the series and the number of leading
    observations dropped."""
    T = len(c)
    Z = np.column_stack([np.ones(T), lagmat(pol, range(q + 1))])
    ok = ~np.isnan(Z).any(1)
    co = np.linalg.lstsq(Z[ok], c[ok], rcond=None)[0]
    return c[ok] - Z[ok, 1:] @ co[1:], int((~ok).sum())


def fit_weights(y, Y0, pol, D1, q=12, L=1, constrained=True, breaks=None,
                max_breaks=2, break_cv=8.85, nw=12):
    """Weights with optional level shifts in the counterfactual cycle.

    breaks: None (no shifts), 'auto' (sequential sup-Wald tests on the cycle net of the policy
    terms, adding a step dummy while the statistic exceeds break_cv, at most max_breaks), or a
    list of observation indices at which the level shifts.
    Returns a dict with a, b, steps (T x nb), shift coefficients, break indices, cycle and the
    sup-Wald statistics before and after."""
    T = len(y)

    def est(bd):
        st = (np.column_stack([(np.arange(T) >= k).astype(float) for k in bd])
              if bd else None)
        a, b, sh = dols(y, Y0, pol, D1, exog=st, q=q, L=L, constrained=constrained)
        stm = st if st is not None else np.zeros((T, 0))
        return a, b, sh, stm, y - a - Y0 @ b - stm @ sh

    a, b, sh, st, c = est([])
    c0, off = net_cycle(c, pol, q)
    sf0, kb = supf_mean_break(c0, nw=nw)
    bd = []
    if breaks == "auto":
        sf = sf0
        while sf > break_cv and len(bd) < max_breaks:
            bd.append(kb + off)
            a, b, sh, st, c = est(bd)
            cn, off = net_cycle(c, pol, q)
            sf, kb = supf_mean_break(cn, nw=nw)
    elif breaks:
        bd = [int(k) for k in breaks]
        a, b, sh, st, c = est(bd)
    cn, _ = net_cycle(c, pol, q)
    sf1, _ = supf_mean_break(cn, nw=nw)
    return dict(a=a, b=b, shift=sh, steps=st, breaks=bd, cycle=c, supf_before=sf0, supf_after=sf1)
