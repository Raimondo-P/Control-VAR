"""Shared CVAR estimator (Section 4 of the paper): DOLS weights with policy terms and
leads/lags of the trend directions, cycle, VAR with optional exogenous dummies,
Cholesky response to the policy, residual bootstrap that re-estimates the weights."""
import numpy as np
from scipy.optimize import nnls
from scipy.linalg import null_space
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller, kpss

def lagmat(x, lags):
    T = x.shape[0]; x = x.reshape(T, -1); out = []
    for j in lags:
        z = np.full_like(x, np.nan, dtype=float)
        if j > 0: z[j:] = x[:T - j]
        elif j == 0: z[:] = x
        else: z[:T + j] = x[-j:]
        out.append(z)
    return np.hstack(out)

def rank_step(Y0):
    K = Y0.shape[1]
    if K == 1: return 0, np.eye(1)
    jo = coint_johansen(Y0, 0, 2)
    m = next((i for i in range(K) if jo.lr1[i] < jo.cvt[i, 1]), K)
    return m, (np.eye(K) if m == 0 else null_space(jo.evec[:, :m].T))

def dols(Y1, Y0, pol, D1, exog=None, q=12, L=1, constrained=True, return_exog=False):
    T, K = Y0.shape
    dX1 = np.vstack([np.full((1, D1.shape[1]), np.nan), np.diff(Y0, axis=0) @ D1])
    parts = [np.ones((T, 1)), lagmat(pol, range(0, q + 1)), lagmat(dX1, range(-L, L + 1))]
    if exog is not None: parts.append(exog.reshape(T, -1))
    Fm = np.hstack(parts); ok = ~np.isnan(Fm).any(1)
    y, X, Fm = Y1[ok], Y0[ok], Fm[ok]
    if not constrained:
        c = np.linalg.lstsq(np.hstack([Fm, X]), y, rcond=None)[0]
        g, b = c[:Fm.shape[1]], c[Fm.shape[1]:]
    else:
        P = np.linalg.pinv(Fm)
        b, _ = nnls(X - Fm @ (P @ X), y - Fm @ (P @ y))
        g = P @ (y - X @ b)
    if return_exog:
        ne = 0 if exog is None else exog.reshape(T, -1).shape[1]
        return g[0], b, (g[-ne:] if ne else np.zeros(0))
    return g[0], b

def var_fit(Z, p, exog=None):
    T, n = Z.shape
    X = [np.ones((T - p, 1))] + [Z[p - j - 1:T - j - 1] for j in range(p)]
    if exog is not None: X.append(exog[p:].reshape(T - p, -1))
    X = np.hstack(X)
    A = np.linalg.lstsq(X, Z[p:], rcond=None)[0]
    return A, Z[p:] - X @ A

def irf(A, U, p, n, shock, H):
    S = np.linalg.cholesky(np.cov(U.T)); s = S[:, shock] / S[shock, shock]
    Al = [A[1 + j * n:1 + (j + 1) * n].T for j in range(p)]
    R = np.zeros((H + 1, n)); R[0] = s
    for h in range(1, H + 1):
        R[h] = sum(Al[j] @ R[h - j - 1] for j in range(min(p, h)))
    return R

def var_sim(A, U, Z0, p, T, rng, exog=None):
    n = U.shape[1]; E = U[rng.integers(0, len(U), T - p)] - U.mean(0)
    Z = np.zeros((T, n)); Z[:p] = Z0
    for t in range(p, T):
        x = [np.ones(1)] + [Z[t - j - 1] for j in range(p)]
        if exog is not None: x.append(np.atleast_1d(exog[t]))
        Z[t] = np.concatenate(x) @ A + E[t - p]
    return Z

def ljungbox_p(x, m=12):
    x = x - x.mean(); n = len(x); d = x @ x
    q = n * (n + 2) * sum((x[k:] @ x[:-k] / d) ** 2 / (n - k) for k in range(1, m + 1))
    return 1 - chi2.cdf(q, m)

def supf_mean_break(c, trim=0.15, nw=12):
    """Andrews (1993) sup-Wald for a single break in the mean, Newey-West variance.
    5% critical value for one restriction and 15% trimming: 8.85."""
    T = len(c); best = 0; arg = None
    for k in range(int(trim * T), int((1 - trim) * T)):
        X = np.column_stack([np.ones(T), (np.arange(T) >= k).astype(float)])
        b = np.linalg.lstsq(X, c, rcond=None)[0]; u = c - X @ b
        Q = X.T @ X / T; g = X * u[:, None]
        S = g.T @ g / T
        for l in range(1, nw + 1):
            w = 1 - l / (nw + 1); G = g[l:].T @ g[:-l] / T; S += w * (G + G.T)
        V = np.linalg.inv(Q) @ S @ np.linalg.inv(Q) / T
        wstat = b[1] ** 2 / V[1, 1]
        if wstat > best: best, arg = wstat, k
    return best, arg

def stationarity(c):
    adf = adfuller(c, maxlag=12, autolag='BIC')[0]
    kp = kpss(c, regression='c', nlags='auto')
    return adf, kp[0], kp[1]
