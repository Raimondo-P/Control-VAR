"""Reduced-form VAR, recursive impulse responses, residual-bootstrap simulation, and the
recursion of Theorem 4(a) for the impulse causal effects."""
import numpy as np


def var_fit(Z, p, exog=None):
    """OLS VAR(p) with a constant and optional exogenous regressors (e.g. dummies).
    Returns the stacked coefficients A ((1 + n p + ne) x n) and the residuals U ((T - p) x n)."""
    Z = np.asarray(Z, dtype=float)
    T, n = Z.shape
    X = [np.ones((T - p, 1))] + [Z[p - j - 1:T - j - 1] for j in range(p)]
    if exog is not None:
        X.append(np.asarray(exog, dtype=float).reshape(T, -1)[p:])
    X = np.hstack(X)
    A = np.linalg.lstsq(X, Z[p:], rcond=None)[0]
    return A, Z[p:] - X @ A


def irf(A, U, p, shock, H):
    """Responses of all variables at horizons 0..H to the recursive (Cholesky) innovation of
    variable `shock`, normalised to a unit impact on that variable. Returns (H + 1) x n."""
    n = U.shape[1]
    S = np.linalg.cholesky(np.cov(U.T))
    s = S[:, shock] / S[shock, shock]
    Al = [A[1 + j * n:1 + (j + 1) * n].T for j in range(p)]
    R = np.zeros((H + 1, n))
    R[0] = s
    for h in range(1, H + 1):
        R[h] = sum(Al[j] @ R[h - j - 1] for j in range(min(p, h)))
    return R


def var_sim(A, U, Z0, p, T, idx, exog=None):
    """Rebuild a VAR path recursively from the initial values Z0 (p x n) and the centred
    residual rows U[idx] (residual bootstrap). idx has length T - p."""
    n = U.shape[1]
    E = U[idx] - U.mean(0)
    Al = [A[1 + j * n:1 + (j + 1) * n] for j in range(p)]
    c = A[0]
    Ex = None
    if exog is not None:
        ex = np.asarray(exog, dtype=float).reshape(T, -1)
        Ex = ex @ A[1 + n * p:]
    Z = np.zeros((T, n))
    Z[:p] = Z0
    for t in range(p, T):
        z = c + E[t - p]
        for j in range(p):
            z = z + Z[t - j - 1] @ Al[j]
        if Ex is not None:
            z = z + Ex[t]
        Z[t] = z
    return Z


def ice_from(gamma, psi):
    """Impulse causal effects from Theorem 4(a): tau_0 = gamma_0 and
    tau_h = gamma_h - sum_{j<h} tau_j psi_{h-j}. Works column by column for a matrix gamma."""
    gamma = np.asarray(gamma, dtype=float)
    tau = np.zeros_like(gamma)
    tau[0] = gamma[0]
    for h in range(1, len(gamma)):
        tau[h] = gamma[h] - sum(tau[j] * psi[h - j] for j in range(h))
    return tau
