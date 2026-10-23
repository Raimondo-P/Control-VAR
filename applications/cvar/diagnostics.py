"""Lag-length diagnostics for the VAR of the CVAR: information criteria on a common sample and
residual autocorrelation tests at each candidate order."""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .var import var_fit


def ljungbox_p(x, m=12, df_adjust=0):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    d = x @ x
    q = n * (n + 2) * sum((x[k:] @ x[:-k] / d) ** 2 / (n - k) for k in range(1, m + 1))
    return 1 - chi2.cdf(q, max(m - df_adjust, 1))


def portmanteau_p(U, h, p):
    """Adjusted multivariate portmanteau test (Hosking 1980) of no residual autocorrelation up to
    lag h in a VAR(p); chi-square with n^2 (h - p) degrees of freedom (requires h > p)."""
    U = U - U.mean(0)
    T, n = U.shape
    C0i = np.linalg.inv(U.T @ U / T)
    Q = 0.0
    for j in range(1, h + 1):
        Cj = U[j:].T @ U[:-j] / T
        Q += np.trace(Cj.T @ C0i @ Cj @ C0i) / (T - j)
    Q *= T ** 2
    return 1 - chi2.cdf(Q, n * n * (h - p))


def lag_diagnostics(Z, lags=(1, 2, 3, 4, 6, 8, 12), exog=None, extra=12, names=None):
    """For each candidate order p, on the common sample that starts after max(lags):
    AIC, BIC and HQ, the adjusted multivariate portmanteau p-value up to lag p + extra, and the
    Ljung-Box p-value of each equation's residuals up to lag `extra`.
    Returns a DataFrame with one row per order (the minimum of each criterion is flagged)."""
    Z = np.asarray(Z, dtype=float)
    T, n = Z.shape
    pm = max(lags)
    names = names or [f"eq{i}" for i in range(n)]
    rows = []
    for p in lags:
        Zs = Z[pm - p:]
        ex = None if exog is None else np.asarray(exog, dtype=float).reshape(T, -1)[pm - p:]
        A, U = var_fit(Zs, p, ex)
        Te = U.shape[0]
        k = A.shape[0] * n
        ld = np.linalg.slogdet(U.T @ U / Te)[1]
        row = dict(p=p, AIC=ld + 2 * k / Te, BIC=ld + np.log(Te) * k / Te,
                   HQ=ld + 2 * np.log(np.log(Te)) * k / Te,
                   portmanteau_p=portmanteau_p(U, p + extra, p))
        for i in range(n):
            row[f"LB_{names[i]}"] = ljungbox_p(U[:, i], extra)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("p")
    for c in ["AIC", "BIC", "HQ"]:
        df[c + "_min"] = df[c] == df[c].min()
    return df
