"""Test of the design (Section 5): point-optimal GLS test of the null that the treated unit has a
stochastic trend of its own (the CVAR is not valid), with the sieve-bootstrap p-value of
Proposition 6. Rejection supports the design."""
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import lfilter

from .utils import lagmat, lag_rule

# Elliott, Rothenberg and Stock (1996), Table I, constant, cbar = -7, asymptotic.
ERS_CV_CONSTANT = {0.01: 1.99, 0.025: 2.55, 0.05: 3.26, 0.10: 4.48}


def ssr_qd(y, X, alpha):
    """Sum of squared residuals of the quasi-differenced regression, first observation in levels."""
    yd = np.r_[y[0], y[1:] - alpha * y[:-1]]
    Xd = np.vstack([X[0], X[1:] - alpha * X[:-1]])
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    e = yd - Xd @ b
    return e @ e


def gls_resid(y, X, alpha):
    yd = np.r_[y[0], y[1:] - alpha * y[:-1]]
    Xd = np.vstack([X[0], X[1:] - alpha * X[:-1]])
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    return y - X @ b


def maic_lrv(u, pmax=12):
    """Autoregressive spectral estimator of the long-run variance from the ADF regression on the
    GLS residual, without deterministic terms, lag chosen by the modified AIC of Ng and Perron (2001)."""
    du = np.diff(u)
    n = len(du) - pmax
    best, lrv = None, {}
    for p in range(0, pmax + 1):
        Y = du[pmax:]
        Z = np.column_stack([u[pmax:-1]] + [du[pmax - j:len(du) - j] for j in range(1, p + 1)])
        b = np.linalg.lstsq(Z, Y, rcond=None)[0]
        e = Y - Z @ b
        s2 = e @ e / n
        tau = b[0] ** 2 * (u[pmax:-1] @ u[pmax:-1]) / s2
        crit = np.log(s2) + 2 * (tau + p) / n
        lrv[p] = s2 / (1 - b[1:].sum()) ** 2 if p > 0 else s2
        if best is None or crit < best[0]:
            best = (crit, p)
    return lrv[best[1]], best[1]


def po_statistic(y, X, cbar=-7.0, pmax=12):
    """P_T = [S(abar) - abar S(1)] / omega2, abar = 1 + cbar / T. X must contain the constant.
    Returns (statistic, MAIC lag of the long-run variance)."""
    T = len(y)
    a = 1 + cbar / T
    num = ssr_qd(y, X, a) - a * ssr_qd(y, X, 1.0)
    w2, k = maic_lrv(gls_resid(y, X, a), pmax)
    return num / w2, k


def sieve_pvalue(y, X, stat, B, rng, p_sieve, cbar=-7.0, pmax=12, burn=100):
    """Sieve bootstrap under the null: residuals of the regression in differences (alpha = 1) are
    fitted by an AR(p_sieve), its residuals resampled, and Y1 rebuilt by cumulating the fitted
    changes and the bootstrap errors with the regressors held at their observed values."""
    dy = np.diff(y)
    dX = np.diff(X, axis=0)
    keep = np.abs(dX).sum(0) > 1e-12
    Z = np.column_stack([np.ones(len(dy)), dX[:, keep]])
    g = np.linalg.lstsq(Z, dy, rcond=None)[0]
    fit = Z @ g
    e = dy - fit
    p = p_sieve
    if p > 0:
        Zp = np.column_stack([e[p - j:len(e) - j] for j in range(1, p + 1)])
        phi = np.linalg.lstsq(Zp, e[p:], rcond=None)[0]
        res = e[p:] - Zp @ phi
    else:
        phi, res = np.zeros(0), e
    res = res - res.mean()
    draws = np.empty(B)
    for i in range(B):
        es = lfilter([1.0], np.r_[1.0, -phi], res[rng.integers(0, len(res), len(e) + burn)])[burn:]
        ys = np.r_[y[0], y[0] + np.cumsum(fit + es)]
        draws[i] = po_statistic(ys, X, cbar, pmax)[0]
    return (np.sum(draws <= stat) + 1) / (B + 1), draws


def sieve_pvalue_residual(y, X, stat, B, rng, p_sieve, cbar=-7.0, pmax=12, burn=100):
    """Residual-based sieve bootstrap under the null (Paparoditis and Politis 2003; Palm, Smeekes and
    Urbain 2008). The innovations come from the unrestricted autoregression in levels,
        du_t = mu + gamma u_{t-1} + sum_{j=1..p} phi_j du_{t-j} + eps_t,   u = y - X b (OLS in levels),
    so that under the alternative they are close to the innovations of the stationary cycle. The bootstrap
    imposes the unit root (gamma = 0) when rebuilding the series, du*_t = mu + sum_j phi_j du*_{t-j} + eps*_t,
    and keeps the regressors at their observed values, y* = X b + u*."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    du = np.diff(u)
    p = int(p_sieve)
    n = len(du) - p
    Z = np.column_stack([np.ones(n), u[p:-1]] + [du[p - j:len(du) - j] for j in range(1, p + 1)])
    coef = np.linalg.lstsq(Z, du[p:], rcond=None)[0]
    eps = du[p:] - Z @ coef
    eps = eps - eps.mean()
    mu, phi = coef[0], coef[2:]
    draws = np.empty(B)
    T = len(y)
    for i in range(B):
        e = eps[rng.integers(0, len(eps), T - 1 + burn)] + mu
        dus = lfilter([1.0], np.r_[1.0, -phi], e)[burn:]
        us = np.r_[u[0], u[0] + np.cumsum(dus)]
        draws[i] = po_statistic(X @ b + us, X, cbar, pmax)[0]
    return (np.sum(draws <= stat) + 1) / (B + 1), draws


@dataclass
class DesignTestResult:
    statistic: float
    pvalue_bootstrap: float
    critical_values: dict
    T: int
    p_sieve: int
    maic_lag: int
    cbar: float
    B: int
    draws: np.ndarray = field(repr=False, default=None)

    def reject(self, level=0.05, method="bootstrap"):
        """True when the null of a unit root in the cycle is rejected (the design is supported)."""
        if method == "bootstrap":
            return self.pvalue_bootstrap <= level
        return self.statistic < self.critical_values[level]

    def summary(self):
        cv = ", ".join(f"{100 * k:g}%: {v}" for k, v in self.critical_values.items())
        return (f"P_T = {self.statistic:.3f} (T = {self.T}, cbar = {self.cbar}); "
                f"bootstrap p-value {self.pvalue_bootstrap:.3f} (B = {self.B}, sieve order {self.p_sieve}); "
                f"ERS critical values {cv}")


def design_test(y, controls, policy=None, q=12, exog=None, cbar=-7.0, B=999, p_sieve="rule",
                pmax=12, seed=None, keep_draws=False, bootstrap="difference"):
    """Test of the design for one treated outcome.

    y: (T,) treated outcome in levels. controls: (T, K) controls in levels. policy: (T, np) policies
    (the treated policy and any other policy), entered with their current value and q lags, as in
    the weights regression. exog: (T, ne) further regressors of the weights regression, such as
    step dummies for level shifts (their quasi-differences are impulses and leave the limit
    unchanged). The first q observations are dropped to build the lags.
    cbar = -7 (constant): asymptotic critical values of ERS (1996). p_sieve: 'rule' for
    ceil(16 (T/200)^(1/3)) or an integer. bootstrap: 'difference' (the sieve fitted to the changes, as in
    Proposition 6) or 'residual' (innovations from the unrestricted autoregression in levels, unit root
    imposed when rebuilding the series). Returns a DesignTestResult; small values reject."""
    if cbar != -7.0:
        raise NotImplementedError("tabulated critical values are implemented for cbar = -7 with a constant")
    y = np.asarray(y, dtype=float)
    T0 = len(y)
    parts = [np.ones((T0, 1)), np.asarray(controls, dtype=float).reshape(T0, -1)]
    if policy is not None and np.asarray(policy).size:
        parts.append(lagmat(policy, range(q + 1)))
    if exog is not None and np.asarray(exog).size:
        parts.append(np.asarray(exog, dtype=float).reshape(T0, -1))
    X = np.hstack(parts)
    ok = ~np.isnan(X).any(1)
    y, X = y[ok], X[ok]
    # drop exogenous columns that are constant in the retained sample (they duplicate the constant)
    var = X.std(0)
    X = np.column_stack([X[:, 0], X[:, 1:][:, var[1:] > 1e-12]])
    T = len(y)
    stat, k = po_statistic(y, X, cbar, pmax)
    ps = lag_rule(T) if p_sieve == "rule" else int(p_sieve)
    rng = np.random.default_rng(seed)
    if B <= 0:
        pval, draws = np.nan, None
    elif bootstrap == "residual":
        pval, draws = sieve_pvalue_residual(y, X, stat, B, rng, ps, cbar, pmax)
    elif bootstrap == "difference":
        pval, draws = sieve_pvalue(y, X, stat, B, rng, ps, cbar, pmax)
    else:
        raise ValueError("bootstrap must be 'difference' or 'residual'")
    return DesignTestResult(statistic=stat, pvalue_bootstrap=pval, critical_values=dict(ERS_CV_CONSTANT),
                            T=T, p_sieve=ps, maic_lag=k, cbar=cbar, B=B,
                            draws=draws if keep_draws else None)
