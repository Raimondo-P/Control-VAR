#!/usr/bin/env python3
"""
Power envelope and point-optimal tests for the CVAR stationarity problem.
Self-contained (numpy only).

Canonical problem (local to unity):
  x_t : r random walks (the common trends of the controls), with innovations eta_t
  W_t = 0.5 W_{t-1} + 0.5 sum(eta_t)/sqrt(r) + e_W,t      (policy reacting to common shocks)
  y_t = a + beta'x_t + sum_j tau_j W_{t-j} + u_t,  tau_j = 0.2*0.8^j
  u_t = rho u_{t-1} + eps_t,  rho = 1 + c/T,  eps_t ~ N(0,1) independent of the rest
  H0: c = 0 (unit root: the treated unit has its own trend, CVAR invalid)
  H1: c < 0 (stationary cycle, CVAR valid)

Tests (all reject H0 for small values; size-adjusted with critical values from c = 0):
  ENVELOPE   infeasible Neyman-Pearson test of c = 0 against the true c, with beta, tau
             known and sigma = 1 (its power at c is the power envelope)
  PO-known   ERS point-optimal statistic at cbar = -7 on the true u_t (beta, tau known)
  PO-CVAR    the same statistic on the residual of the CVAR weights regression
             (DOLS with current and 12 lags of W and one lead/lag of dx), long-run
             variance from a Bartlett kernel
  DFGLS-CVAR ERS DF-GLS t-statistic on the same residual
  ADF-CVAR   ADF t-statistic (constant, BIC lags) on the same residual
  ENVELOPE-INV infeasible Neyman-Pearson test at the true c, invariant to the weights:
             the regressors (constant, trends, policy terms) enter the quasi-differenced
             regression at alpha = 1 + c/T and at alpha = 1 (sigma = 1). Its power is the
             envelope among tests that do not depend on the unknown weights.
  PO-GLS     the feasible version at cbar = -7, with sigma from the differenced regression:
             the candidate optimal CVAR test.
Also reports the size of PO-CVAR with the critical value of PO-known (does the
estimation of the weights change the null distribution?).

Usage: python mc_envelope.py --quick ;  python mc_envelope.py
Output: mc_env/envelope.csv  (send me the folder)
"""
import argparse, os, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool

CBAR = -7.0
CS = [0, -2.5, -5, -7.5, -10, -15, -20, -30]

def gls_ss(u, alpha):
    """Sum of squared residuals of the quasi-differenced regression of u on a constant."""
    T = len(u)
    ud = np.r_[u[0], u[1:] - alpha * u[:-1]]
    zd = np.r_[1.0, np.full(T - 1, 1 - alpha)]
    m = (zd @ ud) / (zd @ zd)
    r = ud - zd * m
    return r @ r, m

def po_stat(u, c, s2):
    T = len(u); a = 1 + c / T
    S_a, _ = gls_ss(u, a); S_1, _ = gls_ss(u, 1.0)
    return (S_a - a * S_1) / s2

def gls_reg_ss(y, X, alpha):
    """SSR of the quasi-differenced regression of y on X (X includes a constant)."""
    yd = np.r_[y[0], y[1:] - alpha * y[:-1]]; Xd = np.vstack([X[0], X[1:] - alpha * X[:-1]])
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]; r = yd - Xd @ b
    return r @ r

def po_inv(y, X, c, s2):
    """Point-optimal statistic invariant to the weights: the regressors enter the quasi-differenced regression."""
    T = len(y); a = 1 + c / T
    return (gls_reg_ss(y, X, a) - a * gls_reg_ss(y, X, 1.0)) / s2

def lrv(e):
    T = len(e); l = int(np.floor(4 * (T / 100) ** (2 / 9)))
    e = e - e.mean(); s = e @ e / T
    for j in range(1, l + 1): s += 2 * (1 - j / (l + 1)) * (e[j:] @ e[:-j]) / T
    return s

def adf_t(y, pmax=8, gls_alpha=None):
    T = len(y)
    if gls_alpha is not None:
        _, m = gls_ss(y, gls_alpha); y = y - m          # GLS demeaning (DF-GLS)
        det = False
    else:
        det = True
    dy = np.diff(y); best = None
    for p in range(0, pmax + 1):
        Y = dy[pmax:]; X = [y[pmax:-1]] + [dy[pmax - j:-j] for j in range(1, p + 1)]
        if det: X.append(np.ones(len(Y)))
        X = np.column_stack(X); b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b
        s2 = e @ e / (len(Y) - X.shape[1]); n = len(Y)
        bic = np.log(e @ e / n) + X.shape[1] * np.log(n) / n
        if best is None or bic < best[0]:
            V = s2 * np.linalg.inv(X.T @ X); best = (bic, b[0] / np.sqrt(V[0, 0]))
    return best[1]

def lag(x, j):
    z = np.full_like(x, np.nan)
    if j > 0: z[j:] = x[:-j]
    elif j < 0: z[:j] = x[-j:]
    else: z = x.copy()
    return z

def one(args):
    T, r, c, seed = args
    rng = np.random.default_rng(seed); burn = 200; n = T + burn
    eta = rng.standard_normal((n, r)); x = np.cumsum(eta, 0)
    W = np.zeros(n); eW = rng.standard_normal(n)
    for t in range(1, n): W[t] = 0.5 * W[t-1] + 0.5 * eta[t].sum() / np.sqrt(r) + eW[t]
    E = np.zeros(n)
    for t in range(1, n): E[t] = 0.8 * E[t-1] + 0.2 * W[t]
    rho = 1 + c / T; u = np.zeros(n); eps = rng.standard_normal(n)
    for t in range(1, n): u[t] = rho * u[t-1] + eps[t]
    beta = np.ones(r) / r
    sl = slice(burn, n)
    x, W, E, u = x[sl], W[sl], E[sl], u[sl] - u[burn]
    y = 0.5 + x @ beta + E + u
    out = {}
    # envelope: NP test at the true c (for c = 0 use cbar to get the null distribution of each)
    for cc in CS[1:]:
        out[f"np_{cc}"] = po_stat(u, cc, 1.0)
    out["po_known"] = po_stat(u, CBAR, lrv(np.diff(u)) if False else 1.0)
    # CVAR weights regression: y on const, x, W_t..W_{t-12}, dx lead/lag
    dx = np.vstack([np.full((1, r), np.nan), np.diff(x, axis=0)])
    Z = [np.ones(T), x] + [lag(W, j) for j in range(13)] + [np.column_stack([lag(dx[:, k], j) for j in (-1, 0, 1)]) for k in range(r)]
    Z = np.column_stack(Z); ok = ~np.isnan(Z).any(1)
    b = np.linalg.lstsq(Z[ok], y[ok], rcond=None)[0]
    uh = y[ok] - Z[ok] @ b
    # long-run variance under the alternative from the GLS-demeaned residual's AR(1) innovations
    s2 = lrv(np.diff(uh))
    out["po_cvar"] = po_stat(uh, CBAR, s2)
    Xg = np.column_stack([np.ones(T), x] + [lag(W, j) for j in range(13)])[12:]; yg = y[12:]
    for cc in CS[1:]:
        out[f"inv_{cc}"] = po_inv(yg, Xg, cc, 1.0)
    # feasible GLS point-optimal: sigma estimated from the differenced regression residual
    yd = np.diff(yg); Xd = np.diff(Xg[:, 1:], axis=0); bd = np.linalg.lstsq(np.column_stack([np.ones(len(yd)), Xd]), yd, rcond=None)[0]
    ed = yd - np.column_stack([np.ones(len(yd)), Xd]) @ bd
    out["po_gls"] = po_inv(yg, Xg, CBAR, lrv(ed))
    out["dfgls_cvar"] = adf_t(uh, gls_alpha=1 + CBAR / len(uh))
    out["adf_cvar"] = adf_t(uh)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_env")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.reps = 200
    os.makedirs(a.out, exist_ok=True); rows = []; t0 = time.time()
    for T, r in itertools.product([200, 400], [1, 3]):
        res = {}
        for c in CS:
            args = [(T, r, c, 100000 * CS.index(c) + i) for i in range(a.reps)]
            with Pool(a.workers) as p: res[c] = pd.DataFrame(p.map(one, args, chunksize=max(1, a.reps // (4 * a.workers))))
        null = res[0]
        cv = {k: np.quantile(null[k], 0.05) for k in null.columns}
        for c in CS:
            d = res[c]
            row = dict(T=T, r=r, c=c, rho=round(1 + c / T, 4))
            row["envelope"] = np.nan if c == 0 else round(np.mean(d[f"np_{c}"] < cv[f"np_{c}"]), 3)
            row["envelope_invariant"] = np.nan if c == 0 else round(np.mean(d[f"inv_{c}"] < cv[f"inv_{c}"]), 3)
            for k in ["po_known", "po_gls", "po_cvar", "dfgls_cvar", "adf_cvar"]:
                row[k] = round(np.mean(d[k] < cv[k]), 3)
            # size of PO-CVAR at the critical value of PO-known (effect of estimating the weights)
            row["po_cvar_at_known_cv"] = round(np.mean(d["po_cvar"] < cv["po_known"]), 3)
            rows.append(row)
        print(f"T={T} r={r} done ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "envelope.csv"), index=False)
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
