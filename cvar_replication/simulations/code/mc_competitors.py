#!/usr/bin/env python3
"""
Power comparison of the CVAR point-optimal GLS test with the tests of the literature on the
null of no cointegration. Needs mc_bootstrap_cvar.py and po_critical_values.py in the same folder.

All tests have the null of NO cointegration (the treated unit has a stochastic trend of its own,
the CVAR is invalid), except KPSS_HT, whose null is stationarity of the contrast.

Tests
  PO      CVAR point-optimal GLS test: [S(a) - a S(1)] / omega2, the weights concentrated out of
          the quasi-differenced regression under both the null and the alternative
          (cbar = -12 with a constant, -7 without deterministic terms).
  PR_MPT, PR_ADF   Perron and Rodriguez (2016, EJ): each series GLS-detrended separately with
          their cbar for m regressors (Table 1), OLS residual of the detrended data, then the
          Ng-Perron MPT statistic and the ADF t-statistic (MAIC lags) on the residual.
  EG      Engle-Granger ADF on the OLS residual (BIC lags).
  JOH     Johansen maximum eigenvalue test of rank 0 against rank 1.
  ECR     t-statistic on y_{t-1} in the conditional error correction regression
          (Banerjee, Dolado and Mestre 1998; Kremers, Ericsson and Dolado 1992).
  WALD    Boswijk (1994) Wald statistic on (y_{t-1}, x_{t-1}) in the same regression.
  KPSS_HT Harvey and Thiele (2021): KPSS on the contrast with restricted least squares weights
          (summing to one); null = stationarity. Reported as its own rejection rate.
  MAX     Elliott and Pesavento (2009) weighted-average power bound, eq. (10): infeasible
          (true covariance), no deterministic terms, normal weights N(b0, h^-1 ...) on the
          cointegrating vector; computed only in the design "ep", where beta is drawn from the
          same weighting in every replication, as in their Section 4.

Designs
  ep      their design: k = 1, no deterministic terms, iid errors, corr(e1, e2) = delta in
          {0, -0.5, 0.5}, h in {1, 10}, b0 = 1, T = 1000 (approximating the limit) and T = 200.
  pes     triangular design with a constant: k in {1, 3}, R2 = delta^2 with delta in {0, 0.6, 0.9}
          (Pesavento's nuisance parameter), T = 200.
  cvar    the CVAR design of mc_bootstrap_cvar.py (persistent idiosyncratic components, two
          policies with feedback), K = 3, r = 3 and K = 6, r = 5, T = 200.
Local alternatives rho = 1 - g/T, g in {0, 2.5, 5, 7.5, 10, 15, 20, 30}.

Reported: size-adjusted power for every test (critical values from the g = 0 replications of
the same design), and actual size and power for the tests with tables: PR (their Table 2, demeaned
case, m = 1, 3, 4, 5), EG (MacKinnon), JOH (Osterwald-Lenum via statsmodels), KPSS_HT (0.463).
PO (cbar = -12) has no table here: its size is handled by the sieve bootstrap of robust_po.py.
A separate block "nullcv" checks the PR implementation against their published 5% critical values
and reports how the null 5% quantile of PO and of PR_MPT changes with the number of regressors.

Output in --out (default mc_comp): results.csv and nullcv.csv, appended as each block finishes,
and power_curves.png at the end.
Usage:  python mc_competitors.py            (2000 replications per point)
"""
import argparse, os, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.adfvalues import mackinnoncrit
import warnings; warnings.filterwarnings("ignore")
import mc_bootstrap_cvar as M
import po_critical_values as PC

GS = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
NLAG = 12
# Perron-Rodriguez (2016), Table 1 (demeaned case px = py = 0): cbar by m
PR_CBAR = {1: -13.75, 2: -18.25, 3: -22.25, 4: -26.25, 5: -30.00}
# Their Table 2, 5% critical values (m = 2 entries in the published table appear misprinted; not used)
PR_CV_MPT = {1: 6.325, 3: 8.923, 4: 10.556, 5: 11.656}
PR_CV_ADF = {1: -2.736, 3: -3.710, 4: -4.028, 5: -4.366}

# ----------------------------------------------------------------- building blocks
def lag1(v):
    return np.r_[0.0, v[:-1]] if v.ndim == 1 else np.vstack([np.zeros((1, v.shape[1])), v[:-1]])

def ols(Y, X):
    b = np.linalg.lstsq(X, Y, rcond=None)[0]; return b, Y - X @ b

def adf_maic(u, pmax=8, const=False):
    """ADF regression on u (no deterministic terms unless const), lag by Ng-Perron MAIC.
    Returns (t-statistic, AR spectral long-run variance)."""
    du = np.diff(u); n = len(du) - pmax; best = None
    for p in range(0, pmax + 1):
        Y = du[pmax:]
        Z = [u[pmax:-1]] + [du[pmax - j:len(du) - j] for j in range(1, p + 1)]
        if const: Z.append(np.ones(len(Y)))
        Z = np.column_stack(Z); b, e = ols(Y, Z); s2 = e @ e / n
        tau = b[0] ** 2 * (u[pmax:-1] @ u[pmax:-1]) / s2
        crit = np.log(s2) + 2 * (tau + p) / n
        if best is None or crit < best[0]:
            V = (e @ e / (n - Z.shape[1])) * np.linalg.inv(Z.T @ Z)
            lrv = s2 / (1 - b[1:1 + p].sum()) ** 2 if p > 0 else s2
            best = (crit, b[0] / np.sqrt(V[0, 0]), lrv)
    return best[1], best[2]

def adf_bic(u, pmax=8, const=True):
    du = np.diff(u); best = None
    for p in range(0, pmax + 1):
        Y = du[pmax:]
        Z = [u[pmax:-1]] + [du[pmax - j:len(du) - j] for j in range(1, p + 1)]
        if const: Z.append(np.ones(len(Y)))
        Z = np.column_stack(Z); b, e = ols(Y, Z); n = len(Y)
        bic = np.log(e @ e / n) + Z.shape[1] * np.log(n) / n
        if best is None or bic < best[0]:
            V = (e @ e / (n - Z.shape[1])) * np.linalg.inv(Z.T @ Z); best = (bic, b[0] / np.sqrt(V[0, 0]))
    return best[1]

def gls_demean(v, cbar):
    T = len(v); a = 1 + cbar / T
    vd = np.r_[v[0], v[1:] - a * v[:-1]]; zd = np.r_[1.0, np.full(T - 1, 1 - a)]
    return v - (zd @ vd) / (zd @ zd)

# ----------------------------------------------------------------- tests
def test_po(y, X, cbar):
    """CVAR point-optimal GLS statistic (X includes the constant if any)."""
    return PC.po_stat(y, X, cbar)

def test_pr(y, x, m, const=True):
    """Perron-Rodriguez: GLS-detrend every series separately, OLS residual, MPT and ADF (MAIC)."""
    cbar = PR_CBAR.get(m, -30.0 - 3.75 * (m - 5))
    if const:
        yd = gls_demean(y, cbar); xd = np.column_stack([gls_demean(x[:, j], cbar) for j in range(x.shape[1])])
    else:
        yd, xd = y, x
    _, u = ols(yd, xd); T = len(u)
    t, s2 = adf_maic(u)
    mpt = (cbar ** 2 * (u[:-1] @ u[:-1]) / T ** 2 - cbar * u[-1] ** 2 / T) / s2
    return mpt, t

def test_eg(y, x, const=True):
    X = np.column_stack([np.ones(len(y)), x]) if const else x
    _, u = ols(y, X); return adf_bic(u, const=False)

def test_joh(y, x, const=True, k=1):
    j = coint_johansen(np.column_stack([y, x]), 0 if const else -1, k)
    return j.lr2[0], j.cvm[0, 1]

def test_ecm(y, x, W=None, const=True, p=1):
    """Conditional ECM: dy_t on [1], y_{t-1}, x_{t-1}, dx_t, p lags of dy and dx, policy terms."""
    dy = np.diff(y); dx = np.diff(x, axis=0); T = len(dy)
    cols = [y[:-1][:, None], x[:-1]]
    cols.append(dx)
    for j in range(1, p + 1):
        cols.append(np.r_[np.full(j, np.nan), dy[:-j]][:, None]); cols.append(np.vstack([np.full((j, dx.shape[1]), np.nan), dx[:-j]]))
    if W is not None: cols.append(W[1:])
    if const: cols.append(np.ones((T, 1)))
    Z = np.hstack(cols); ok = ~np.isnan(Z).any(1)
    b, e = ols(dy[ok], Z[ok]); n, q = Z[ok].shape
    V = (e @ e / (n - q)) * np.linalg.inv(Z[ok].T @ Z[ok]); kk = 1 + x.shape[1]
    t = b[0] / np.sqrt(V[0, 0]); th = b[:kk]; wald = th @ np.linalg.solve(V[:kk, :kk], th)
    return t, wald

def test_kpss_ht(y, x):
    """Harvey-Thiele: restricted least squares weights (sum to one), KPSS on the contrast."""
    K = x.shape[1]; T = len(y)
    if K == 1:
        c = y - x[:, 0]
    else:
        Z = np.column_stack([np.ones(T), x[:, :-1] - x[:, [-1]]])
        b, _ = ols(y - x[:, -1], Z); w = np.r_[b[1:], 1 - b[1:].sum()]; c = y - x @ w
    e = c - c.mean(); S = np.cumsum(e); l = int(np.floor(4 * (T / 100) ** 0.25))
    s = e @ e / T
    for j in range(1, l + 1): s += 2 * (1 - j / (l + 1)) * (e[j:] @ e[:-j]) / T
    return (S @ S) / (T ** 2 * s)

def ep_stat(y, x, Sig, g, b0, h):
    """Elliott and Pesavento (2009), eq. (10), no deterministic terms, Sigma known."""
    T, k = x.shape
    S11, S12, S22 = Sig[:k, :k], Sig[:k, k], Sig[k, k]
    S11i = np.linalg.inv(S11); S21_ = S22 - S12 @ S11i @ S12
    L = np.linalg.cholesky(S11); S11mh = np.linalg.inv(L)          # Sigma11^{-1/2} (lower-triangular root)
    yt = y - x @ (S11i @ S12)
    ym1, xm1 = lag1(y), lag1(x)
    dyt = np.r_[yt[0], np.diff(yt)]                                   # A(1) applied to y-tilde
    bt0 = (S21_ ** -0.5) * (L.T @ b0)                                 # Sigma21^{-1/2} Sigma11^{1/2'} b0
    a = g ** 2 * (ym1 @ ym1) / T ** 2 + 2 * g * (ym1 @ dyt) / T
    v = g ** 2 * (ym1 @ xm1) / T ** 2 + g * (dyt @ xm1) / T
    Mx = (g ** 2 + h) * (xm1.T @ xm1) / T ** 2
    xs = xm1 @ S11mh.T                                                # x_{-1} Sigma11^{-1/2'}
    c3 = g ** 2 * (h / (g ** 2 + h)) * (bt0 @ (xs.T @ xs) @ bt0) / T ** 2
    c4 = 2 * S21_ ** -0.5 * (h / (g ** 2 + h)) * (g ** 2 * (ym1 @ xs @ bt0) / T ** 2 + g * (dyt @ xs @ bt0) / T)
    return -a / S21_ + (v @ np.linalg.solve(Mx, v)) / S21_ - c3 + c4

# ----------------------------------------------------------------- designs
def sim_triangular(T, k, g, delta, const, rng, beta=None, burn=50):
    n = T + burn
    Sig = np.eye(k + 1); Sig[:k, k] = Sig[k, :k] = delta / np.sqrt(k)   # total correlation delta spread over x
    e = rng.multivariate_normal(np.zeros(k + 1), Sig, n)
    x = np.cumsum(e[:, :k], axis=0)
    u = lfilter([1.0], [1.0, -(1 - g / T)], e[:, k])
    if beta is None: beta = np.ones(k) / k
    y = (0.5 if const else 0.0) + x @ beta + u
    return y[burn:], x[burn:] - x[burn], Sig

def one(args):
    design, T, k, K, r, g, delta, h, seed = args
    rng = np.random.default_rng(seed); out = {}
    if design == "ep":
        # beta drawn from the Elliott-Pesavento weighting, centred at b0 = 1
        y0, x, Sig = sim_triangular(T, 1, g, delta, False, rng, beta=np.ones(1))
        s21 = Sig[1, 1] - Sig[0, 1] ** 2
        beta = 1.0 + rng.standard_normal() * np.sqrt(s21 / (h * (lag1(x[:, 0]) @ lag1(x[:, 0]) / T ** 2)))
        y = y0 + (beta - 1.0) * x[:, 0]
        for gg in GS[1:]:
            out[f"MAX_{gg}"] = ep_stat(y, x, Sig, gg, np.ones(1), h)
        out["PO"] = test_po(y, x, -7.0)
        out["PR_MPT"], out["PR_ADF"] = test_pr(y, x, 1, const=False)
        out["EG"] = test_eg(y, x, const=False)
        out["JOH"], out["JOH_cv"] = test_joh(y, x, const=False)
        out["ECR"], out["WALD"] = test_ecm(y, x, const=False)
        return out
    if design == "pes":
        y, x, Sig = sim_triangular(T, k, g, delta, True, rng)
        X = np.column_stack([np.ones(T), x]); m = k; W = None
    else:
        d = dict(K=K, r=r, feedback=0.3, c_eta=0.5, theta0=0.2, constrained=True, boundary=False, lagrule="12")
        s = M.simulate(d, T + NLAG, rng)
        u = lfilter([1.0], [1.0, -(1 - g / T)], rng.standard_normal(T + NLAG))
        W = PC.lagmat(np.column_stack([s["R"], s["W"]]), range(0, NLAG + 1))
        sl = slice(NLAG, T + NLAG)
        y = (s["Y1"] + u)[sl]; x = s["Y0"][sl]; W = W[sl]
        X = np.column_stack([np.ones(T), x, W]); m = K
    out["PO"] = test_po(y, X, -12.0)
    out["PR_MPT"], out["PR_ADF"] = test_pr(y, x, m)
    out["EG"] = test_eg(y, x)
    out["JOH"], out["JOH_cv"] = test_joh(y, x, k=1 if design == "pes" else 2)
    out["ECR"], out["WALD"] = test_ecm(y, x, W=W if design == "cvar" else None, p=1 if design == "pes" else 4)
    out["KPSS_HT"] = test_kpss_ht(y, x)
    return out

# direction of rejection: True = reject for small values
SMALL = dict(PO=True, PR_MPT=True, PR_ADF=True, EG=True, JOH=False, ECR=True, WALD=False, KPSS_HT=False)

def actual_cv(test, design, T, m):
    # PO uses cbar = -12 here, for which the response surface of po_critical_values.py (cbar = -7) does not
    # apply: its size is controlled by the sieve bootstrap (robust_po.py); the null quantiles by k are in nullcv.csv
    if test == "PR_MPT" and design != "ep": return PR_CV_MPT.get(m)
    if test == "PR_ADF" and design != "ep": return PR_CV_ADF.get(m)
    if test == "EG":
        # MacKinnon's tables cover cointegration residuals only with deterministic terms; in the design
        # without them ("ep") Engle-Granger is reported size-adjusted only
        if design == "ep": return None
        return mackinnoncrit(N=min(m + 1, 6), regression="c", nobs=T)[1]
    if test == "KPSS_HT": return 0.463
    return None

def block(pool, design, T, k, K, r, delta, h, reps, seed0):
    res = {}
    for gi, g in enumerate(GS):
        args = [(design, T, k, K, r, g, delta, h, seed0 + 100_000 * gi + i) for i in range(reps)]
        res[g] = pd.DataFrame(pool.map(one, args, chunksize=max(1, reps // 64)))
    null = res[0.0]; m = k if design != "cvar" else K
    rows = []
    for g in GS:
        d = res[g]; row = dict(design=design, T=T, k=k, K=K, r=r, delta=delta, h=h, g=g)
        for t in SMALL:
            if t not in d: continue
            q = np.quantile(null[t], 0.05 if SMALL[t] else 0.95)
            row[t + "_sa"] = round(float(np.mean(d[t] < q) if SMALL[t] else np.mean(d[t] > q)), 3)
            cv = actual_cv(t, design, T, m)
            if t == "JOH":
                row["JOH"] = round(float(np.mean(d.JOH > d.JOH_cv)), 3)
            elif cv is not None:
                row[t] = round(float(np.mean(d[t] < cv) if SMALL[t] else np.mean(d[t] > cv)), 3)
        if design == "ep" and g > 0:
            col = f"MAX_{g}"
            row["MAX"] = round(float(np.mean(d[col] > np.quantile(null[col], 0.95))), 3)
        rows.append(row)
    return rows

def nullcv(pool, reps, T=500):
    """Null 5% quantiles of PO and PR_MPT by number of regressors (pes design, R2 = 0)."""
    rows = []
    for k in [1, 2, 3, 4, 5]:
        d = pd.DataFrame(pool.map(one, [("pes", T, k, k, k, 0.0, 0.0, 0, 9_000_000 + 10_000 * k + i) for i in range(reps)]))
        rows.append(dict(k=k, PO_q5=round(float(np.quantile(d.PO, 0.05)), 3),
                         PR_MPT_q5=round(float(np.quantile(d.PR_MPT, 0.05)), 3), PR_MPT_table=PR_CV_MPT.get(k),
                         PR_ADF_q5=round(float(np.quantile(d.PR_ADF, 0.05)), 3), PR_ADF_table=PR_CV_ADF.get(k)))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_comp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.csv")
    import glob
    for f in ["results.csv", "nullcv.csv"] + [os.path.basename(p) for p in glob.glob(os.path.join(a.out, "block_*.csv"))]:
        if os.path.exists(os.path.join(a.out, f)): os.remove(os.path.join(a.out, f))
    t0 = time.time()
    plan = ([("ep", T, 1, 1, 1, dl, h) for T in [1000, 200] for dl in [0.0, -0.5, 0.5] for h in [1, 10]] +
            [("pes", 200, k, k, k, dl, 0) for k in [1, 3] for dl in [0.0, 0.6, 0.9]] +
            [("cvar", 200, K, K, r, 0.0, 0) for K, r in [(3, 3), (6, 5)]])
    with Pool(a.workers) as pool:
        pd.DataFrame(nullcv(pool, max(a.reps, 5000))).to_csv(os.path.join(a.out, "nullcv.csv"), index=False)
        print(f"nullcv done ({time.time()-t0:.0f}s)", flush=True)
        for bi, (design, T, k, K, r, dl, h) in enumerate(plan):
            rows = block(pool, design, T, k, K, r, dl, h, a.reps, 20_000_000 * (bi + 1))
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"block_{bi:02d}.csv"), index=False)
            print(f"{design} T={T} k={k} K={K} delta={dl} h={h} done ({time.time()-t0:.0f}s)", flush=True)
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import glob
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.out, "block_*.csv")))], ignore_index=True)
    df.to_csv(path, index=False); keys = df[["design", "T", "k", "K", "delta", "h"]].drop_duplicates().values.tolist()
    n = len(keys); cols = 6; rws = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rws, cols, figsize=(4 * cols, 3.4 * rws), sharey=True); ax = np.atleast_2d(ax)
    styles = dict(MAX=("k-", "Bound (EP 2009)"), PO_sa=("b-o", "CVAR PO"), PR_MPT_sa=("g--", "PR MPT-GLS"),
                  PR_ADF_sa=("g:", "PR ADF-GLS"), EG_sa=("r:", "Engle-Granger"), JOH_sa=("m--", "Johansen"),
                  ECR_sa=("c--", "ECR"), WALD_sa=("y-", "Boswijk Wald"))
    for i, key in enumerate(keys):
        a_ = ax.flat[i]
        x = df[(df.design == key[0]) & (df["T"] == key[1]) & (df.k == key[2]) & (df.K == key[3]) & (df.delta == key[4]) & (df.h == key[5])]
        for c, (st, lab) in styles.items():
            if c in x and x[c].notna().any(): a_.plot(x.g, x[c], st, ms=3, lw=1.2, label=lab)
        a_.axhline(0.05, color="grey", lw=0.5)
        a_.set_title(f"{key[0]} T={key[1]} k={key[2]} K={key[3]} d={key[4]} h={key[5]}", fontsize=8)
    for j in range(n, rws * cols): ax.flat[j].axis("off")
    ax.flat[0].legend(fontsize=6); fig.supxlabel("g = T(1 - rho)"); fig.supylabel("Size-adjusted rejection rate")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "power_curves.png"), dpi=130)
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
