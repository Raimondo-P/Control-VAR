#!/usr/bin/env python3
"""
Conditional and adaptive point-optimal tests for the CVAR design, against the error-correction
and system tests, with the known-vector Neyman-Pearson power bound.
Needs, in the same folder: mc_competitors.py, mc_bootstrap_cvar.py, po_critical_values.py.

Null of every test: no cointegration (the treated unit has a trend of its own; the CVAR is invalid).
All point-optimal statistics reject for SMALL values, with cbar = -7 (constant case), the value of the paper.

Tests
  PO      current CVAR point-optimal GLS statistic: quasi-differenced regression on the constant,
          the controls in levels and the policy terms; weights concentrated out under both
          hypotheses; Bartlett long-run variance from the regression in differences.
  CPO     conditional point-optimal statistic (error-correction rotation): the same regression plus
          the changes of the controls dx_t, dx_{t-1}, dx_{t+1} entered WITHOUT quasi-differencing,
          as covariates (Elliott and Jansson 2003; Hansen 1995). At alpha = 1 the quasi-differenced
          levels coincide with dx_t and the redundant columns drop out (least squares with pinv).
          It uses the correlation between the cycle's shocks and the controls' shocks.
  MINP    adaptive test: the smaller of the p-values of PO and CPO, each p-value from the null
          distribution of the same design with uncorrelated shocks (delta = 0); critical value of
          MINP from the same null.
  ECR, WALD   error-correction t-test and Boswijk Wald test (mc_competitors.test_ecm).
  PR_ADF  Perron-Rodriguez ADF-GLS (mc_competitors.test_pr), table critical values for m = 1, 3, 4, 5.
  JOH     Johansen maximum eigenvalue, tabulated critical value (statsmodels).
  EG      Engle-Granger, MacKinnon critical value.
  BOUND   known-vector Neyman-Pearson bound (Elliott, Jansson and Pesavento 2005): for each local
          alternative g, the most powerful test of rho = 1 against rho = 1 - g/T when the
          cointegrating vector and the covariance are known, invariant to the constant. No test can
          exceed it. Computed in the triangular design only.

Two kinds of rejection rates
  actual  with critical values from the delta = 0 null of the same design and number of controls,
          so that, for delta != 0, they show whether the null distribution moves with the correlation
          (PO, CPO, MINP); tabulated critical values for PR_ADF, JOH, EG.
  _sa     size-adjusted within each design (critical value from its own g = 0 replications).

Designs (T = 200)
  tri     triangular design with a constant, beta = 1/k, shocks of x and of the cycle with total
          correlation delta in {0, 0.3, 0.6, 0.9, -0.6}; k in {1, 3}; u_0 = 0.
  cvar    CVAR design of mc_bootstrap_cvar.py (K = 3, r = 3 and K = 6, r = 5; persistent idiosyncratic
          components, two policies with feedback) plus an extra component with root 1 - g/T whose
          innovations have correlation delta in {0, 0.6} with the average change of the controls.
Local alternatives g in {0, 2.5, 5, 7.5, 10, 15, 20, 30}.

Output in --out (default mc_adapt): one CSV per block, results.csv combining them, power_curves.png.
Usage: py mc_adaptive.py            (2000 replications per point)
"""
import argparse, os, time, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
import warnings; warnings.filterwarnings("ignore")
import mc_competitors as C
import mc_bootstrap_cvar as M
import po_critical_values as PC

GS = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
CBAR = -7.0   # the value of the paper (Theorem po, ERS 1996); earlier versions used -12
NLAG = 12

def bartlett(e):
    T = len(e); l = int(np.floor(4 * (T / 100) ** (2 / 9)))
    e = e - e.mean(); s = e @ e / T
    for j in range(1, l + 1): s += 2 * (1 - j / (l + 1)) * (e[j:] @ e[:-j]) / T
    return s

def qd(v, a):
    return np.r_[v[:1], v[1:] - a * v[:-1]] if v.ndim == 1 else np.vstack([v[:1], v[1:] - a * v[:-1]])

def ssr(Y, Z):
    b = np.linalg.pinv(Z) @ Y; e = Y - Z @ b
    return e @ e, e

def cpo_stat(y, X, dx, cbar=CBAR):
    """Conditional point-optimal statistic. X = [constant, levels of the controls, policy terms];
    dx = changes of the controls (T x K), entered with one lead and one lag, not quasi-differenced."""
    T = len(y); a = 1 + cbar / T
    L = np.column_stack([_lag(dx[:, k], j) for k in range(dx.shape[1]) for j in (-1, 0, 1)])
    ok = ~(np.isnan(L).any(1) | np.isnan(X).any(1)); ok[0] = False
    yy, XX, LL = y[ok], X[ok], L[ok]
    S_a, _ = ssr(qd(yy, a), np.column_stack([qd(XX, a), LL]))
    S_1, e1 = ssr(qd(yy, 1.0), np.column_stack([qd(XX, 1.0), LL]))     # first observation in levels, as in ERS
    return (S_a - a * S_1) / bartlett(e1[1:])

def _lag(v, j):
    z = np.full(len(v), np.nan)
    if j > 0: z[j:] = v[:-j]
    elif j < 0: z[:j] = v[-j:]
    else: z = v.copy()
    return z

def np_bound_stat(y, x, beta, Sig, g):
    """Known-vector Neyman-Pearson statistic (reject for LARGE values), invariant to the constant:
    [S(1) - S(rho_bar)] / Sigma_2.1, with S(rho) the GLS sum of squares of the conditional residuals
    c_t - rho c_{t-1} - (1 - rho) mu - omega' dx_t, mu concentrated out."""
    T, k = x.shape; rb = 1 - g / T
    S11 = Sig[:k, :k]; S12 = Sig[:k, k]; om = np.linalg.solve(S11, S12); s21 = Sig[k, k] - S12 @ om
    c = y - x @ beta; dx = np.vstack([np.zeros((1, k)), np.diff(x, axis=0)])
    def S(rho):
        q = np.r_[c[0], c[1:] - rho * c[:-1] - dx[1:] @ om]
        z = np.r_[1.0, np.full(T - 1, 1 - rho)]
        return q @ q - (z @ q) ** 2 / (z @ z)
    return (S(1.0) - S(rb)) / s21

def sim_tri(T, k, g, delta, rng):
    Sig = np.eye(k + 1); Sig[:k, k] = Sig[k, :k] = delta / np.sqrt(k)
    e = rng.multivariate_normal(np.zeros(k + 1), Sig, T)
    x = np.cumsum(e[:, :k], axis=0)
    u = lfilter([1.0], [1.0, -(1 - g / T)], e[:, k])          # u_0 = 0
    beta = np.ones(k) / k
    y = 0.5 + x @ beta + u
    return y, x, beta, Sig

def sim_cvar(T, K, r, g, delta, rng):
    d = dict(K=K, r=r, feedback=0.3, c_eta=0.5, theta0=0.2, constrained=True, boundary=False, lagrule="12")
    n = T + NLAG
    s = M.simulate(d, n, rng)
    dm = np.r_[0.0, np.diff(s["Y0"].mean(1))]; dm = (dm - dm.mean()) / dm.std()
    eps = delta * dm + np.sqrt(1 - delta ** 2) * rng.standard_normal(n)
    u = lfilter([1.0], [1.0, -(1 - g / T)], eps)
    W = PC.lagmat(np.column_stack([s["R"], s["W"]]), range(0, NLAG + 1))
    sl = slice(NLAG, n)
    return (s["Y1"] + u)[sl], s["Y0"][sl], W[sl]

def one(args):
    design, T, k, K, r, g, delta, seed = args
    rng = np.random.default_rng(seed); out = {}
    if design == "tri":
        y, x, beta, Sig = sim_tri(T, k, g, delta, rng)
        X = np.column_stack([np.ones(T), x]); W = None; m = k
        for gg in GS[1:]:
            out[f"NP_{gg}"] = np_bound_stat(y, x, beta, Sig, gg)
    else:
        y, x, W = sim_cvar(T, K, r, g, delta, rng)
        X = np.column_stack([np.ones(T), x, W]); m = K
    dx = np.vstack([np.full((1, x.shape[1]), np.nan), np.diff(x, axis=0)])
    out["PO"] = PC.po_stat(y, X, CBAR)
    out["CPO"] = cpo_stat(y, X, dx)
    out["PR_ADF"] = C.test_pr(y, x, m)[1]
    out["EG"] = C.test_eg(y, x)
    out["JOH"], out["JOH_cv"] = C.test_joh(y, x, k=1 if design == "tri" else 2)
    out["ECR"], out["WALD"] = C.test_ecm(y, x, W=W, p=1 if design == "tri" else 4)
    return out

SMALL = dict(PO=True, CPO=True, MINP=True, PR_ADF=True, EG=True, JOH=False, ECR=True, WALD=False)

def run_block(pool, design, T, k, K, r, delta, reps, seed0):
    res = {}
    for gi, g in enumerate(GS):
        args = [(design, T, k, K, r, g, delta, seed0 + 100_000 * gi + i) for i in range(reps)]
        res[g] = pd.DataFrame(pool.map(one, args, chunksize=max(1, reps // 64)))
    return res

def pval(stat, null_sorted):
    """Left-tail p-value of stat against a sorted null sample."""
    return (np.searchsorted(null_sorted, stat, side="right") + 1) / (len(null_sorted) + 1)

def summarise(res, design, T, k, K, r, delta, ref):
    """ref = null (g = 0, delta = 0) replications of the same design and number of controls."""
    m = k if design == "tri" else K
    nPO, nCPO = np.sort(ref["PO"].values), np.sort(ref["CPO"].values)
    ref_minp = np.minimum(pval(ref["PO"].values, nPO), pval(ref["CPO"].values, nCPO))
    cv = dict(PO=np.quantile(ref["PO"], 0.05), CPO=np.quantile(ref["CPO"], 0.05), MINP=np.quantile(ref_minp, 0.05))
    for g in GS:
        res[g]["MINP"] = np.minimum(pval(res[g]["PO"].values, nPO), pval(res[g]["CPO"].values, nCPO))
    null = res[0.0]; rows = []
    for g in GS:
        d = res[g]; row = dict(design=design, T=T, k=k, K=K, r=r, delta=delta, g=g)
        for t, small in SMALL.items():
            q = np.quantile(null[t], 0.05 if small else 0.95)
            row[t + "_sa"] = round(float(np.mean(d[t] < q) if small else np.mean(d[t] > q)), 3)
        for t in ["PO", "CPO", "MINP"]:
            row[t] = round(float(np.mean(d[t] < cv[t])), 3)
        cvpr = C.PR_CV_ADF.get(m)
        row["PR_ADF"] = round(float(np.mean(d.PR_ADF < cvpr)), 3) if cvpr is not None else np.nan
        row["JOH"] = round(float(np.mean(d.JOH > d.JOH_cv)), 3)
        row["EG"] = round(float(np.mean(d.EG < C.mackinnoncrit(N=min(m + 1, 6), regression="c", nobs=T)[1])), 3)
        if design == "tri" and g > 0:
            col = f"NP_{g}"
            row["BOUND"] = round(float(np.mean(d[col] > np.quantile(null[col], 0.95))), 3)
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_adapt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "block_*.csv")) + glob.glob(os.path.join(a.out, "results.csv")):
        os.remove(f)
    groups = [("tri", 200, 1, 1, 1, [0.0, 0.3, 0.6, 0.9, -0.6]),
              ("tri", 200, 3, 3, 3, [0.0, 0.3, 0.6, 0.9, -0.6]),
              ("cvar", 200, 3, 3, 3, [0.0, 0.6]),
              ("cvar", 200, 6, 6, 5, [0.0, 0.6])]
    t0 = time.time(); bi = 0
    with Pool(a.workers) as pool:
        for gi, (design, T, k, K, r, deltas) in enumerate(groups):
            ref = None
            for di, delta in enumerate(deltas):            # delta = 0 first: its null is the reference
                res = run_block(pool, design, T, k, K, r, delta, a.reps, 100_000_000 * (gi + 1) + 10_000_000 * di)
                if delta == 0.0: ref = res[0.0].copy()
                rows = summarise(res, design, T, k, K, r, delta, ref)
                pd.DataFrame(rows).to_csv(os.path.join(a.out, f"block_{bi:02d}.csv"), index=False); bi += 1
                print(f"{design} k={k} K={K} delta={delta} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.out, "block_*.csv")))], ignore_index=True)
    df.to_csv(os.path.join(a.out, "results.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = df[["design", "k", "K", "delta"]].drop_duplicates().values.tolist()
    n = len(keys); cols = 4; rws = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rws, cols, figsize=(4.2 * cols, 3.4 * rws), sharey=True); ax = np.atleast_2d(ax)
    styles = dict(BOUND=("k-", "Bound (known vector)"), PO_sa=("b-o", "PO"), CPO_sa=("c-s", "CPO"),
                  MINP_sa=("k--", "MINP (adaptive)"), ECR_sa=("m:", "ECR"), WALD_sa=("y-", "Boswijk Wald"),
                  PR_ADF_sa=("g:", "PR ADF-GLS"), JOH_sa=("r--", "Johansen"))
    for i, key in enumerate(keys):
        a_ = ax.flat[i]
        x = df[(df.design == key[0]) & (df.k == key[1]) & (df.K == key[2]) & (df.delta == key[3])]
        for c, (st, lab) in styles.items():
            if c in x and x[c].notna().any(): a_.plot(x.g, x[c], st, ms=3, lw=1.2, label=lab)
        a_.axhline(0.05, color="grey", lw=0.5)
        a_.set_title(f"{key[0]} k={int(key[1])} K={int(key[2])} delta={key[3]}", fontsize=9)
    for j in range(n, rws * cols): ax.flat[j].axis("off")
    ax.flat[0].legend(fontsize=6); fig.supxlabel("g = T(1 - rho)"); fig.supylabel("Size-adjusted rejection rate")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "power_curves.png"), dpi=130)
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
