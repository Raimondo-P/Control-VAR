#!/usr/bin/env python3
"""
CPO_K: conditional point-optimal test with the conditional coefficient fixed at its estimate from the
differenced data, compared with PO, CPO_C and the known-vector bound on the same designs as mc_cpo2.py.
Needs, in the same folder: mc_cpo2.py, mc_adaptive.py, mc_competitors.py, mc_bootstrap_cvar.py,
po_critical_values.py.

Model behind CPO_K (constant case, controls x, policy terms W):
    y_t = mu + b'x_t + tau'W_t + u_t,     (1 - rho L) u_t = omega' dx_t + eta_t,
so that, with kappa = b + omega,
    y_t - alpha y_{t-1} - kappa' dx_t = mu (1 - alpha) + b'[(x_t - alpha x_{t-1}) - dx_t] + tau'(W_t - alpha W_{t-1}) + eta_t
at alpha = rho, where (x_t - alpha x_{t-1}) - dx_t = (1 - alpha) x_{t-1}.
  Step 1: kappa_hat from the regression of dy_t on dx_t and dW_t (identified under the null and consistent
          under local alternatives: the rotation in which Elliott and Pesavento 2009 estimate the correlation).
  Step 2: S(alpha) = SSR of  y^alpha - kappa_hat' dx  on  [constant^alpha, x^alpha - dx, W^alpha],
          for alpha = alpha_bar and alpha = 1, first observation in levels (dx_0 = 0), as in ERS.
  Statistic: [S(alpha_bar) - alpha_bar S(1)] / omega2, omega2 Bartlett from the alpha = 1 residuals,
          cbar = -7 (the value of the paper), reject for SMALL values.
kappa is held fixed in both regressions, so no free coefficient on the lagged levels is added (the source
of the bias of CPO_C), while the correlation enters through kappa_hat. PO is the special case kappa = b.

Tests: PO, CPO_C (from mc_cpo2), CPO_K, MINP_K (smaller p-value of PO and CPO_K, p-values and critical
value from the delta = 0 null of the same design), ECR and WALD (reference), BOUND (triangular designs).
Designs (T = 200; g in {0, 2.5, 5, 7.5, 10, 15, 20, 30}):
  tri k = 3, delta in {0, 0.3, 0.6, 0.9};  tri k = 1, delta in {0, 0.6};
  cvar K = 3, r = 3 and K = 6, r = 5, delta in {0, 0.6}.
Reported: size-adjusted power (_sa) and actual rejection rates with critical values from the delta = 0
null of the same design.
Output in --out (default mc_cpok): one CSV per block, results.csv, power_curves.png.
Usage: py mc_cpok.py            (2000 replications per point)
"""
import argparse, os, time, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
import warnings; warnings.filterwarnings("ignore")
import mc_adaptive as A
import mc_competitors as C
import mc_cpo2 as Q
import po_critical_values as PC

GS = A.GS
CBAR = -7.0   # the value of the paper (Theorem po, ERS 1996); earlier versions used -12

def cpo_k(y, x, W, cbar=CBAR):
    """y (T,), x (T,k) controls in levels, W (T,q) policy terms or None."""
    T, k = x.shape; a = 1 + cbar / T
    const = np.ones((T, 1))
    Wm = np.zeros((T, 0)) if W is None else W
    D = np.vstack([np.zeros((1, k)), np.diff(x, axis=0)])                 # dx, with dx_0 = 0
    # step 1: kappa from the differenced regression (rows t >= 1)
    dy = np.diff(y); dW = np.diff(Wm, axis=0)
    Z0 = np.hstack([D[1:], dW])
    kappa = (np.linalg.pinv(Z0) @ dy)[:k]
    def S(alpha):
        Ya = A.qd(y, alpha) - D @ kappa
        Za = np.hstack([A.qd(const, alpha), A.qd(x, alpha) - D, A.qd(Wm, alpha)])
        return A.ssr(Ya, Za)
    S_a, _ = S(a)
    S_1, e1 = S(1.0)
    return (S_a - a * S_1) / A.bartlett(e1[1:])

def one(args):
    design, T, k, K, r, g, delta, seed = args
    rng = np.random.default_rng(seed); out = {}
    if design == "tri":
        y, x, beta, Sig = A.sim_tri(T, k, g, delta, rng)
        X = np.column_stack([np.ones(T), x]); W = None
        for gg in GS[1:]:
            out[f"NP_{gg}"] = A.np_bound_stat(y, x, beta, Sig, gg)
    else:
        y, x, W = A.sim_cvar(T, K, r, g, delta, rng)
        X = np.column_stack([np.ones(T), x, W])
    dx = np.vstack([np.full((1, x.shape[1]), np.nan), np.diff(x, axis=0)])
    out["PO"] = PC.po_stat(y, X, CBAR)
    out["CPO_C"] = Q.cpo_c(y, X, dx)
    out["CPO_K"] = cpo_k(y, x, W)
    out["ECR"], out["WALD"] = C.test_ecm(y, x, W=W, p=1 if design == "tri" else 4)
    return out

PTESTS = ["PO", "CPO_C", "CPO_K"]
SMALL = dict(PO=True, CPO_C=True, CPO_K=True, MINP_K=True, ECR=True, WALD=False)

def summarise(res, design, T, k, K, r, delta, ref):
    nul = {t: np.sort(ref[t].values) for t in PTESTS}
    def minp(df):
        return np.minimum(A.pval(df["PO"].values, nul["PO"]), A.pval(df["CPO_K"].values, nul["CPO_K"]))
    cv = {t: np.quantile(ref[t], 0.05) for t in PTESTS}
    cv["MINP_K"] = np.quantile(minp(ref), 0.05)
    for g in GS: res[g]["MINP_K"] = minp(res[g])
    null = res[0.0]; rows = []
    for g in GS:
        d = res[g]; row = dict(design=design, T=T, k=k, K=K, r=r, delta=delta, g=g)
        for t, small in SMALL.items():
            q = np.quantile(null[t], 0.05 if small else 0.95)
            row[t + "_sa"] = round(float(np.mean(d[t] < q) if small else np.mean(d[t] > q)), 3)
        for t in PTESTS + ["MINP_K"]:
            row[t] = round(float(np.mean(d[t] < cv[t])), 3)
        if design == "tri" and g > 0:
            col = f"NP_{g}"
            row["BOUND"] = round(float(np.mean(d[col] > np.quantile(null[col], 0.95))), 3)
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_cpok")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "block_*.csv")) + glob.glob(os.path.join(a.out, "results.csv")):
        os.remove(f)
    groups = [("tri", 200, 3, 3, 3, [0.0, 0.3, 0.6, 0.9]),
              ("tri", 200, 1, 1, 1, [0.0, 0.6]),
              ("cvar", 200, 3, 3, 3, [0.0, 0.6]),
              ("cvar", 200, 6, 6, 5, [0.0, 0.6])]
    t0 = time.time(); bi = 0
    with Pool(a.workers) as pool:
        for gi, (design, T, k, K, r, deltas) in enumerate(groups):
            ref = None
            for di, delta in enumerate(deltas):
                res = {}
                for gj, g in enumerate(GS):
                    args = [(design, T, k, K, r, g, delta, 1_100_000_000 + 100_000_000 * gi + 10_000_000 * di + 100_000 * gj + i)
                            for i in range(a.reps)]
                    res[g] = pd.DataFrame(pool.map(one, args, chunksize=max(1, a.reps // 64)))
                if delta == 0.0: ref = res[0.0].copy()
                pd.DataFrame(summarise(res, design, T, k, K, r, delta, ref)).to_csv(
                    os.path.join(a.out, f"block_{bi:02d}.csv"), index=False); bi += 1
                print(f"{design} k={k} K={K} delta={delta} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.out, "block_*.csv")))], ignore_index=True)
    df.to_csv(os.path.join(a.out, "results.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = df[["design", "k", "K", "delta"]].drop_duplicates().values.tolist()
    n = len(keys); cols = 5; rws = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rws, cols, figsize=(4.2 * cols, 3.4 * rws), sharey=True); ax = np.atleast_2d(ax)
    styles = dict(BOUND=("k-", "Bound (known vector)"), PO_sa=("b-o", "PO"), CPO_C_sa=("c:", "CPO_C"),
                  CPO_K_sa=("r-s", "CPO_K"), MINP_K_sa=("r--", "MINP_K"), WALD_sa=("y-", "Boswijk Wald"),
                  ECR_sa=("m:", "ECR"))
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
