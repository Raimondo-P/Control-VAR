#!/usr/bin/env python3
"""
Two repaired conditional point-optimal tests, on the designs where the first conditional test failed.
Needs, in the same folder: mc_adaptive.py, mc_competitors.py, mc_bootstrap_cvar.py, po_critical_values.py.

All point-optimal statistics: [S(a) - a S(1)] / omega2, cbar = -7 (the value of the paper), reject for SMALL values.
  PO      current test (no covariates).
  CPO     first conditional test (dx_t, dx_{t-1}, dx_{t+1} as covariates), for reference.
  CPO_C   contemporaneous change only: dx_t enters both regressions, not quasi-differenced.
          At alpha = 1 it coincides with the quasi-differenced levels and drops out, so the two
          regressions differ by k columns instead of 3k.
  CPO_R   correlated part removed before quasi-differencing, with the same columns in both regressions:
          (1) weights from the PO regression at alpha_bar; (2) omega_hat from the regression of the
          quasi-differenced PO residual on dx_t; (3) y_t - alpha y_{t-1} - omega_hat' dx_t regressed on
          the quasi-differenced constant, controls and policy terms, at alpha_bar and at 1.
  MINP_C, MINP_R   adaptive tests: the smaller p-value of PO and CPO_C (resp. CPO_R), p-values and
          critical value from the delta = 0 null of the same design.
  ECR, WALD   error-correction t-test and Boswijk Wald test, for reference.
  BOUND   known-vector Neyman-Pearson bound (triangular design).

Designs (T = 200; g in {0, 2.5, 5, 7.5, 10, 15, 20, 30})
  tri  k = 3, delta in {0, 0.3, 0.6, 0.9};  tri  k = 1, delta in {0, 0.6} (reference)
  cvar K = 3, r = 3 and K = 6, r = 5, delta in {0, 0.6}
Reported: size-adjusted power (_sa) and actual rejection rates with critical values from the delta = 0
null of the same design (shows whether the null distribution moves with the correlation).
Output in --out (default mc_cpo2): one CSV per block, results.csv, power_curves.png.
Usage: py mc_cpo2.py            (2000 replications per point)
"""
import argparse, os, time, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
import warnings; warnings.filterwarnings("ignore")
import mc_adaptive as A
import mc_competitors as C
import po_critical_values as PC

GS = A.GS
CBAR = -7.0   # the value of the paper (Theorem po, ERS 1996); earlier versions used -12

def cpo_c(y, X, dx1, cbar=CBAR):
    """dx1: contemporaneous change of the controls (T x k), first row NaN."""
    T = len(y); a = 1 + cbar / T
    ok = ~(np.isnan(dx1).any(1) | np.isnan(X).any(1))
    yy, XX, DD = y[ok], X[ok], dx1[ok]
    S_a, _ = A.ssr(A.qd(yy, a), np.column_stack([A.qd(XX, a), DD]))
    S_1, e1 = A.ssr(A.qd(yy, 1.0), np.column_stack([A.qd(XX, 1.0), DD]))
    return (S_a - a * S_1) / A.bartlett(e1[1:])

def cpo_r(y, X, dx1, cbar=CBAR):
    T = len(y); a = 1 + cbar / T
    ok = ~(np.isnan(dx1).any(1) | np.isnan(X).any(1))
    yy, XX, DD = y[ok], X[ok], dx1[ok]
    ya, Xa = A.qd(yy, a), A.qd(XX, a)
    b = np.linalg.pinv(Xa) @ ya                          # (1) weights from the PO regression at alpha_bar
    ra = ya - Xa @ b                                     # quasi-differenced PO residual
    om = np.linalg.pinv(DD[1:]) @ ra[1:]                 # (2) correlated part
    corr = DD @ om; corr[0] = 0.0                        # first observation kept in levels
    S_a, _ = A.ssr(ya - corr, Xa)                        # (3) same columns at alpha_bar and at 1
    S_1, e1 = A.ssr(A.qd(yy, 1.0) - corr, A.qd(XX, 1.0))
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
    out["CPO"] = A.cpo_stat(y, X, dx)
    out["CPO_C"] = cpo_c(y, X, dx)
    out["CPO_R"] = cpo_r(y, X, dx)
    out["ECR"], out["WALD"] = C.test_ecm(y, x, W=W, p=1 if design == "tri" else 4)
    return out

SMALL = dict(PO=True, CPO=True, CPO_C=True, CPO_R=True, MINP_C=True, MINP_R=True, ECR=True, WALD=False)

def summarise(res, design, T, k, K, r, delta, ref):
    nul = {t: np.sort(ref[t].values) for t in ["PO", "CPO_C", "CPO_R"]}
    def minp(df, t2):
        return np.minimum(A.pval(df["PO"].values, nul["PO"]), A.pval(df[t2].values, nul[t2]))
    refm = {"MINP_C": minp(ref, "CPO_C"), "MINP_R": minp(ref, "CPO_R")}
    cv = {t: np.quantile(ref[t], 0.05) for t in ["PO", "CPO", "CPO_C", "CPO_R"]}
    cv.update({t: np.quantile(v, 0.05) for t, v in refm.items()})
    for g in GS:
        res[g]["MINP_C"] = minp(res[g], "CPO_C"); res[g]["MINP_R"] = minp(res[g], "CPO_R")
    null = res[0.0]; rows = []
    for g in GS:
        d = res[g]; row = dict(design=design, T=T, k=k, K=K, r=r, delta=delta, g=g)
        for t, small in SMALL.items():
            q = np.quantile(null[t], 0.05 if small else 0.95)
            row[t + "_sa"] = round(float(np.mean(d[t] < q) if small else np.mean(d[t] > q)), 3)
        for t in ["PO", "CPO", "CPO_C", "CPO_R", "MINP_C", "MINP_R"]:
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
    ap.add_argument("--out", default="mc_cpo2")
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
                    args = [(design, T, k, K, r, g, delta, 300_000_000 + 100_000_000 * gi + 10_000_000 * di + 100_000 * gj + i)
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
    styles = dict(BOUND=("k-", "Bound (known vector)"), PO_sa=("b-o", "PO"), CPO_sa=("c:", "CPO (old)"),
                  CPO_C_sa=("c-s", "CPO_C"), CPO_R_sa=("g-^", "CPO_R"), MINP_C_sa=("k--", "MINP_C"),
                  MINP_R_sa=("g--", "MINP_R"), WALD_sa=("y-", "Boswijk Wald"), ECR_sa=("m:", "ECR"))
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
