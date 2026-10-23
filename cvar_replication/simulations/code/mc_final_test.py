#!/usr/bin/env python3
"""
Final Monte Carlo for the test of the CVAR design (Section "Testing the design" and the supplementary
appendix). Needs, in the same folder: mc_cpok.py, mc_cpo2.py, mc_adaptive.py, mc_competitors.py,
mc_bootstrap_cvar.py, po_critical_values.py, robust_po.py.

Null of every test: no cointegration (the treated unit has a trend of its own; the CVAR is not valid),
except KPSS_HT, whose null is stationarity of the contrast.

Tests
  PO      the test of the paper: [S(a) - a S(1)] / omega2 with cbar = -7, omega2 from the ADF regression on
          the GLS-detrended residual with the Ng-Perron MAIC lag, critical value of ERS (1996) with a
          constant (5%: 3.26), as Theorem po implies.
  PO_BOOT the same statistic with the sieve-bootstrap p-value of Proposition sieve: AR(12) for the
          residuals of the alpha = 1 regression, controls and policy terms held fixed, B draws.
  MINP_K  robust alternative of the supplementary appendix (smaller p-value of PO and CPO_K; critical
          value from the delta = 0 null of the same design).
  PR_ADF  Perron-Rodriguez (2016) ADF-GLS, their Table 2 critical values (m = 1, 3, 4, 5).
  EG      Engle-Granger, MacKinnon critical values.
  JOH     Johansen maximum eigenvalue, tabulated critical values.
  ECR, WALD  error-correction t-test and Boswijk Wald test (size-adjusted only: no tables in this setting).
  KPSS_HT Harvey-Thiele KPSS on the restricted least squares contrast (null: stationarity), 5%: 0.463.
  BOUND   known-vector Neyman-Pearson bound (triangular designs).

Designs: T in {200, 400};
  tri  k = 1 and k = 3, delta in {0, 0.6}   (delta = correlation of the cycle's shocks with the controls')
  cvar K = 3, r = 3 and K = 6, r = 5, delta in {0, 0.6}
Local alternatives g in {0, 2.5, 5, 7.5, 10, 15, 20, 30}; the bootstrap is computed at g in BOOT_GS only.

Outputs in --out (default mc_final):
  block_XX.csv        one per design
  results.csv         all designs: actual rejection rates (columns without suffix) and size-adjusted power (_sa)
  size_table.csv      actual size (g = 0) of every test in every design
  power_table.csv     size-adjusted power at g = 5, 10, 20
  power_curves_T200.png, power_curves_T400.png   figures for the supplementary appendix
Usage: py mc_final_test.py                      (1000 replications, 199 bootstrap draws)
       py mc_final_test.py --reps 2000 --boot 399
"""
import argparse, os, time, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
import warnings; warnings.filterwarnings("ignore")
import mc_adaptive as A
import mc_competitors as C
import mc_cpok as K_
import robust_po as RP

GS = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
BOOT_GS = [0.0, 5.0, 10.0, 20.0]
CBAR = -7.0
ERS_CV5 = 3.26          # ERS (1996), Table 1, constant, asymptotic 5% critical value of P_T with cbar = -7
P_SIEVE = 12

def po(y, X):
    return RP.po_maic(y, X, CBAR)

def boot_p(y, X, B, rng):
    """Sieve bootstrap p-value of PO under the null with an AR(P_SIEVE) for the alpha = 1 residuals."""
    stat = po(y, X)
    dy = np.diff(y); dX = np.diff(X, axis=0); keep = np.abs(dX).sum(0) > 1e-12
    Z = np.column_stack([np.ones(len(dy)), dX[:, keep]])
    g = np.linalg.lstsq(Z, dy, rcond=None)[0]; fit = Z @ g; e = dy - fit
    p = P_SIEVE
    Zp = np.column_stack([e[p - j:len(e) - j] for j in range(1, p + 1)])
    phi = np.linalg.lstsq(Zp, e[p:], rcond=None)[0]; res = e[p:] - Zp @ phi; res = res - res.mean()
    cnt = 0
    for _ in range(B):
        es = lfilter([1.0], np.r_[1.0, -phi], res[rng.integers(0, len(res), len(e) + 100)])[100:]
        ys = np.r_[y[0], y[0] + np.cumsum(fit + es)]
        cnt += po(ys, X) <= stat
    return (cnt + 1) / (B + 1)

def one(args):
    design, T, k, K, r, g, delta, seed, B = args
    rng = np.random.default_rng(seed); out = {}
    if design == "tri":
        y, x, beta, Sig = A.sim_tri(T, k, g, delta, rng)
        X = np.column_stack([np.ones(T), x]); W = None; m = k
        for gg in GS[1:]:
            out[f"NP_{gg}"] = A.np_bound_stat(y, x, beta, Sig, gg)
    else:
        y, x, W = A.sim_cvar(T, K, r, g, delta, rng)
        X = np.column_stack([np.ones(T), x, W]); m = K
    out["PO"] = po(y, X)
    out["CPO_K"] = K_.cpo_k(y, x, W)
    if B > 0: out["PO_BOOT_p"] = boot_p(y, X, B, rng)
    out["PR_ADF"] = C.test_pr(y, x, m)[1]
    out["EG"] = C.test_eg(y, x)
    out["JOH"], out["JOH_cv"] = C.test_joh(y, x, k=1 if design == "tri" else 2)
    out["ECR"], out["WALD"] = C.test_ecm(y, x, W=W, p=1 if design == "tri" else 4)
    out["KPSS_HT"] = C.test_kpss_ht(y, x)
    return out

SMALL = dict(PO=True, MINP_K=True, PR_ADF=True, EG=True, JOH=False, ECR=True, WALD=False, KPSS_HT=False)

def summarise(res, design, T, k, K, r, delta, ref):
    m = k if design == "tri" else K
    nPO, nK = np.sort(ref["PO"].values), np.sort(ref["CPO_K"].values)
    def minp(df): return np.minimum(A.pval(df["PO"].values, nPO), A.pval(df["CPO_K"].values, nK))
    cv_minp = np.quantile(minp(ref), 0.05)
    for g in GS: res[g]["MINP_K"] = minp(res[g])
    null = res[0.0]; rows = []
    for g in GS:
        d = res[g]; row = dict(design=design, T=T, k=k, K=K, r=r, delta=delta, g=g)
        for t, small in SMALL.items():
            q = np.quantile(null[t], 0.05 if small else 0.95)
            row[t + "_sa"] = round(float(np.mean(d[t] < q) if small else np.mean(d[t] > q)), 3)
        row["PO"] = round(float(np.mean(d.PO < ERS_CV5)), 3)
        row["MINP_K"] = round(float(np.mean(d.MINP_K < cv_minp)), 3)
        if "PO_BOOT_p" in d: row["PO_BOOT"] = round(float(np.mean(d.PO_BOOT_p <= 0.05)), 3)
        cvpr = C.PR_CV_ADF.get(m)
        row["PR_ADF"] = round(float(np.mean(d.PR_ADF < cvpr)), 3) if cvpr is not None else np.nan
        row["EG"] = round(float(np.mean(d.EG < C.mackinnoncrit(N=min(m + 1, 6), regression="c", nobs=T)[1])), 3)
        row["JOH"] = round(float(np.mean(d.JOH > d.JOH_cv)), 3)
        row["KPSS_HT"] = round(float(np.mean(d.KPSS_HT > 0.463)), 3)
        if design == "tri" and g > 0:
            col = f"NP_{g}"
            row["BOUND"] = round(float(np.mean(d[col] > np.quantile(null[col], 0.95))), 3)
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=199)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_final")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "*.csv")): os.remove(f)
    groups = []
    for T in [200, 400]:
        groups += [("tri", T, 1, 1, 1, [0.0, 0.6]), ("tri", T, 3, 3, 3, [0.0, 0.6]),
                   ("cvar", T, 3, 3, 3, [0.0, 0.6]), ("cvar", T, 6, 6, 5, [0.0, 0.6])]
    t0 = time.time(); bi = 0
    with Pool(a.workers) as pool:
        for gi, (design, T, k, K, r, deltas) in enumerate(groups):
            ref = None
            for di, delta in enumerate(deltas):
                res = {}
                for gj, g in enumerate(GS):
                    B = a.boot if g in BOOT_GS else 0
                    args = [(design, T, k, K, r, g, delta, 1_500_000_000 + 50_000_000 * gi + 10_000_000 * di + 100_000 * gj + i, B)
                            for i in range(a.reps)]
                    res[g] = pd.DataFrame(pool.map(one, args, chunksize=1 if B else max(1, a.reps // 64)))
                if delta == 0.0: ref = res[0.0].copy()
                pd.DataFrame(summarise(res, design, T, k, K, r, delta, ref)).to_csv(
                    os.path.join(a.out, f"block_{bi:02d}.csv"), index=False); bi += 1
                print(f"T={T} {design} k={k} K={K} delta={delta} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.out, "block_*.csv")))], ignore_index=True)
    df.to_csv(os.path.join(a.out, "results.csv"), index=False)
    keys = ["design", "T", "k", "K", "delta"]
    size_cols = ["PO", "PO_BOOT", "MINP_K", "PR_ADF", "EG", "JOH", "KPSS_HT"]
    df[df.g == 0][keys + [c for c in size_cols if c in df]].to_csv(os.path.join(a.out, "size_table.csv"), index=False)
    pw = [c for c in ["BOUND", "PO_sa", "PO_BOOT", "MINP_K_sa", "PR_ADF_sa", "EG_sa", "JOH_sa", "ECR_sa", "WALD_sa"] if c in df]
    df[df.g.isin([5.0, 10.0, 20.0])][keys + ["g"] + pw].to_csv(os.path.join(a.out, "power_table.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    styles = dict(BOUND=("k-", "Bound (known vector)"), PO_sa=("b-o", "CVAR test"), PO_BOOT=("b:", "CVAR test, bootstrap (actual)"),
                  MINP_K_sa=("r--", "Robust (MINP)"), PR_ADF_sa=("g:", "PR ADF-GLS"), EG_sa=("r:", "Engle-Granger"),
                  JOH_sa=("m--", "Johansen"), ECR_sa=("c--", "ECR"), WALD_sa=("y-", "Boswijk Wald"))
    for T in [200, 400]:
        d0 = df[df["T"] == T]; ks = d0[["design", "k", "K", "delta"]].drop_duplicates().values.tolist()
        n = len(ks); cols = 4; rws = int(np.ceil(n / cols))
        fig, ax = plt.subplots(rws, cols, figsize=(4.3 * cols, 3.4 * rws), sharey=True); ax = np.atleast_2d(ax)
        for i, key in enumerate(ks):
            a_ = ax.flat[i]
            x = d0[(d0.design == key[0]) & (d0.k == key[1]) & (d0.K == key[2]) & (d0.delta == key[3])]
            for c, (st, lab) in styles.items():
                if c in x and x[c].notna().any():
                    xx = x[x[c].notna()]; a_.plot(xx.g, xx[c], st, ms=3, lw=1.2, label=lab)
            a_.axhline(0.05, color="grey", lw=0.5)
            name = ("Triangular, k=%d" % key[1]) if key[0] == "tri" else ("CVAR design, K=%d" % key[2])
            a_.set_title(f"{name}, delta={key[3]}", fontsize=9)
        for j in range(n, rws * cols): ax.flat[j].axis("off")
        ax.flat[0].legend(fontsize=6); fig.supxlabel("g = T(1 - rho)"); fig.supylabel("Rejection rate")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"power_curves_T{T}.png"), dpi=150)
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
