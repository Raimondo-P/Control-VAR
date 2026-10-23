#!/usr/bin/env python3
"""
Reproduce Tables 11 and 14 of the paper with the sieve-bootstrap lag rule of Section 5,
    p_T = ceil(16 (T/200)^(1/3))   (16 lags at T = 200, 20 at T = 400).
Needs, in the same folder: mc_final_test.py, mc_feedback.py and the files they import
(mc_cpok.py, mc_cpo2.py, mc_adaptive.py, mc_competitors.py, mc_bootstrap_cvar.py,
po_critical_values.py, robust_po.py, mc_power.py, mc_envelope.py).

Table 11 (tab:mc-test-size): rejection rates under the null (g = 0) at 5% of the CVAR test with the ERS
    critical value, its sieve-bootstrap version, the robust adaptive test, Perron-Rodriguez ADF-GLS,
    Engle-Granger and Johansen, in the 16 designs of mc_final_test.py (triangular k = 1, 3 and CVAR
    K = 3, 6; delta in {0, 0.6}; T in {200, 400}).
Table 14 (tab:mc-test-feedback): size at 5% (g = 0) of the CVAR test, its bootstrap version and the
    augmented statistic with 4 and 12 lags, and size-adjusted power at g = 10 of the CVAR test and the
    augmented statistic, in the five feedback designs of mc_feedback.py at T in {200, 400}.

Outputs in --out (default tables_11_14):
    table11.csv, table14.csv        the numbers
    table11.tex, table14.tex        complete LaTeX tables, same layout and labels as the paper
Usage: py make_tables_11_14.py              (1000 replications, 199 bootstrap draws)
"""
import argparse, os, time, math
import numpy as np, pandas as pd
from multiprocessing import Pool
import warnings; warnings.filterwarnings("ignore")
import mc_final_test as F
import mc_feedback as FB
import mc_adaptive as A
import robust_po as RP

def lag_rule(T):
    return int(math.ceil(16 * (T / 200) ** (1 / 3)))

# ------------------------------------------------------------------ Table 11
def one11(args):
    design, T, k, K, r, delta, seed, B = args
    F.P_SIEVE = lag_rule(T)                       # used by F.boot_p inside F.one
    return F.one((design, T, k, K, r, 0.0, delta, seed, B))

def table11(pool, reps, B):
    rows = []
    designs = []
    for T in [200, 400]:
        designs += [("tri", T, 1, 1, 1), ("tri", T, 3, 3, 3), ("cvar", T, 3, 3, 3), ("cvar", T, 6, 6, 5)]
    for di, (design, T, k, K, r) in enumerate(designs):
        res = {}
        for dj, delta in enumerate([0.0, 0.6]):
            args = [(design, T, k, K, r, delta, 4_000_000_000 + 10_000_000 * di + 1_000_000 * dj + i, B) for i in range(reps)]
            res[delta] = pd.DataFrame(pool.map(one11, args, chunksize=1))
        ref = res[0.0]
        nPO, nK = np.sort(ref.PO.values), np.sort(ref.CPO_K.values)
        minp = lambda d: np.minimum(A.pval(d.PO.values, nPO), A.pval(d.CPO_K.values, nK))
        cv_minp = np.quantile(minp(ref), 0.05)
        m = k if design == "tri" else K
        for delta in [0.0, 0.6]:
            d = res[delta]
            cvpr = F.C.PR_CV_ADF.get(m)
            rows.append(dict(T=T, design=design, k=k, K=K, delta=delta,
                             PO=np.mean(d.PO < F.ERS_CV5), PO_BOOT=np.mean(d.PO_BOOT_p <= 0.05),
                             MINP_K=np.mean(minp(d) < cv_minp),
                             PR_ADF=np.mean(d.PR_ADF < cvpr) if cvpr is not None else np.nan,
                             EG=np.mean(d.EG < F.C.mackinnoncrit(N=min(m + 1, 6), regression="c", nobs=T)[1]),
                             JOH=np.mean(d.JOH > d.JOH_cv), p_sieve=lag_rule(T)))
        print(f"Table 11: T={T} {design} k={k} K={K} done", flush=True)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ Table 14
def one14(args):
    T, g, fb, phi_v, lev, seed, B = args
    rng = np.random.default_rng(seed)
    y, X = FB.simulate(T, g, fb, phi_v, lev, rng)
    out = dict(PO=RP.po_maic(y, X, FB.CBAR), APO_4=FB.apo(y, X, 4), APO_12=FB.apo(y, X, 12))
    if B > 0: out["PO_BOOT_p"] = FB.boot_p(y, X, B, rng, psieve=lag_rule(T))
    return out

def table14(pool, reps, B):
    rows = []
    for T in [200, 400]:
        for di, (fb, phi_v, lev) in enumerate(FB.DESIGNS):
            res = {}
            for gj, g in enumerate([0.0, 10.0]):
                args = [(T, g, fb, phi_v, lev, 4_500_000_000 + 100_000_000 * di + 10_000_000 * gj + 1_000_000 * (T == 400) + i,
                         B if g == 0.0 else 0) for i in range(reps)]
                res[g] = pd.DataFrame(pool.map(one14, args, chunksize=1 if g == 0.0 else max(1, reps // 64)))
            n0, a10 = res[0.0], res[10.0]
            row = dict(T=T, fb=fb, phi_v=phi_v, lev=lev, p_sieve=lag_rule(T),
                       PO=np.mean(n0.PO < FB.ERS_CV5), PO_BOOT=np.mean(n0.PO_BOOT_p <= 0.05),
                       APO_4=np.mean(n0.APO_4 < FB.ERS_CV5), APO_12=np.mean(n0.APO_12 < FB.ERS_CV5))
            for t in ["PO", "APO_4", "APO_12"]:
                row[t + "_sa"] = np.mean(a10[t] < np.quantile(n0[t], 0.05))
            rows.append(row)
            print(f"Table 14: T={T} fb={fb} phi_v={phi_v} lev={lev} done", flush=True)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ LaTeX
def f(v): return "" if pd.isna(v) else f"{v:.3f}"

def tex11(df):
    rows = []
    for _, r in df.sort_values(["T", "design", "k", "K", "delta"], ascending=[True, False, True, True, True]).iterrows():
        lab = f"Triangular, $k={int(r.k)}$" if r.design == "tri" else f"CVAR, $K={int(r.K)}$"
        rows.append(" & ".join([str(int(r["T"])), lab, f"{r.delta:.1f}", f(r.PO), f(r.PO_BOOT), f(r.MINP_K), f(r.PR_ADF), f(r.EG), f(r.JOH)]) + r"\\")
    return (r"""\begin{table}[H]
\centering\small
\caption{Test of the design: rejection rates under the null ($g=0$) at the 5 percent level. CVAR test: statistic \eqref{eq:PT} with $\bar c=-7$ and the critical value of \citet{ERS1996}; bootstrap: sieve bootstrap p-value with an autoregression of order $p_T=\lceil16(T/200)^{1/3}\rceil$; robust: adaptive test of Appendix \ref{sec:supp-test}, with the critical value from the null with uncorrelated innovations of the same design, so that its size is nominal by construction when $\delta=0$; PR: ADF-GLS of \citet{PerronRodriguez2016} with their critical values (not tabulated for six regressors); EG: Engle-Granger with the critical values of \citet{MacKinnon2010}; Johansen: maximum eigenvalue test. $\delta$: correlation between the innovations of the cycle and those of the controls. 1000 replications, 199 bootstrap draws.\label{tab:mc-test-size}}
\begin{tabular}{rlc cccccc}
\toprule
$T$ & Design & $\delta$ & CVAR test & Bootstrap & Robust & PR & EG & Johansen\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

def tex14(df):
    names = {(0.0, 0.0, 0.0): "No feedback", (0.0, 0.5, 0.0): "Serially correlated cycle", (0.5, 0.0, 0.0): "Feedback from own change",
             (0.5, 0.5, 0.0): "Feedback and serial correlation", (0.0, 0.0, 0.3): "Feedback from own level"}
    rows = []
    for _, r in df.iterrows():
        nm = names[(r.fb, r.phi_v, r.lev)]
        rows.append(" & ".join([str(int(r["T"])), nm, f(r.PO), f(r.PO_BOOT), f(r.APO_4), f(r.APO_12), f(r.PO_sa), f(r.APO_4_sa), f(r.APO_12_sa)]) + r"\\")
    return (r"""\begin{table}[H]
\centering\small
\caption{Policies that react to the treated unit. Size at 5 percent ($g=0$) and size-adjusted power at $\rho=1-10/T$ ($g=10$) of the CVAR test, of its bootstrap version (autoregression of order $p_T=\lceil16(T/200)^{1/3}\rceil$), and of the augmented statistic with 4 and 12 lags of $\Delta Y^1_t$. Feedback from own change: the policy reacts to the twelve-month change of the treated unit; feedback from own level: to the level of its specific component. 1000 replications, 199 bootstrap draws.\label{tab:mc-test-feedback}}
\resizebox{\textwidth}{!}{\setlength{\tabcolsep}{4pt}
\begin{tabular}{rl cccc ccc}
\toprule
 & & \multicolumn{4}{c}{Size} & \multicolumn{3}{c}{Size-adjusted power}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-9}
$T$ & Design & CVAR test & Bootstrap & Augm. 4 & Augm. 12 & CVAR test & Augm. 4 & Augm. 12\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=199)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="tables_11_14")
    ap.add_argument("--only", choices=["11", "14", "both"], default="both")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    with Pool(a.workers) as pool:
        if a.only in ("11", "both"):
            t11 = table11(pool, a.reps, a.boot)
            t11.round(3).to_csv(os.path.join(a.out, "table11.csv"), index=False)
            open(os.path.join(a.out, "table11.tex"), "w").write(tex11(t11))
            print(f"Table 11 written ({time.time()-t0:.0f}s)", flush=True)
        if a.only in ("14", "both"):
            t14 = table14(pool, a.reps, a.boot)
            t14.round(3).to_csv(os.path.join(a.out, "table14.csv"), index=False)
            open(os.path.join(a.out, "table14.tex"), "w").write(tex14(t14))
            print(f"Table 14 written ({time.time()-t0:.0f}s)", flush=True)
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
