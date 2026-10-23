#!/usr/bin/env python3
"""
Build the LaTeX tables of the test of the design that are not written directly by a simulation script:
  tab:mc-test-power  (Table 12)  from results/mc_final/results.csv
  tab:mc-test-lags   (Table 13)  from results/mc_bootfix/results.csv and results/mc_bootfix2/results.csv
  tab:supp-cpo       (Table 15)  from results/mc_cpok/block_*.csv
Tables 11 and 14 are written by make_tables_11_14.py.
Usage: py build_tex_tables.py [--results ../results] [--out ../results/tex]
"""
import argparse, os, glob
import pandas as pd

def f(v): return "" if pd.isna(v) else f"{v:.3f}"
def lab(r): return f"Triangular, $k={int(r.k)}$" if r.design == "tri" else f"CVAR, $K={int(r.K)}$"

def table12(res):
    d = pd.read_csv(os.path.join(res, "mc_final", "results.csv"))
    z = d[d.g == 10].sort_values(["T", "design", "k", "K", "delta"], ascending=[True, False, True, True, True])
    rows = [" & ".join([str(int(r["T"])), lab(r), f"{r.delta:.1f}", f(r.BOUND), f(r.PO_sa), f(r.MINP_K_sa), f(r.PR_ADF_sa),
                        f(r.EG_sa), f(r.JOH_sa), f(r.ECR_sa), f(r.WALD_sa)]) + r"\\" for _, r in z.iterrows()]
    return (r"""\begin{table}[H]
\centering\small
\caption{Test of the design: size-adjusted power at the local alternative $\rho=1-10/T$. Bound: power of the Neyman-Pearson test that knows the weights and the covariance \citep{ElliottJanssonPesavento2005}, computed in the triangular designs. ECR: error-correction $t$-test; Wald: \citet{Boswijk1994}. Other columns as in Table \ref{tab:mc-test-size}.\label{tab:mc-test-power}}
\resizebox{\textwidth}{!}{\setlength{\tabcolsep}{4pt}
\begin{tabular}{rlc cccccccc}
\toprule
$T$ & Design & $\delta$ & Bound & CVAR test & Robust & PR & EG & Johansen & ECR & Wald\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

def table13(res):
    b1 = pd.read_csv(os.path.join(res, "mc_bootfix", "results.csv"))
    b2 = pd.read_csv(os.path.join(res, "mc_bootfix2", "results.csv"))
    def gv(df, T, K, dl, col, g):
        x = df[(df["T"] == T) & (df.K == K) & (df.delta == dl) & (df.g == g)]
        return f(x[col].iloc[0]) if len(x) and col in x else "--"
    rows = []
    for T in [200, 400]:
        for K in [3, 6]:
            for dl in [0.0, 0.6]:
                r = [str(T), str(K), f"{dl:.1f}", gv(b1, T, K, dl, "ASY_c7", 0.0)]
                for g in [0.0, 10.0]:
                    for col, src in [("BOOT_c7_p12", b1), ("BOOT_c7_p16", b2), ("BOOT_c7_p20", b2), ("BOOT_c7_p24", b1)]:
                        r.append(gv(src, T, K, dl, col, g))
                rows.append(" & ".join(r) + r"\\")
    return (r"""\begin{table}[H]
\centering\small
\caption{Order of the autoregression in the sieve bootstrap, CVAR design. Rejection rates at 5 percent under the null ($g=0$) and at $\rho=1-10/T$ ($g=10$), for the asymptotic critical value and for the bootstrap with $p$ lags; $\bar c=-7$. The rule $p_T=\lceil16(T/200)^{1/3}\rceil$ gives 16 lags at $T=200$ and 21 at $T=400$. 1000 replications, 199 bootstrap draws; dashes mark orders not run.\label{tab:mc-test-lags}}
\resizebox{\textwidth}{!}{\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrc c cccc cccc}
\toprule
 & & & & \multicolumn{4}{c}{Size, bootstrap with $p$ lags} & \multicolumn{4}{c}{Power, bootstrap with $p$ lags}\\
\cmidrule(lr){5-8}\cmidrule(lr){9-12}
$T$ & $K$ & $\delta$ & Asymptotic & 12 & 16 & 20 & 24 & 12 & 16 & 20 & 24\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

def table15(res):
    k = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(res, "mc_cpok", "block_*.csv")))])
    rows = []
    for _, r in k[k.g == 10].iterrows():
        s0 = k[(k.design == r.design) & (k.k == r.k) & (k.K == r.K) & (k.delta == r.delta) & (k.g == 0)].iloc[0]
        rows.append(" & ".join([lab(r), f"{r.delta:.1f}", f(r.BOUND), f(r.PO_sa), f(r.CPO_C_sa), f(r.CPO_K_sa), f(r.MINP_K_sa),
                                f(r.ECR_sa), f(r.WALD_sa), f(s0.PO), f(s0.MINP_K)]) + r"\\")
    return (r"""\begin{table}[H]
\centering\small
\caption{Conditional and adaptive tests, $T=200$. Size-adjusted power at $\rho=1-10/T$ for the bound, the CVAR test, the conditional tests CPO$_C$ and CPO$_K$, the adaptive test, and the error-correction tests; size at 5 percent of the CVAR test and of the adaptive test with critical values from the null with uncorrelated innovations of the same design. $\bar c=-7$, as in Section \ref{sec:test}. 2000 replications.\label{tab:supp-cpo}}
\resizebox{\textwidth}{!}{\setlength{\tabcolsep}{4pt}
\begin{tabular}{lc ccccccc cc}
\toprule
 & & \multicolumn{7}{c}{Size-adjusted power} & \multicolumn{2}{c}{Size}\\
\cmidrule(lr){3-9}\cmidrule(lr){10-11}
Design & $\delta$ & Bound & CVAR test & CPO$_C$ & CPO$_K$ & Adaptive & ECR & Wald & CVAR test & Adaptive\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "results"))
    ap.add_argument("--out", default=os.path.join(here, "..", "results", "tex"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for name, fn in [("table12", table12), ("table13", table13), ("table15", table15)]:
        try:
            open(os.path.join(a.out, name + ".tex"), "w").write(fn(a.results)); print("wrote", name)
        except FileNotFoundError as e:
            print(f"skipped {name}: {e}")

if __name__ == "__main__":
    main()
