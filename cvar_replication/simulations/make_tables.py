r"""make_tables.py: LaTeX tables of the supplement from the outputs of run_all.py.

  tab_mc_var.tex     VAR without controls and CVAR (results/mc_sim1)
  tab_mc_lags.tex    lag length (results/mc_lag_aic, mc_lag_8, mc_lag_12)
  tab_mc_boot.tex    average coverage across the grid (results/mc_grid_p12)
  tab_mc_grid.tex    coverage in every design of the grid (results/mc_grid_p12)
  tab_mc_spill.tex   spillovers (results/mc_ext/spill.csv)
  tab_mc_nonlin.tex  non-linear effects (results/mc_ext/nonlin.csv)
  tab_mc_stat.tex    stationarity diagnostics (results/mc_ext/stat.csv)
Written to results/tex, which the supplement reads. A table whose inputs are missing is skipped.
Usage: python make_tables.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "tex")
HS = [0, 1, 3, 6, 12, 24]


def f3(x):
    return "" if pd.isna(x) else (f"{x:.3f}" if x >= 0 else f"$-${-x:.3f}")


def f2(x):
    return "" if pd.isna(x) else (f"{x:.2f}" if x >= 0 else f"$-${-x:.2f}")


def summaries(run):
    f = os.path.join(RES, run, "all_summaries.csv")
    if not os.path.exists(f):
        print(f"  missing {f}")
        return None
    return pd.read_csv(f)


def meta(run):
    ms = sorted(glob.glob(os.path.join(RES, run, "meta_*.json")))
    return json.load(open(ms[0])) if ms else {}


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")
    print(f"  {name} written")


def pick(d, scheme, target, h, col):
    z = d[(d.scheme == scheme) & (d.target == target) & (d.h == h)]
    return z[col].mean() if len(z) else np.nan


def tab_var():
    d = summaries("mc_sim1")
    if d is None:
        return
    m = meta("mc_sim1")
    rows = []
    for h in HS:
        rows.append(f"{h} & {pick(d, 'plainvar', 'theta', h, 'truth'):.3f} & {f3(pick(d, 'plainvar', 'theta', h, 'bias'))} & "
                    f"{f3(pick(d, 'plainvar', 'theta', h, 'rmse'))} & {f2(pick(d, 'plainvar', 'theta', h, 'coverage'))} & & "
                    f"{f3(pick(d, 'refit', 'theta', h, 'bias'))} & {f3(pick(d, 'refit', 'theta', h, 'rmse'))} & "
                    f"{f2(pick(d, 'refit', 'theta', h, 'coverage'))}\\\\")
    write("tab_mc_var.tex", [r"\begin{table}[H]", r"\centering",
          r"\caption{VAR without controls and CVAR, both with 12 lags. Bias, root mean squared error and coverage of the 90 percent bands "
          rf"for the effect of a policy surprise $\theta_h$. $T=200$, $K=6$, $r=5$, $\phi=0.3$, {m.get('reps', 500)} replications, "
          rf"{m.get('B', 499)} bootstrap draws.\label{{tab:mc-var}}}}",
          r"\begin{tabular}{rr ccc c ccc}", r"\toprule",
          r" & & \multicolumn{3}{c}{VAR without controls} & & \multicolumn{3}{c}{CVAR}\\", r"\cmidrule{3-5}\cmidrule{7-9}",
          r"$h$ & $\theta_h$ & Bias & RMSE & Coverage & & Bias & RMSE & Coverage\\", r"\midrule"] + rows +
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def tab_lags():
    ds = {lr: summaries(f"mc_lag_{lr}") for lr in ["aic", "8", "12"]}
    if any(v is None for v in ds.values()):
        return
    m = meta("mc_lag_aic")
    mean_p = m.get("mean_p", np.nan)
    rows = []
    for h in HS:
        c = [f"{h}", f"{pick(ds['12'], 'refit', 'gamma', h, 'truth'):.3f}"]
        for lr in ["aic", "8", "12"]:
            c += [f3(pick(ds[lr], 'refit', 'gamma', h, 'bias')), f2(pick(ds[lr], 'refit', 'gamma', h, 'coverage'))]
        rows.append(" & ".join(c) + r"\\")
    write("tab_mc_lags.tex", [r"\begin{table}[H]", r"\centering",
          r"\caption{CVAR with different lag lengths. Bias of the response and coverage of the 90 percent bands for $\gamma_h$, bootstrap "
          rf"with re-estimated weights. $T=200$, $K=6$, $r=5$, $\phi=0.3$, {m.get('reps', 500)} replications.\label{{tab:mc-lags}}}}",
          r"\begin{tabular}{rr cc cc cc}", r"\toprule",
          rf" & & \multicolumn{{2}}{{c}}{{Akaike (mean {mean_p:.1f})}} & \multicolumn{{2}}{{c}}{{8 lags}} & \multicolumn{{2}}{{c}}{{12 lags}}\\",
          r"$h$ & $\gamma_h$ & Bias & Coverage & Bias & Coverage & Bias & Coverage\\", r"\midrule"] + rows +
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def tab_boot_grid():
    d = summaries("mc_grid_p12")
    if d is None:
        return
    m = meta("mc_grid_p12")
    g = d[d.target == "gamma"]
    rows = []
    for h in HS:
        c = [f"{h}"]
        for i, fb in enumerate([0.0, 0.3]):
            z = g[(np.isclose(g.feedback, fb)) & (g.h == h)]
            c += [f2(z[z.scheme == s].coverage.mean()) for s in ["oracle", "fixed", "refit"]]
            if i == 0:
                c.append("")
        rows.append(" & ".join(c) + r"\\")
    n0 = g[np.isclose(g.feedback, 0.0)].design.nunique()
    n1 = g[np.isclose(g.feedback, 0.3)].design.nunique()
    write("tab_mc_boot.tex", [r"\begin{table}[H]", r"\centering",
          r"\caption{Average coverage of the 90 percent bands for $\gamma_h$ across the designs of the grid, with 12 lags, "
          rf"{m.get('reps', 500)} replications per design and {m.get('B', 499)} bootstrap draws. {n0} designs without feedback ($\phi=0$) "
          rf"and {n1} with feedback ($\phi=0.3$).\label{{tab:mc-boot}}}}",
          r"\begin{tabular}{r ccc c ccc}", r"\toprule",
          r" & \multicolumn{3}{c}{$\phi=0$} & & \multicolumn{3}{c}{$\phi=0.3$}\\", r"\cmidrule{2-4}\cmidrule{6-8}",
          r"$h$ & Known & Fixed & Re-estimated & & Known & Fixed & Re-estimated\\", r"\midrule"] + rows +
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    keys = g[["design", "T", "K", "r", "feedback", "boundary"]].drop_duplicates()
    keys = keys.assign(bd=keys.boundary.astype(bool)).sort_values(["T", "K", "r", "feedback", "bd"], ascending=[True, True, True, True, False])
    rows = []
    for k in keys.itertuples():
        z = g[g.design == k.design]
        c = [f"{k.T}", f"{k.K}", f"{k.r}", f"{k.feedback:.1f}", "yes" if k.bd else "no"]
        for h in [0, 12]:
            c += [f2(z[(z.scheme == s) & (z.h == h)].coverage.mean()) for s in ["oracle", "fixed", "refit"]]
        rows.append(" & ".join(c) + r"\\")
    write("tab_mc_grid.tex", [r"\begin{table}[H]", r"\centering\small",
          r"\caption{Coverage of the 90 percent bands for $\gamma_h$ in every design of the grid of Appendix \ref{sec:montecarlo}, on impact "
          r"and after twelve months. Known: weights known. Fixed: estimated weights held fixed. Re-est.: weights re-estimated in every draw. "
          r"Boundary: one true weight equal to zero.\label{tab:mc-grid}}",
          r"\begin{tabular}{ccccc ccc ccc}", r"\toprule",
          r" & & & & & \multicolumn{3}{c}{$h=0$} & \multicolumn{3}{c}{$h=12$}\\", r"\cmidrule{6-8}\cmidrule{9-11}",
          r"$T$ & $K$ & $r$ & $\phi$ & Boundary & Known & Fixed & Re-est. & Known & Fixed & Re-est.\\", r"\midrule"] + rows +
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def ext(name):
    f = os.path.join(RES, "mc_ext", f"{name}.csv")
    if not os.path.exists(f):
        print(f"  missing {f}")
        return None
    return pd.read_csv(f)


def ext_reps():
    f = os.path.join(RES, "mc_ext", "meta.json")
    return json.load(open(f)).get("reps", 1000) if os.path.exists(f) else 1000


def tab_spill():
    d = ext("spill")
    if d is None:
        return
    d = d[(d["T"] == 200) & (d["K"] == 5)]
    lab = {"syn": "CVAR, all controls", "near": "CVAR, nearest control", "far": "CVAR, most distant control", "expo": "Exposure estimator"}
    rows = []
    for e, name in lab.items():
        c = [name]
        for pi in [0.0, 0.5]:
            z = d[(d.estimator == e) & np.isclose(d.pi, pi)]
            c += [f3(z[z.h == 0].bias.mean()), f3(z[z.h == 3].bias.mean()), f3(z.rmse.mean())]
        rows.append(" & ".join(c) + r"\\")
    write("tab_mc_spill.tex", [r"\begin{table}[H]", r"\centering\small",
          r"\caption{Spillovers. Bias at impact ($h=0$) and after three months ($h=3$), and root mean squared error averaged over horizons "
          rf"0 to 12, for the effect of a policy surprise. Five controls at increasing distance, exposure $x_k=(d_1/d_k)^2$, $T=200$, "
          rf"{ext_reps()} replications, 12 lags.\label{{tab:mc-spill}}}}",
          r"\begin{tabular}{l ccc ccc}", r"\toprule",
          r" & \multicolumn{3}{c}{No spillover, $\pi=0$} & \multicolumn{3}{c}{Spillover, $\pi=0.5$}\\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          r"Estimator & $h=0$ & $h=3$ & RMSE & $h=0$ & $h=3$ & RMSE\\", r"\midrule"] + rows +
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def tab_nonlin():
    d = ext("nonlin")
    if d is None:
        return
    d = d[d["T"] == 200]
    rows = []
    for k in sorted(d.kappa.unique()):
        for dl in sorted(d.delta.unique()):
            c = [f"{k:.1f}", f"{dl:.2f}"]
            for e in ["linear", "dictionary"]:
                z = d[np.isclose(d.kappa, k) & np.isclose(d.delta, dl) & (d.estimator == e)]
                c += [f3(z[z.h == 0].bias.mean()), f3(z[z.h == 3].bias.mean()), f3(z.rmse.mean())]
            rows.append(" & ".join(c) + r"\\")
    write("tab_mc_nonlin.tex", [r"\begin{table}[H]", r"\centering\small",
          r"\caption{Non-linear effects. Bias at impact and after three months, and root mean squared error averaged over horizons 0 to 12, "
          r"for the response to a policy innovation of size $\delta$, with the effect $u+\kappa u^2$. $\delta$ equal to minus one, one and two "
          rf"standard deviations of the innovation, $T=200$, {ext_reps()} replications, 12 lags.\label{{tab:mc-nonlin}}}}",
          r"\begin{tabular}{cc ccc ccc}", r"\toprule",
          r" & & \multicolumn{3}{c}{Linear CVAR} & \multicolumn{3}{c}{Dictionary, Theorem \ref{thm:nonlin}}\\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
          r"$\kappa$ & $\delta$ & Bias $h=0$ & Bias $h=3$ & RMSE & Bias $h=0$ & Bias $h=3$ & RMSE\\", r"\midrule"] + rows +
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def tab_stat():
    d = ext("stat")
    if d is None:
        return
    d = d[d["K"] == 6]
    lab = {"adf_cycle": "ADF, cycle", "adf_net": "ADF, cycle net of policy", "kpss_cycle": "KPSS, cycle", "kpss_net": "KPSS, cycle net of policy"}
    rows = []
    for T in sorted(d["T"].unique()):
        for s, name in lab.items():
            z = d[(d["T"] == T) & (d.stat == s)]
            g = lambda col, rho: f2(z[np.isclose(z.rho, rho)][col].mean())
            rows.append(f"{T} & {name} & {g('reject_std_cv', 1.0)} & {g('reject_std_cv', 0.9)} & {g('reject_sim_cv', 1.0)} & {g('reject_sim_cv', 0.9)}\\\\")
    write("tab_mc_stat.tex", [r"\begin{table}[H]", r"\centering\small",
          r"\caption{Stationarity diagnostics for the estimated cycle. Rejection frequencies at 5 percent with the standard critical values "
          r"(Engle-Granger for ADF, 0.463 for KPSS) and with simulated critical values (from the unit-root design for ADF and from the design "
          r"with $\rho=0.9$ for KPSS). $\rho=1$: the counterfactual cycle has a unit root. $\rho=0.9$: it is a stationary AR(1). The ADF test "
          r"takes a unit root as the null hypothesis, the KPSS test stationarity. Six controls, weights estimated by constrained dynamic least "
          rf"squares with policy terms, {ext_reps()} replications.\label{{tab:mc-stat}}}}",
          r"\begin{tabular}{ll cc cc}", r"\toprule",
          r" & & \multicolumn{2}{c}{Standard critical values} & \multicolumn{2}{c}{Simulated critical values}\\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
          r"$T$ & Statistic & $\rho=1$ & $\rho=0.9$ & $\rho=1$ & $\rho=0.9$\\", r"\midrule"] + rows +
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


if __name__ == "__main__":
    for f in [tab_var, tab_lags, tab_boot_grid, tab_spill, tab_nonlin, tab_stat]:
        f()
