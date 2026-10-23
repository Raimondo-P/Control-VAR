"""eb_6_exhibits.py: figures and LaTeX tables for the extended benefits application.
Requires eb_1 to eb_4 (results/eb/1_unit_roots.csv, 2a, 2b, 3_lags.csv, 4_states.csv).
Re-estimates the average across states with the joint bootstrap (one common lag order, the larger
of the median rule orders of the CVAR and of the VAR) to obtain full paths.
Outputs in results/paper/:
  fig_eb_average.pdf/.png  (a) cumulated response of the unemployment rate, CVAR with 68 and 90
                           percent bands and VAR without controls with its 90 percent band;
                           (b) Proposition 1: response of the treated state and of the weighted
                           controls in the VAR with both, with 90 percent bands; (c) response of the
                           weeks of EB to their own innovation and impulse causal effect.
  fig_eb_states.pdf/.png   response at 12 months by state, CVAR and VAR with 90 percent bands, and
                           the CVAR / VAR band-width ratio at 6, 12 and 24 months.
  tab_eb_tests.tex         unit-root and design tests.
  tab_eb_average.tex       average responses at 0, 6, 12, 24 and 36 months.
  tab_eb_bounds.tex        bounds of Proposition 2 for the average, at 6 and 12 months.
  tab_eb_states.tex        every state at 12 months (appendix).
Usage: py eb_6_exhibits.py [--B 999]
"""
from common import *

a = args()
H = 36
SRC = outdir("eb")
OUT = outdir("paper")
for f in ["1_unit_roots.csv", "2a_estimated_weights.csv", "2b_imposed_weights.csv", "3_lags.csv", "4_states.csv"]:
    if not os.path.exists(os.path.join(SRC, f)):
        sys.exit(f"{f} not found in {SRC}: run eb_1 to eb_4 first")
UR1 = pd.read_csv(os.path.join(SRC, "1_unit_roots.csv"))
T2a = pd.read_csv(os.path.join(SRC, "2a_estimated_weights.csv"))
T2b = pd.read_csv(os.path.join(SRC, "2b_imposed_weights.csv"))
L = pd.read_csv(os.path.join(SRC, "3_lags.csv")).set_index("st")
S4 = pd.read_csv(os.path.join(SRC, "4_states.csv"))
UR, EB, EUC, elig = load_eb()
sts = [s for s in elig if s in L.index]
N = len(sts)


def f2(x):
    return f"{x:.2f}".replace("-", "$-$")


def band(lo, hi):
    return f"[{f2(lo)}, {f2(hi)}]"


# ------------------------------------------------------------------ joint estimation of the average
p = int(max(L.loc[sts].p_cvar.median(), L.loc[sts].p_var.median()))
print(f"{N} states, common lag order {p}", flush=True)
rs = [fit_eb_diff(UR, EB, EUC, s, p) for s in sts]
per, avg = cvar.joint_bootstrap(rs, B=a.B, H=H, seed=21)
perc, avgc = cvar.joint_controls_response(rs, B=a.B, H=H, seed=22)
pt = {k: avg.point[k][:, 0] for k in avg.point}
b90 = {k: avg.bands(k, 0.10) for k in avg.point}
b68 = {k: avg.bands(k, 0.32) for k in avg.point}
c90 = {k: avgc.bands(k, 0.10) for k in avgc.point}
pd.DataFrame({"h": np.arange(H + 1), **{f"{k}": pt[k] for k in pt},
              **{f"{k}_lo90": b90[k][0][:, 0] for k in pt}, **{f"{k}_hi90": b90[k][1][:, 0] for k in pt},
              **{f"{k}_lo68": b68[k][0][:, 0] for k in pt}, **{f"{k}_hi68": b68[k][1][:, 0] for k in pt},
              **{f"ctrl_{k}": avgc.point[k][:, 0] for k in avgc.point},
              **{f"ctrl_{k}_lo90": c90[k][0][:, 0] for k in avgc.point},
              **{f"ctrl_{k}_hi90": c90[k][1][:, 0] for k in avgc.point}}).to_csv(
    os.path.join(OUT, "fig_eb_average.csv"), index=False)

# ------------------------------------------------------------------ figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hh = np.arange(H + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    k = "cycle"
    ax[0].fill_between(hh, b90[k][0][:, 0], b90[k][1][:, 0], color="C0", alpha=0.18, lw=0, label="CVAR, 90%")
    ax[0].fill_between(hh, b68[k][0][:, 0], b68[k][1][:, 0], color="C0", alpha=0.35, lw=0, label="CVAR, 68%")
    ax[0].plot(hh, pt[k], color="C0", lw=1.8, label="CVAR")
    ax[0].plot(hh, pt["no_controls"], color="C3", lw=1.2, ls="--", label="VAR without controls")
    ax[0].plot(hh, b90["no_controls"][0][:, 0], color="C3", lw=0.7, ls=":")
    ax[0].plot(hh, b90["no_controls"][1][:, 0], color="C3", lw=0.7, ls=":", label="VAR, 90%")
    ax[0].set_title("(a) Unemployment rate", fontsize=10)
    ax[0].set_ylabel("percentage points")
    for kk, col, lab in [("treated", "C3", "treated state"), ("controls", "C2", "weighted controls"),
                         ("difference", "C0", "difference")]:
        ax[1].plot(hh, avgc.point[kk][:, 0], color=col, lw=1.5, label=lab)
        ax[1].fill_between(hh, c90[kk][0][:, 0], c90[kk][1][:, 0], color=col, alpha=0.12, lw=0)
    ax[1].set_title("(b) Treated state and weighted controls", fontsize=10)
    ax[2].plot(hh, pt["policy"], color="C1", lw=1.5, label="weeks of EB (tens)")
    ax[2].fill_between(hh, b90["policy"][0][:, 0], b90["policy"][1][:, 0], color="C1", alpha=0.15, lw=0)
    ax[2].plot(hh, pt["ice"], color="C0", lw=1.5, label="impulse causal effect")
    ax[2].fill_between(hh, b90["ice"][0][:, 0], b90["ice"][1][:, 0], color="C0", alpha=0.15, lw=0)
    ax[2].set_title("(c) Policy path and impulse causal effect", fontsize=10)
    for x in ax:
        x.axhline(0, color="k", lw=0.7)
        x.set_xlim(0, H)
        x.set_xlabel("months")
        x.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_eb_average.{ext}"), dpi=200)
    plt.close(fig)

    r12 = S4[(S4.spec == "rule") & (S4.h == 12)].set_index("st").loc[sts].sort_values("cvar")
    y = np.arange(len(r12))
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.22 * len(r12) + 1.5), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].errorbar(r12.cvar, y + 0.15, xerr=[r12.cvar - r12.cvar_lo, r12.cvar_hi - r12.cvar], fmt="o", ms=3,
                   color="C0", capsize=0, lw=1, label="CVAR")
    ax[0].errorbar(r12["var"], y - 0.15, xerr=[r12["var"] - r12.var_lo, r12.var_hi - r12["var"]], fmt="s", ms=3,
                   color="C3", capsize=0, lw=1, label="VAR without controls")
    ax[0].axvline(0, color="k", lw=0.7)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(r12.index, fontsize=7)
    ax[0].set_xlabel("percentage points, 12 months")
    ax[0].legend(fontsize=7, frameon=False, loc="lower right")
    for h, mk, col in [(6, "o", "C1"), (12, "s", "C0"), (24, "^", "C2")]:
        z = S4[(S4.spec == "rule") & (S4.h == h)].set_index("st").loc[r12.index]
        ax[1].plot(z.width_ratio, y, mk, ms=3, color=col, label=f"{h} months")
    ax[1].axvline(1, color="k", lw=0.7)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].set_xlabel("band width, CVAR / VAR")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_eb_states.{ext}"), dpi=200)
    plt.close(fig)
    print("figures written")
except ImportError:
    print("matplotlib not installed: figures skipped")

# ------------------------------------------------------------------ tables
def wtab(name, caption, label, colspec, header, rows, note):
    t = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + r"\label{" + label + "}}", r"\small",
         r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\", r"\midrule"] + rows + [
         r"\bottomrule", r"\end{tabular}", r"\par\smallskip\parbox{0.95\textwidth}{\footnotesize " + note + "}",
         r"\end{table}"]
    open(os.path.join(OUT, name), "w").write("\n".join(t) + "\n")
    print(name, "written")


# tests
rows = []
ers_t = UR1[UR1.state.isin(sts)]
ers_c = UR1[UR1.state.isin(CTRL)]
rows.append(rf"Unemployment rate, treated states & ERS & {int(ers_t.reject5.sum())} of {len(ers_t)} & {f2(ers_t.p_boot.median())} \\")
rows.append(rf"Unemployment rate, controls & ERS & {int(ers_c.reject5.sum())} of {len(ers_c)} & {f2(ers_c.p_boot.median())} \\")
rows.append(r"\midrule")
for sh, lab in [("none", "no level shifts"), ("auto", "level shifts")]:
    for q in (12, 6, 3, 0):
        z = T2a[(T2a.shifts == sh) & (T2a.q == q)]
        rows.append(rf"Cycle, estimated weights, {lab} & $q={q}$ & {int((z.p_boot <= 0.05).sum())} of {len(z)} & {f2(z.p_boot.median())} \\")
rows.append(r"\midrule")
rows.append(rf"Cycle, equal weights & $q=12$ & {int((T2b.p_boot <= 0.05).sum())} of {len(T2b)} & {f2(T2b.p_boot.median())} \\")
rows.append(rf"Cycle, equal weights, no policy terms & & {int((T2b.p_boot_nopolicy <= 0.05).sum())} of {len(T2b)} & {f2(T2b.p_boot_nopolicy.median())} \\")
wtab("tab_eb_tests.tex", "Unit roots and the test of the design", "tab:eb-tests", "llcc",
     r"Series & Specification & Rejections at 5 percent & Median p-value", rows,
     r"ERS: point-optimal test of \citet{ERS1996} with a constant, on the unemployment rate. Cycle: test of "
     r"Section \ref{sec:test} of the null that the treated state has a stochastic trend of its own, with the "
     r"current value and $q$ lags of the weeks of EUC and EB, controls the never-triggered states that do not "
     r"border the treated state, weights estimated by \eqref{eq:dols} or set equal. Level shifts: sequential "
     r"sup-Wald tests on the cycle net of the policy terms. Sieve-bootstrap p-values.")

# average
rows = []
for h in (0, 6, 12, 24, 36):
    w = (b90["cycle"][1][h, 0] - b90["cycle"][0][h, 0]) / (b90["no_controls"][1][h, 0] - b90["no_controls"][0][h, 0])
    rows.append(rf"{h} & {f2(pt['cycle'][h])} & {f2(pt['no_controls'][h])} & {f2(w)} & {f2(avgc.point['controls'][h, 0])} & {f2(pt['ice'][h])} \\")
    rows.append(r"\footnotesize & " + band(b90["cycle"][0][h, 0], b90["cycle"][1][h, 0]) + " & "
                + band(b90["no_controls"][0][h, 0], b90["no_controls"][1][h, 0]) + " & & "
                + band(c90["controls"][0][h, 0], c90["controls"][1][h, 0]) + " & "
                + band(b90["ice"][0][h, 0], b90["ice"][1][h, 0]) + r" \\[2pt]")
wtab("tab_eb_average.tex", "Response of the unemployment rate to ten additional weeks of EB, average across states",
     "tab:eb-avg", "cccccc",
     r"Months & CVAR & VAR without controls & Width ratio & Weighted controls & Impulse causal effect", rows,
     rf"Average across the {N} states with at least twelve months of EB, October 2002 to December 2019. Cumulated "
     rf"response in percentage points to an EB innovation of ten weeks, with EUC ordered first. CVAR: VAR in the "
     rf"weeks of EUC and EB and the change in the cycle, equal weights on the controls. VAR without controls: the "
     rf"same with the change in the unemployment rate of the treated state. Weighted controls: response in the VAR "
     rf"with both the treated state and the weighted controls (Proposition \ref{{prop:placebo}}). Impulse causal "
     rf"effect of one week of EB from Theorem \ref{{thm:direct}}. {p} lags in every state; 90 percent bands from "
     rf"{a.B} draws of a residual bootstrap that draws the same months for every state.")

# bounds
bd = cvar.bounds(avgc, lams=(0.0, 0.25, 0.5), horizons=(6, 12))
rows = []
for _, r in bd.iterrows():
    rows.append(rf"{int(r.h)} & {f2(r.treated_response)} & " + " & ".join(
        band(r[f"lam{l:g}_lower"], r[f"lam{l:g}_upper"]) for l in (0.0, 0.25, 0.5))
        + rf" & {f2(r.breakdown_lambda) if np.isfinite(r.breakdown_lambda) else '$>3$'} \\")
wtab("tab_eb_bounds.tex", "Sensitivity to the exogeneity assumption, average across states", "tab:eb-bounds",
     "cccccc", r"Months & VAR with controls added & $\lambda=0$ & $\lambda=0.25$ & $\lambda=0.5$ & $\lambda$ reaching the VAR",
     rows,
     r"Response of the treated state in the VAR with the weighted controls added, and 90 percent bands for the "
     r"bounds of Proposition \ref{prop:bounds}, with the bias specific to the treated state at most $\lambda$ "
     r"times the bias from the common trends. Last column: smallest $\lambda$ at which the band contains the "
     r"response of the treated state.")

# states (appendix)
rows = []
for s in sorted(sts):
    z = S4[(S4.spec == "rule") & (S4.st == s) & (S4.h == 12)].iloc[0]
    rows.append(rf"{s} & {int(z.p_cvar)}/{int(z.p_var)} & {f2(z.cvar)} {band(z.cvar_lo, z.cvar_hi)} & "
                rf"{f2(z['var'])} {band(z.var_lo, z.var_hi)} & {f2(z.width_ratio)} \\")
wtab("tab_eb_states.tex", "Response of the unemployment rate to ten additional weeks of EB after twelve months, by state",
     "tab:eb-states", "lcccc", r"State & Lags & CVAR & VAR without controls & Width ratio", rows,
     r"Cumulated response in percentage points, with 90 percent bands from 999 residual bootstrap draws. Lags: "
     r"CVAR / VAR without controls, chosen by the rule of Section \ref{sec:empirical}. Width ratio: width of the "
     r"CVAR band over that of the VAR without controls.")
print("written to", OUT)
