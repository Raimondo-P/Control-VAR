"""ca_4_exhibits.py: main figure and LaTeX tables for the California application.
Requires ca_1_cec.py and ca_2_robustness.py. Outputs in results/paper/:
  fig_california_main.pdf/.png (+ fig_california_main.tex with the figure environment and caption,
      and the plotted paths in fig_california_<controls>.csv): CVAR response with 68 and 90 percent
      bootstrap bands and the VAR without controls, three control sets; the main specification is
      re-estimated here for the full path with --B draws.
  tab_california_main.tex: main specification at impact and one month (paper).
  tab_california_robust.tex: main specification, samples, leaving one PADD out, at 0, 1 and 12 months (supplement).
  tab_california_appendix.tex: lag orders, policy measures, the 2014-2019 sample, local projections.
Marks: dagger, the 68 percent band excludes zero; *, the 90 percent band excludes zero.
Usage: py ca_4_exhibits.py [--B 999] [--H 24]
"""
from common import *

a = args(H=dict(type=int, default=24))
H = a.H
SRC = outdir("california")
OUT = outdir("paper")
for f in ["B_main.csv", "C_lags.csv", "D_samples.csv", "E_controls.csv", "F_lp.csv"]:
    if not os.path.exists(os.path.join(SRC, f)):
        sys.exit(f"{f} not found in {SRC}: run ca_2_robustness.py first")
Bm, Cl, Ds, Ec, Fl = [pd.read_csv(os.path.join(SRC, f)) for f in
                      ["B_main.csv", "C_lags.csv", "D_samples.csv", "E_controls.csv", "F_lp.csv"]]
D, FIRST_CEC = load_california()
y, _, dose, _, idx = ca_series(D, ["gas_gulf"], "spliced", FIRST_CEC)
D = D.loc[idx]
mon = D.index.month


def lags_of(spec):
    r = Bm[(Bm.spec == spec) & (Bm.h == 0)].iloc[0]
    return int(r.p_cvar), int(r.p_var)


# ------------------------------------------------------------------ figure
paths = {}
for cs, (cc, lab) in SETS.items():
    C = np.column_stack([sa_changes(100 * D[c].values, [mon]) for c in cc])
    w = np.ones(len(cc)) / len(cc)
    pc, pv = lags_of(f"{cs}_spliced")
    bc = cvar.CVAR(p=pc).fit(y, C, dose, weights=w, difference=True).bootstrap(a.B, H, seed=1, no_controls=False)
    bv = cvar.CVAR(p=pv).fit(y, C, dose, weights=w, difference=True).bootstrap(a.B, H, seed=2, no_controls=True)
    paths[cs] = dict(pt=bc.point["cycle"][:, 0], b90=bc.bands("cycle", 0.10), b68=bc.bands("cycle", 0.32),
                     v=bv.point["no_controls"][:, 0], v90=bv.bands("no_controls", 0.10), lab=lab, p=(pc, pv))
    print(f"{lab}: lags {pc}/{pv}, impact {paths[cs]['pt'][0]:.2f}, 12m {paths[cs]['pt'][min(12, H)]:.2f}", flush=True)
    pd.DataFrame({"h": np.arange(H + 1), "cvar": paths[cs]["pt"],
                  "cvar_lo90": paths[cs]["b90"][0][:, 0], "cvar_hi90": paths[cs]["b90"][1][:, 0],
                  "cvar_lo68": paths[cs]["b68"][0][:, 0], "cvar_hi68": paths[cs]["b68"][1][:, 0],
                  "var": paths[cs]["v"], "var_lo90": paths[cs]["v90"][0][:, 0],
                  "var_hi90": paths[cs]["v90"][1][:, 0]}).to_csv(os.path.join(OUT, f"fig_california_{cs}.csv"), index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hh = np.arange(H + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for k, cs in enumerate(SETS):
        P = paths[cs]
        ax[k].fill_between(hh, P["b90"][0][:, 0], P["b90"][1][:, 0], color="C0", alpha=0.18, lw=0, label="CVAR, 90%")
        ax[k].fill_between(hh, P["b68"][0][:, 0], P["b68"][1][:, 0], color="C0", alpha=0.35, lw=0, label="CVAR, 68%")
        ax[k].plot(hh, P["pt"], color="C0", lw=1.8, label="CVAR")
        ax[k].plot(hh, P["v"], color="C3", lw=1.2, ls="--", label="VAR without controls")
        ax[k].plot(hh, P["v90"][0][:, 0], color="C3", lw=0.7, ls=":")
        ax[k].plot(hh, P["v90"][1][:, 0], color="C3", lw=0.7, ls=":", label="VAR, 90%")
        ax[k].axhline(0, color="k", lw=0.7)
        ax[k].axhline(1, color="grey", lw=0.8, ls="-.", label="full pass-through")
        ax[k].set_title(P["lab"], fontsize=10)
        ax[k].set_xlabel("months")
        ax[k].set_xlim(0, H)
    ax[0].set_ylabel("cents per gallon")
    ax[0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_california_main.{ext}"), dpi=200)
    plt.close(fig)
    print("figure written")
except ImportError:
    print("matplotlib not installed: figure skipped (the paths are in fig_california_<controls>.csv)")



fig_tex = [r"\begin{figure}[htbp]", r"\centering",
           r"\includegraphics[width=\textwidth]{fig_california_main.pdf}",
           r"\caption{Pass-through of the LCFS cost to the California retail gasoline price}",
           r"\label{fig:california_main}",
           r"\begin{minipage}{0.95\textwidth}\footnotesize",
           r"Note: cumulated response of the California retail price, in cents per gallon, to an increase "
           r"of 1 cent per gallon in the LCFS cost. Solid line and shaded areas: CVAR in the differenced "
           r"branch with equal weights on the controls named in each panel, with 68 and 90 percent residual "
           r"bootstrap bands. Dashed and dotted lines: VAR without controls in the LCFS cost and the change "
           r"in the California price, with its 90 percent band. Dash-dotted line: full pass-through.",
           r"\end{minipage}", r"\end{figure}"]
open(os.path.join(OUT, "fig_california_main.tex"), "w").write("\n".join(fig_tex) + "\n")


def mark(r, nm="cvar"):
    if r[f"{nm}_lo"] > 0 or r[f"{nm}_hi"] < 0:
        return "$^{*}$"
    if r[f"{nm}_lo68"] > 0 or r[f"{nm}_hi68"] < 0:
        return r"$^{\dagger}$"
    return ""


def cells(df, spec, label, lags, note=""):
    z = df[df.spec == spec].set_index("h")
    if z.empty:
        return []
    top, bot = [label + note, lags], ["", ""]
    for h in (0, 1, 12):
        r = z.loc[h]
        top.append(f"{r.cvar:.2f}{mark(r)}")
        bot.append(f"[{r.cvar_lo:.2f}, {r.cvar_hi:.2f}]")
    top.append(f"{z.loc[12].width_ratio_boot:.2f}")
    bot.append("")
    return [" & ".join(top) + r" \\", r"\footnotesize " + " & ".join(bot) + r" \\[2pt]"]


def lagtxt(df, spec):
    z = df[df.spec == spec]
    return "{}/{}".format(int(z.p_cvar.iloc[0]), int(z.p_var.iloc[0])) if not z.empty else ""


def block(title, lines):
    return [r"\multicolumn{6}{l}{" + title + r"} \\", r"\midrule"] + lines + [r"\midrule"] if lines else []


NOTE = (r"Note: cumulated response of the California retail price, in cents per gallon, to an increase "
        r"of 1 cent per gallon in the LCFS cost, from the CVAR in the differenced branch with equal weights "
        r"on the controls. Policy: the cost per gallon implied by the CARB formula before May 2024 and the "
        r"cost reported by refiners to the California Energy Commission from then on, in monthly changes, "
        r"unless stated otherwise. Lags: CVAR / VAR without controls. 90 percent residual bootstrap bands in "
        r"brackets; $^{\dagger}$ the 68 percent band excludes zero; $^{*}$ the 90 percent band excludes zero. "
        r"Width ratio: width of the CVAR band over that of the VAR without controls, at 12 months.")


def table(name, caption, label, body, extra_note=""):
    body = body[:-1] if body and body[-1] == r"\midrule" else body
    t = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r" & Lags & Impact & 1 month & 12 months & Width ratio \\", r"\midrule"] + body + [
         r"\bottomrule", r"\end{tabular}", r"\begin{minipage}{0.95\textwidth}\footnotesize",
         NOTE + extra_note, r"\end{minipage}", r"\end{table}"]
    open(os.path.join(OUT, name), "w").write("\n".join(t) + "\n")
    print(name, "written")


# main table: compact, main specification only (impact and one month)
def cells_short(df, spec, label, lags):
    z = df[df.spec == spec].set_index("h")
    if z.empty:
        return []
    r0, r1 = z.loc[0], z.loc[1]
    return [" & ".join([label, lags, f"{r0.cvar:.2f}{mark(r0)}", f"{r1.cvar:.2f}{mark(r1)}",
                        f"{r1.width_ratio_boot:.2f}"]) + r" \\",
            r"\footnotesize " + " & ".join(["", "", f"[{r0.cvar_lo:.2f}, {r0.cvar_hi:.2f}]",
                                              f"[{r1.cvar_lo:.2f}, {r1.cvar_hi:.2f}]", ""]) + r" \\[2pt]"]


rows = sum([cells_short(Bm, f"{cs}_spliced", SETS[cs][1], lagtxt(Bm, f"{cs}_spliced")) for cs in SETS], [])
tmain = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Pass-through of the LCFS cost to the California retail gasoline price\label{tab:california_main}}",
         r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Controls & Lags & Impact & 1 month & Width ratio \\", r"\midrule"] + rows + [
         r"\bottomrule", r"\end{tabular}", r"\begin{minipage}{0.95\textwidth}\footnotesize",
         NOTE.replace("at 12 months", "at one month") +
         r" The full paths are in Figure \ref{fig:california_main}, and robustness checks in Tables \ref{tab:california_robust} and \ref{tab:california_appendix}.",
         r"\end{minipage}", r"\end{table}"]
open(os.path.join(OUT, "tab_california_main.tex"), "w").write("\n".join(tmain) + "\n")
print("tab_california_main.tex written")

# robustness table (supplement): main specification, samples, leaving one PADD out, at 0, 1 and 12 months
body = block("Main specification", sum([cells(Bm, f"{cs}_spliced", SETS[cs][1], lagtxt(Bm, f"{cs}_spliced")) for cs in SETS], []))
smp = {"2017_end": "2017 onward", "2020_end": "2020 onward", "2014_2024": "2014 to 2024"}
s_lines = []
for cs in ["gulf", "padd1_4"]:
    for k, lab in smp.items():
        s_lines += cells(Ds, f"{cs}_{k}_rule", f"{SETS[cs][1]}, {lab}", lagtxt(Ds, f"{cs}_{k}_rule"))
    s_lines += cells(Ds, f"{cs}_dummies2020_22_p2", f"{SETS[cs][1]}, 2020 to 2022 dummied out", "2/2")
body += block("Samples", s_lines)
e_lines = []
for drop, lab in REGION.items():
    e_lines += cells(Ec, f"without_{drop}", f"PADDs 1 to 4 without {lab}", lagtxt(Ec, f"without_{drop}"))
body += block("Leaving one PADD out", e_lines)
table("tab_california_robust.tex", "Pass-through of the LCFS cost: samples and controls",
      "tab:california_robust", body)

# appendix table
body = []
lag_lines = []
for cs in ["gulf", "padd1_4"]:
    for p in [1, 2, 4, 6, 8, 12]:
        lag_lines += cells(Cl, f"{cs}_p{p}", f"{SETS[cs][1]}, lag order {p}", f"{p}/{p}")
body += block("Lag order (both models)", lag_lines)
MEAS = {"spliced_jan": "spliced, January dummy", "tv": "CARB formula only", "const": "constant factor"}
m_lines = []
for cs in ["gulf", "padd1_4"]:
    for k, lab in MEAS.items():
        m_lines += cells(Bm, f"{cs}_{k}", f"{SETS[cs][1]}, {lab}", lagtxt(Bm, f"{cs}_{k}"))
body += block("Policy measure", m_lines)
a_lines = []
for cs in ["gulf", "padd1_4"]:
    for lt in ["rule", "p2"]:
        z = Ds[Ds.spec == f"{cs}_2014_2019_{lt}"]
        if not z.empty:
            a_lines += cells(Ds, f"{cs}_2014_2019_{lt}", f"{SETS[cs][1]}, 2014 to 2019",
                             lagtxt(Ds, f"{cs}_2014_2019_{lt}"), note=("$^{a}$" if z.branch.iloc[0] == "levels" else ""))
body += block("Sample 2014 to 2019", a_lines)
LPK = [("linear", "coef1", "linear"), ("increase_decrease", "coef1", "increases"),
       ("increase_decrease", "coef2", "decreases"), ("high_low_credit", "coef1", "high credit price"),
       ("high_low_credit", "coef2", "low credit price")]
l_lines = []
for cs in ["gulf", "padd1_4"]:
    for kind, term, lab in LPK:
        z = Fl[(Fl.controls == cs) & (Fl.outcome == "cvar_cycle") & (Fl.p == 2) & (Fl.kind == kind) & (Fl.term == term)].set_index("h")
        if z.empty:
            continue
        top, bot = [f"{SETS[cs][1]}, local projection, {lab}", "2"], ["", ""]
        for h in (0, 1, 12):
            r = z.loc[h]
            top.append(f"{r.coef:.2f}" + ("$^{*}$" if (r.lo > 0 or r.hi < 0) else ""))
            bot.append(f"[{r.lo:.2f}, {r.hi:.2f}]")
        top.append("")
        bot.append("")
        l_lines += [" & ".join(top) + r" \\", r"\footnotesize " + " & ".join(bot) + r" \\[2pt]"]
body += block("Local projections on the cycle", l_lines)
table("tab_california_appendix.tex",
      "Pass-through of the LCFS cost: lag orders, policy measures, the 2014 to 2019 sample and local projections",
      "tab:california_appendix", body,
      extra_note=(r" In 2014 to 2019 the cost per gallon implied by the CARB formula is small, so the policy "
                  r"varies little. $^{a}$ The design test rejects in this sample and the CVAR is in levels "
                  r"with estimated weights, where the response of the stationary cycle to a permanent change "
                  r"in the cost returns to zero by construction. Local projections: regression of the change "
                  r"in the cycle between $t-1$ and $t+h$ on the change in the cost (split into increases and "
                  r"decreases, or interacted with a credit price above or below the sample median at $t-1$), "
                  r"two lags of the changes in the cycle and in the cost, and a constant; 90 percent "
                  r"Newey-West bands."))
print("written to", OUT)
