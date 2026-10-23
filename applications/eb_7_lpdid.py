"""eb_7_lpdid.py: local-projection difference-in-differences (Dube, Girardi, Jorda and Taylor 2023)
on the EB panel, beside the CVAR, for the supplementary appendix.

Panel: the treated states (at least twelve months of EB) and the seven never-triggered controls,
October 2002 to December 2019, monthly.
Event: EB switches on in a state (weeks of EB positive in month t, zero in each of the twelve
previous months).
For each horizon h = 0, 6, 12, 24, 36:
    y_{i,t+h} - y_{i,t-1} = beta_h * event_{it} + delta_t
                            + sum_{l=1..4} g_l dy_{i,t-l} + sum_{l=0..4} k_l dEUC_{i,t-l} + e,
estimated on the event observations and on control observations, with two definitions of controls:
    never : the never-triggered states
    clean : any state-month with no EB in the twelve months before t and in the months t to t+h
            (the clean controls of Dube et al., which include treated states outside their spells).
delta_t are month fixed effects; standard errors are clustered by state; 90 percent bands.
For comparison with the CVAR, whose response is to an EB innovation of ten weeks, beta_h is also
divided by the average size of the switch-on in tens of weeks.
Requires eb_5_average.py (results/eb/5_average.csv) for the CVAR columns.
Outputs: results/eb/7_lpdid.csv and results/paper/tab_eb_lpdid.tex
Usage: py eb_7_lpdid.py
"""
from common import *
import statsmodels.api as sm

HOR = (0, 6, 12, 24, 36)
UR, EB, EUC, elig = load_eb()
states = list(elig) + CTRL
Y, W, E = UR[states], EB[states], EUC[states]      # EB and EUC in tens of weeks
back = W.shift(1).rolling(12, min_periods=12).max()
event = (W > 0) & (back == 0)
never = pd.DataFrame(np.tile([s in CTRL for s in states], (len(W), 1)), index=W.index, columns=states)
dY, dE = Y.diff(), E.diff()


def long(df, name):
    """Wide (date x state) to long, indexed by (date, state), keeping missing values."""
    m = df.rename_axis("date").rename_axis("st", axis=1).reset_index().melt(
        id_vars="date", var_name="st", value_name=name)
    return m.set_index(["date", "st"])[name]


rows = []
size = float(W[event].stack().mean())               # average switch-on, tens of weeks
for h in HOR:
    lhs = Y.shift(-h) - Y.shift(1)
    fwd = W.iloc[::-1].rolling(h + 1, min_periods=h + 1).max().iloc[::-1]
    clean = (back == 0) & (fwd == 0)
    parts = [long(lhs, "lhs"), long(event.astype(float), "event"), long(never.astype(float), "never"),
             long(clean.astype(float), "clean")]
    parts += [long(dY.shift(l), f"dy{l}") for l in range(1, 5)]
    parts += [long(dE.shift(l), f"de{l}") for l in range(0, 5)]
    D = pd.concat(parts, axis=1).dropna()
    D.index.names = ["date", "st"]
    for ctrl in ("never", "clean"):
        S = D[(D.event == 1) | ((D[ctrl] == 1) & (D.event == 0))].reset_index()
        X = pd.concat([S[["event"] + [f"dy{l}" for l in range(1, 5)] + [f"de{l}" for l in range(0, 5)]],
                       pd.get_dummies(S["date"], prefix="t", drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X, has_constant="add")
        groups = pd.factorize(S["st"])[0]
        m = sm.OLS(S["lhs"].values, X.values).fit(cov_type="cluster", cov_kwds={"groups": groups})
        b, se = m.params[1], m.bse[1]
        rows.append(dict(h=h, controls=ctrl, beta=b, lo=b - 1.645 * se, hi=b + 1.645 * se,
                         beta_per10=b / size, lo_per10=(b - 1.645 * se) / size, hi_per10=(b + 1.645 * se) / size,
                         events=int(S.event.sum()), control_obs=int((S.event == 0).sum()),
                         control_states=int(S.loc[S.event == 0, "st"].nunique())))
    print(f"h={h}: " + "; ".join(f"{r['controls']} {r['beta']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
                                 for r in rows[-2:]), flush=True)
R = pd.DataFrame(rows)
R["avg_switch_on_tens_of_weeks"] = size
R.to_csv(os.path.join(outdir("eb"), "7_lpdid.csv"), index=False)

# ------------------------------------------------------------------ table with the CVAR
# CVAR columns: the same numbers as Table tab:eb-avg (results/paper/fig_eb_average.csv from eb_6_exhibits.py),
# or results/eb/5_average.csv if eb_6 has not been run
fa = os.path.join(outdir("paper"), "fig_eb_average.csv")
av = os.path.join(outdir("eb"), "5_average.csv")
A = None
if os.path.exists(fa):
    F = pd.read_csv(fa)
    A = pd.concat([pd.DataFrame({"object": k, "h": F.h, "estimate": F[k], "lower": F[f"{k}_lo90"], "upper": F[f"{k}_hi90"]})
                   for k in ("cycle", "no_controls")])
elif os.path.exists(av):
    A = pd.read_csv(av)
else:
    print("CVAR columns left empty: run eb_6_exhibits.py (or eb_5_average.py) first")


def f2(x):
    return f"{x:.2f}".replace("-", "$-$")


def cell(est, lo, hi):
    return f"{f2(est)} [{f2(lo)}, {f2(hi)}]"


lines = []
for h in HOR:
    c = [str(h)]
    for ctrl in ("never", "clean"):
        r = R[(R.h == h) & (R.controls == ctrl)].iloc[0]
        c.append(cell(r.beta_per10, r.lo_per10, r.hi_per10))
    for obj in ("cycle", "no_controls"):
        if A is not None and ((A.object == obj) & (A.h == h)).any():
            z = A[(A.object == obj) & (A.h == h)].iloc[0]
            c.append(cell(z.estimate, z.lower, z.upper))
        else:
            c.append("")
    lines.append(" & ".join(c) + r" \\")
ev = int(R[(R.h == 0) & (R.controls == "never")].events.iloc[0])
tab = [r"\begin{table}[htbp]", r"\centering",
       r"\caption{Extended benefits: local-projection difference-in-differences and the CVAR\label{tab:eb-lpdid}}",
       r"\small", r"\begin{tabular}{ccccc}", r"\toprule",
       r" & \multicolumn{2}{c}{LP difference-in-differences} & \multicolumn{2}{c}{VAR responses} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r"Months & Never-triggered controls & Clean controls & CVAR & VAR without controls \\", r"\midrule"] + lines + [
       r"\bottomrule", r"\end{tabular}",
       r"\par\smallskip\parbox{0.95\textwidth}{\footnotesize " +
       rf"Change in the unemployment rate in percentage points. LP difference-in-differences \citep{{dube2023local}}: "
       rf"response to EB switching on in a state ({ev} events with no EB in the previous twelve months), with month "
       rf"fixed effects, four lags of the change in the unemployment rate and the current value and four lags of the "
       rf"change in the weeks of EUC; controls are the never-triggered states, or every state-month without EB in the "
       rf"twelve months before the event and up to the horizon. The coefficients are divided by the average size of a "
       rf"switch-on, {size:.2f} tens of weeks, to express them per ten weeks; 90 percent bands from standard errors "
       rf"clustered by state. CVAR and VAR without controls: average response to an EB innovation of ten weeks, from "
       rf"Table \ref{{tab:eb-avg}}.}}",
       r"\end{table}"]
open(os.path.join(outdir("paper"), "tab_eb_lpdid.tex"), "w").write("\n".join(tab) + "\n")
print(R.round(3).to_string(index=False))
print("written results/eb/7_lpdid.csv and results/paper/tab_eb_lpdid.tex")
