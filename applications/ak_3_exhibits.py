"""ak_3_exhibits.py: figure and LaTeX tables for the Alaska Permanent Fund Dividend application.

Main specification: seasonally adjusted series, January 1976 (payroll series January 1990) to December 2019,
the six donor states of Jones and Marinescu (2022), weights estimated in changes, the oil price ordered first,
12 lags. The script re-estimates the main specification for every outcome to obtain full response paths
(same seeds as ak_1_application.py, so the numbers match 1_responses.csv) and reads the rest from:
  results/alaska/0_screen.csv     (ak_0_screen.py: test of the design)
  results/alaska/1_loading.csv, 1_weights.csv, 1_responses.csv  (ak_1_application.py, pre2020/pre2016/pre2008)
  results/alaska/2_bounds.csv, 2_seasonal.csv  (ak_2_checks.py)
Outputs in results/paper/:
  fig_alaska_main.pdf/.png/.csv  CVAR with 68 and 90 percent bands, VAR without controls with its 90 percent
                                 band, and the weighted controls, for each outcome
  tab_alaska_weights.tex         estimated weights, tests of their sum and of equal weights, test of the design
  tab_alaska_main.tex            responses at 3, 12 and 24 months
  tab_alaska_bounds.tex          bounds of Proposition prop:bounds
  tab_alaska_samples.tex         (supplement) samples ending 2019, 2015 and 2007
  tab_alaska_seasonal.tex        (supplement) five treatments of seasonality
Usage: py ak_3_exhibits.py [--B 999] [--workers N] [--block 12] [--tables_only]
  --tables_only  rebuild the tables from fig_alaska_main.csv without re-estimating (no figure)
"""
from common import *
from multiprocessing import Pool
import ak_0_screen as s0
from ak_1_application import design, w_nnls, mbb, fit, col

H = 36
END = "2019-12-01"
CC = s0.SETS_AK["jm_emp"]
P = 12
OUTC = ["epop", "ur", "lfpr", "nonfarm", "leisure", "retail"]
LAB = {"epop": "Employment-population ratio", "ur": "Unemployment rate", "lfpr": "Participation rate",
       "nonfarm": "Payroll employment", "leisure": "Leisure and hospitality", "retail": "Retail trade"}
UNIT = {"epop": "percentage points", "ur": "percentage points", "lfpr": "percentage points",
        "nonfarm": "percent", "leisure": "percent", "retail": "percent"}
SEAS = [("SA", "Adjusted"), ("SA + months", "Adjusted, month effects"), ("NSA + months", "Unadjusted, month effects"),
        ("NSA + months x trend", r"Unadjusted, month effects $\times$ trend"),
        ("NSA + months x decade", "Unadjusted, month effects by decade")]


def f2(x):
    return "" if x is None or not np.isfinite(x) else f"{x:.2f}".replace("-", "$-$")


def band(lo, hi):
    return f"[{f2(lo)}, {f2(hi)}]"


def star(lo, hi):
    return "" if lo <= 0 <= hi else "$^{*}$"


def job_main(item):
    o, df, B, block = item
    dy, X, F = design(df, CC)
    w = w_nnls(dy, X, F)
    rows = mbb(np.random.default_rng(22), len(dy), block, B)
    Wd = np.array([w_nnls(dy[r], X[r], F[r]) for r in rows])
    r = fit(df, CC, w, P)
    n = r.U.shape[0]
    idx = np.random.default_rng(21).integers(0, n, (B, n))
    D = {k: np.zeros((B, H + 1)) for k in ("cycle", "ice", "policy", "var")}
    for b in range(B):
        d = fit(df, CC, Wd[b], P)._draw(idx[b], H, no_controls=False)
        for k in ("cycle", "ice", "policy"):
            D[k][b] = col(d[k])
        D["var"][b] = col(r._nc_draw(idx[b], H))
    pt = r.irf(H)
    PT = {k: col(pt[k]) for k in ("cycle", "ice", "policy")}
    PT["var"] = col(r.irf_no_controls(H))
    cr = r.controls_response(H, B=min(B, 499), seed=13)
    for k in ("treated", "controls"):
        PT["cr_" + k] = col(cr.point[k])
        D["cr_" + k] = cr.draws[k].reshape(cr.draws[k].shape[0], H + 1, -1)[:, :, 0]
    return o, PT, D


if __name__ == "__main__":
    a = args(block=dict(type=int, default=12), tables_only=dict(action="store_true"))
    SRC, OUT = outdir("alaska"), outdir("paper")
    need = ["0_screen.csv", "1_loading.csv", "1_weights.csv", "1_responses.csv", "2_bounds.csv", "2_seasonal.csv"]
    miss = [f for f in need if not os.path.exists(os.path.join(SRC, f))]
    if miss:
        sys.exit(f"missing in {SRC}: {miss}; run ak_0, ak_1 (with --samples pre2020 pre2016 pre2008) and ak_2 first")
    SCR = pd.read_csv(os.path.join(SRC, "0_screen.csv"))
    LD = pd.read_csv(os.path.join(SRC, "1_loading.csv"))
    WT = pd.read_csv(os.path.join(SRC, "1_weights.csv"))
    RS = pd.read_csv(os.path.join(SRC, "1_responses.csv"))
    BD = pd.read_csv(os.path.join(SRC, "2_bounds.csv"))
    SS = pd.read_csv(os.path.join(SRC, "2_seasonal.csv"))
    sel = lambda d: d[(d.controls == "jm_emp") & (d["sample"] == "pre2020") & (d.pmode == "p12")]

    FIG = os.path.join(OUT, "fig_alaska_main.csv")
    if not a.tables_only:
        # ------------------------------------------------------------------ re-estimation of the main specification
        B_, F_ = s0.load_raw(False)
        out, W, doil = s0.build(B_, F_)
        items = [(o, s0.arrays(out, W, doil, o, CC, END), a.B, a.block) for o in OUTC]
        print(f"re-estimating {len(items)} outcomes, B = {a.B}", flush=True)
        with Pool(a.workers) as Pp:
            res = Pp.map(job_main, items)
        R = {o: (PT, D) for o, PT, D in res}
        q = lambda d, al: np.quantile(d, [al / 2, 1 - al / 2], axis=0)
        csv = {"h": np.arange(H + 1)}
        for o in OUTC:
            PT, D = R[o]
            for k in PT:
                lo, hi = q(D[k], 0.10)
                csv[f"{o}_{k}"], csv[f"{o}_{k}_lo90"], csv[f"{o}_{k}_hi90"] = PT[k], lo, hi
            lo, hi = q(D["cycle"], 0.32)
            csv[f"{o}_cycle_lo68"], csv[f"{o}_cycle_hi68"] = lo, hi
        pd.DataFrame(csv).to_csv(os.path.join(OUT, "fig_alaska_main.csv"), index=False)

        # ------------------------------------------------------------------ figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            hh = np.arange(H + 1)
            fig, axs = plt.subplots(2, 3, figsize=(13, 7.2))
            for ax, o in zip(axs.ravel(), OUTC):
                PT, D = R[o]
                l9, h9 = q(D["cycle"], 0.10)
                l6, h6 = q(D["cycle"], 0.32)
                v9 = q(D["var"], 0.10)
                ax.fill_between(hh, l9, h9, color="C0", alpha=0.18, lw=0, label="CVAR, 90%")
                ax.fill_between(hh, l6, h6, color="C0", alpha=0.35, lw=0, label="CVAR, 68%")
                ax.plot(hh, PT["cycle"], color="C0", lw=1.8, label="CVAR")
                ax.plot(hh, PT["var"], color="C3", lw=1.2, ls="--", label="VAR without controls")
                ax.plot(hh, v9[0], color="C3", lw=0.7, ls=":")
                ax.plot(hh, v9[1], color="C3", lw=0.7, ls=":", label="VAR, 90%")
                ax.plot(hh, PT["cr_controls"], color="C2", lw=1.2, ls="-.", label="Weighted controls")
                ax.axhline(0, color="k", lw=0.6)
                ax.set_title(LAB[o], fontsize=10)
                ax.set_ylabel(UNIT[o], fontsize=9)
                ax.set_xlabel("months", fontsize=9)
            axs[0, 0].legend(fontsize=7, loc="lower left")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(OUT, f"fig_alaska_main.{ext}"), dpi=200)
            plt.close(fig)
            print("fig_alaska_main written")
        except ImportError:
            print("matplotlib not installed: figure skipped")

    if not os.path.exists(FIG):
        sys.exit(f"{FIG} not found: run without --tables_only first")
    C = pd.read_csv(FIG)

    # ------------------------------------------------------------------ tables
    def wtab(name, caption, label, colspec, header, rows, note, sep=None):
        t = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + r"\label{" + label + "}}", r"\footnotesize"]
        t += [r"\setlength{\tabcolsep}{" + sep + "}"] if sep else []
        t += [r"\begin{tabular}{" + colspec + "}", r"\toprule"] + header + [r"\midrule"] + rows + [
             r"\bottomrule", r"\end{tabular}", r"\par\smallskip\parbox{0.95\textwidth}{\footnotesize " + note + "}",
             r"\end{table}"]
        open(os.path.join(OUT, name), "w").write("\n".join(t) + "\n")
        print(name, "written")

    def head(o, ncol, first):
        return ([] if first else [r"\addlinespace"]) + [r"\multicolumn{" + str(ncol) + "}{l}{" + LAB[o] + r"} \\"]

    def g(o, k, h):
        return C[f"{o}_{k}"].iloc[h], C[f"{o}_{k}_lo90"].iloc[h], C[f"{o}_{k}_hi90"].iloc[h]

    SHORT = {"epop": "Employment rate", "ur": "Unemployment", "lfpr": "Participation", "nonfarm": "Payroll",
             "leisure": "Leisure", "retail": "Retail"}

    # weights and tests (outcomes in columns)
    ld, wt = sel(LD), sel(WT)
    scr = SCR[SCR["sample"] == "pre2020"]
    L1 = {o: ld[ld.outcome == o].iloc[0] for o in OUTC}
    rows = []
    for c in CC:
        vals = []
        for o in OUTC:
            ww = wt[wt.outcome == o].set_index("control").w_nnls
            vals.append(f2(ww.get(c, np.nan)))
        rows.append(f"{c} & " + " & ".join(vals) + r" \\")
    rows.append(r"\midrule")
    rows.append("Sum & " + " & ".join(f2(L1[o].sum_nnls) for o in OUTC) + r" \\")
    rows.append(" & " + " & ".join(band(L1[o].sum_nnls_lo90, L1[o].sum_nnls_hi90) for o in OUTC) + r" \\")
    rows.append(r"$p$, sum $=1$ & " + " & ".join(f2(L1[o].p_sum1) for o in OUTC) + r" \\")
    rows.append(r"$p$, equal & " + " & ".join(f2(L1[o].p_equal) for o in OUTC) + r" \\")

    def pdes(o, spec, ctrl):
        z = scr[(scr.outcome == o) & (scr.spec == spec) & (scr.controls == ctrl)].p
        return f2(z.iloc[0]) if len(z) else ""
    rows.append(r"\midrule")
    rows.append(r"$p$, design & " + " & ".join(pdes(o, "estimated", "jm_emp") for o in OUTC) + r" \\")
    rows.append(r"$p$, Alaska & " + " & ".join(pdes(o, "alaska_alone", "none") for o in OUTC) + r" \\")
    wtab("tab_alaska_weights.tex", "Weights on the donor states and tests of the design", "tab:ak-weights",
         "l" + "c" * len(OUTC), [" & " + " & ".join(SHORT[o] for o in OUTC) + r" \\"], rows,
         r"Columns: outcomes of Alaska. Non-negative weights from the regression of the change in the Alaska outcome on "
         r"the changes of the donor states, with a constant and the current value and twelve lags of the dividend and of "
         r"the change in the oil price, January 1976 (payroll series January 1990) to December 2019. Sum: estimate and "
         r"90 percent interval from a moving-block bootstrap of the regression. $p$, sum $=1$ and $p$, equal: "
         r"tests of a unit sum and of equal weights from the unconstrained regression with HAC standard errors. $p$, design: test of Section "
         r"\ref{sec:test} with estimated weights (null: a stochastic trend specific to Alaska). $p$, Alaska: the "
         r"same test on the Alaska outcome without controls.", sep="4pt")

    # main responses
    rs = sel(RS)
    rows = []
    for i, o in enumerate(OUTC):
        rows += head(o, 6, i == 0)
        for h in (3, 12, 24):
            c, v, s_ = g(o, "cycle", h), g(o, "var", h), g(o, "cr_controls", h)
            eq = rs[(rs.outcome == o) & (rs.spec == "equal") & (rs.object == "cycle") & (rs.h == h)].estimate
            rows.append(f"{h} & {f2(c[0])}{star(c[1], c[2])} & {f2(eq.iloc[0]) if len(eq) else ''} & "
                        f"{f2(C[o + '_ice'].iloc[h])} & {f2(v[0])}{star(v[1], v[2])} & {f2(s_[0])}{star(s_[1], s_[2])} \\\\")
            rows.append(f" & {band(c[1], c[2])} & & & {band(v[1], v[2])} & {band(s_[1], s_[2])} \\\\")
    wtab("tab_alaska_main.tex", r"Response to a dividend of \$1,000 per person", "tab:ak-main", "lccccc",
         [r"Months & CVAR & Equal & Impulse causal & VAR without & Weighted \\",
          r" & & weights & effect & controls & controls \\"], rows,
         r"Cumulated response to an innovation of \$1,000 (2024 dollars) per person in the dividend, in percentage "
         r"points (rates) or percent (employment), with 90 percent bands below the estimates. CVAR: VAR in the change in "
         r"the oil price, the dividend and the change in the cycle, with the weights of Table \ref{tab:ak-weights}; bands "
         r"from a residual bootstrap that re-estimates the weights in each draw on a moving-block resample. Equal "
         r"weights: the same with weights $1/6$. Impulse causal effect: Theorem \ref{thm:direct}, later dividends held "
         r"fixed. VAR without controls: the same with the change in the Alaska outcome. Weighted controls: response in "
         r"the VAR with Alaska and the weighted controls (Proposition \ref{prop:placebo}). $^{*}$: the 90 percent band "
         r"excludes zero.")

    # bounds
    rows = []
    first = True
    for o in ["epop", "lfpr", "nonfarm", "leisure"]:
        z0 = BD[BD.outcome == o]
        if not len(z0):
            continue
        rows += head(o, 7, first)
        first = False
        for h in (12, 24):
            z = z0[z0.h == h]
            if not len(z):
                continue
            z = z.iloc[0]
            rows.append(f"{h} & {f2(z.difference)} & {f2(z.controls)} & "
                        + " & ".join(band(z[f"lam{l:g}_lower"], z[f"lam{l:g}_upper"]) for l in (0.0, 0.5, 1.0))
                        + f" & {f2(z.zero_lambda) if np.isfinite(z.zero_lambda) else '$>3$'} \\\\")
    wtab("tab_alaska_bounds.tex", "Sensitivity to a bias specific to Alaska", "tab:ak-bounds", "lcccccc",
         [r"Months & CVAR & Weighted & $\lambda=0$ & $\lambda=0.5$ & $\lambda=1$ & $\lambda$ reaching \\",
          r" & & controls & & & & zero \\"], rows,
         r"From the VAR with Alaska and the weighted controls (Proposition \ref{prop:placebo}), 90 percent bands for "
         r"the bounds of Proposition \ref{prop:bounds}, with the bias specific to Alaska at most $\lambda$ times the "
         r"bias from the common trends, measured by the response of the weighted controls. Weights held at their "
         r"estimates. Last column: smallest $\lambda$ at which the band contains zero.")

    # samples (supplement)
    rows = []
    for i, o in enumerate(OUTC):
        rows += head(o, 4, i == 0)
        for h in (12, 24):
            est, bnd = [], []
            for smp in ("pre2020", "pre2016", "pre2008"):
                z = RS[(RS.outcome == o) & (RS["sample"] == smp) & (RS.controls == "jm_emp") & (RS.pmode == "p12")
                       & (RS.spec == "est_full") & (RS.object == "cycle") & (RS.h == h)]
                if len(z):
                    z = z.iloc[0]
                    est.append(f"{f2(z.estimate)}{star(z.lo90, z.hi90)}")
                    bnd.append(band(z.lo90, z.hi90))
                else:
                    est.append("")
                    bnd.append("")
            rows.append(f"{h} & " + " & ".join(est) + r" \\")
            rows.append(" & " + " & ".join(bnd) + r" \\")
    wtab("tab_alaska_samples.tex", "Alaska: samples ending before the legislated dividends and before 2008",
         "tab:ak-samples", "lccc", [r"Months & To 2019 & To 2015 & To 2007 \\"], rows,
         r"CVAR of Table \ref{tab:ak-main} estimated on samples ending in December 2019 (main), December 2015 (before "
         r"the legislature set the amounts) and December 2007 (before the 2008 dividend, the energy rebate and the "
         r"recession), with 90 percent bands below the estimates. Rates from January 1976, payroll series from January "
         r"1990. Weights re-estimated in each sample. $^{*}$: the 90 percent band excludes zero.")

    # seasonality (supplement)
    rows = []
    ss = SS[SS.spec == "cvar"]
    for i, o in enumerate(OUTC):
        rows += head(o, 6, i == 0)
        for v, vl in SEAS:
            gg = ss[(ss.outcome == o) & (ss.version == v)]
            if not len(gg):
                continue
            cells = []
            for h in (1, 3, 12, 24):
                z = gg[gg.h == h].iloc[0]
                cells.append(f"{f2(z.estimate)}{star(z.lo90, z.hi90)}")
            rows.append(f"{vl} & {f2(gg.sum_weights.iloc[0])} & " + " & ".join(cells) + r" \\")
    wtab("tab_alaska_seasonal.tex", "Alaska: treatment of seasonality", "tab:ak-seasonal", "lccccc",
         [r"Series & Sum of & \multicolumn{4}{c}{Months} \\", r"\cmidrule(lr){3-6}",
          r" & weights & 1 & 3 & 12 & 24 \\"], rows,
         r"CVAR of Table \ref{tab:ak-main}, January 1976 (payroll series January 1990) to December 2019, with the "
         r"seasonally adjusted series (main), the adjusted series with month effects, and the unadjusted series with "
         r"month effects, with month effects interacted with a linear trend, and with month effects by decade, in the "
         r"weights regression and in the VAR. Weights re-estimated in each version and held at their estimates in the "
         r"residual bootstrap. $^{*}$: the 90 percent band excludes zero.")
    print("\nwritten to", OUT)
