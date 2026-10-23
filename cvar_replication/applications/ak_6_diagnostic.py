"""ak_6_diagnostic.py: the diagnostic of Proposition prop:wchanges for the Alaska application.

Main design: the six donors of Jones and Marinescu (2022), pre-2020, weights estimated in changes (as ak_1_application.py).
For each outcome:
  A. Reproduction of the main common movement. Principal components of the changes of all 50 other states (after the
     constant and the lags of the dividend and of the oil price are projected out) estimate the common movements. For
     the estimated weights and for equal weights on the six donors, the ratio r of the loading of the weighted donors
     to the loading of Alaska on each component is the share of that movement the weights reproduce.
  B. Adjusted responses. With one dominant common movement, the term that the weights leave in the response equals
     (1 - r) / r times the response of the weighted donors (Proposition prop:wchanges and its discussion), with the same
     sign. From the VAR with Alaska and the weighted donors (Proposition prop:placebo), the adjusted response is
     difference - (1 - r)/r x controls, computed draw by draw with r held at its estimate. Reported for the estimated
     and for equal weights.
  C. Robustness pools: the CVAR with weights estimated from the 12 other states of the Census West region and from all
     50, with the same bootstrap as the main specification, and their reproduction ratios.
Requires ak_0_screen.py, ak_1_application.py, ak_5_donor_pool.py (and the all-states data it downloaded).
Outputs: results/alaska/6_diagnostic.csv, 6_responses.csv; results/paper/tab_alaska_weights.tex (replaces the
version of ak_3_exhibits.py), tab_alaska_adjusted.tex, tab_alaska_pools.tex
Usage: py ak_6_diagnostic.py [--B 999] [--workers N] [--block 12]
"""
from common import *
from multiprocessing import Pool
import ak_0_screen as s0
import ak_5_donor_pool as s5
from ak_1_application import design, w_nnls, mbb, fit, col

H = 36
P = 12
END = "2019-12-01"
OUTC = ["epop", "ur", "lfpr", "nonfarm", "leisure", "retail"]
SIX = s0.SETS_AK["jm_emp"]
LAB = {"epop": "Employment-population ratio", "ur": "Unemployment rate", "lfpr": "Participation rate",
       "nonfarm": "Payroll employment", "leisure": "Leisure and hospitality", "retail": "Retail trade"}
SHORT = {"epop": "Employment rate", "ur": "Unemployment", "lfpr": "Participation", "nonfarm": "Payroll",
         "leisure": "Leisure", "retail": "Retail"}
HS = (12, 24)


def q(d, al):
    return np.quantile(d, [al / 2, 1 - al / 2], axis=0)


def est_full(df, cc, B, block):
    """Weights estimated in changes and the bootstrap of ak_1_application.py (same seeds)."""
    dy, X, F = design(df, cc)
    w = w_nnls(dy, X, F)
    rows = mbb(np.random.default_rng(22), len(dy), block, B)
    Wd = [w_nnls(dy[r], X[r], F[r]) for r in rows]
    r = fit(df, cc, w, P)
    n = r.U.shape[0]
    idx = np.random.default_rng(21).integers(0, n, (B, n))
    D = np.array([col(fit(df, cc, Wd[b], P)._draw(idx[b], H, no_controls=False)["cycle"]) for b in range(B)])
    return w, r, col(r.irf(H)["cycle"]), D, idx


def ratios(lam_all, idx, lam_ak, cc, w):
    lam_w = lam_all[:, [idx[s] for s in cc]] @ w
    return lam_w / lam_ak


def job(item):
    o, out, W, doil, B, block = item
    try:
        pool_all = s5.complete(out, o, s5.DONORS_ALL, END)
        dfa = s0.arrays(out, W, doil, o, pool_all, END)
        Xa = s5.partial_out(dfa, pool_all)
        ya = s5.partial_out(dfa, ["y"])[:, 0]
        Z = (Xa - Xa.mean(0)) / Xa.std(0)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        share = S ** 2 / (S ** 2).sum()
        PC = U[:, :3] * S[:3]
        lam_ak = np.linalg.lstsq(PC, ya, rcond=None)[0]
        lam_all = np.linalg.lstsq(PC, Xa, rcond=None)[0]
        ix = {s: i for i, s in enumerate(pool_all)}
        res = dict(outcome=o, diag=[], resp=[], weights=None)

        def add(spec, pt, D):
            l9, h9 = q(D, 0.10)
            l6, h6 = q(D, 0.32)
            for h in range(H + 1):
                res["resp"].append(dict(outcome=o, spec=spec, h=h, estimate=pt[h], lo90=l9[h], hi90=h9[h],
                                        lo68=l6[h], hi68=h6[h]))

        # six donors: estimated and equal weights
        cc = [s for s in SIX if s in ix]
        df = s0.arrays(out, W, doil, o, cc, END)
        w, r_w, pt_w, D_w, idx = est_full(df, cc, B, block)
        eq = np.ones(len(cc)) / len(cc)
        r_eq = fit(df, cc, eq, P)
        D_eq = np.array([col(r_eq._draw(idx[b], H, no_controls=False)["cycle"]) for b in range(B)])
        add("six_est", pt_w, D_w)
        add("six_equal", col(r_eq.irf(H)["cycle"]), D_eq)
        res["weights"] = pd.DataFrame(dict(outcome=o, state=cc, weight=w))
        for spec, rr, ww in (("est", r_w, w), ("equal", r_eq, eq)):
            rat = ratios(lam_all, ix, lam_ak, cc, ww)
            r1 = rat[0]
            m = (1 - r1) / r1 if r1 > 0.2 else np.nan
            cr = rr.controls_response(H, B=min(B, 499), seed=13)
            dpt, cpt = col(cr.point["difference"]), col(cr.point["controls"])
            dD = cr.draws["difference"].reshape(cr.draws["difference"].shape[0], H + 1, -1)[:, :, 0]
            cD = cr.draws["controls"].reshape(cr.draws["controls"].shape[0], H + 1, -1)[:, :, 0]
            add(f"six_{spec}_controls", cpt, cD)
            add(f"six_{spec}_joint", dpt, dD)
            if np.isfinite(m):
                add(f"six_{spec}_adjusted", dpt - m * cpt, dD - m * cD)
            res["diag"].append(dict(outcome=o, weights=spec, K=len(cc), sum_w=float(ww.sum()),
                                    share_pc1=float(share[0]), share_pc2=float(share[1]), share_pc3=float(share[2]),
                                    ratio_pc1=float(rat[0]), ratio_pc2=float(rat[1]), ratio_pc3=float(rat[2]),
                                    multiplier=float(m) if np.isfinite(m) else np.nan))
        # robustness pools
        for pn in ("west", "all"):
            ccp = [s for s in s5.POOLS[pn] if s in ix]
            dfp = s0.arrays(out, W, doil, o, ccp, END)
            wp, _, ptp, Dp, _ = est_full(dfp, ccp, B, block)
            add(f"{pn}_est", ptp, Dp)
            rat = ratios(lam_all, ix, lam_ak, ccp, wp)
            res["diag"].append(dict(outcome=o, weights=f"{pn}_est", K=len(ccp), sum_w=float(wp.sum()),
                                    share_pc1=float(share[0]), share_pc2=float(share[1]), share_pc3=float(share[2]),
                                    ratio_pc1=float(rat[0]), ratio_pc2=float(rat[1]), ratio_pc3=float(rat[2]),
                                    multiplier=np.nan, n_positive=int((wp > 0.005).sum())))
        return res
    except Exception as e:
        return dict(outcome=o, error=f"{type(e).__name__}: {e}")


def f2(x):
    return "" if x is None or not np.isfinite(x) else f"{x:.2f}".replace("-", "$-$")


def band(lo, hi):
    return f"[{f2(lo)}, {f2(hi)}]"


def star(lo, hi):
    return "" if lo <= 0 <= hi else "$^{*}$"


if __name__ == "__main__":
    a = args(block=dict(type=int, default=12))
    SRC, OUTP = outdir("alaska"), outdir("paper")
    out = s5.load_all()
    B_, F_ = s0.load_raw(False)
    _, W, doil = s0.build(B_, F_)
    print(f"{len(OUTC)} outcomes, B = {a.B}", flush=True)
    with Pool(a.workers) as Pp:
        res = Pp.map(job, [(o, out, W, doil, a.B, a.block) for o in OUTC])
    for r in res:
        if "error" in r:
            print("FAILED", r["outcome"], r["error"])
    res = [r for r in res if "error" not in r]
    DG = pd.DataFrame([x for r in res for x in r["diag"]])
    RS = pd.DataFrame([x for r in res for x in r["resp"]])
    WT = pd.concat([r["weights"] for r in res], ignore_index=True)
    DG.to_csv(os.path.join(SRC, "6_diagnostic.csv"), index=False)
    RS.to_csv(os.path.join(SRC, "6_responses.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nReproduction of the common movements (ratio 1: none left in the cycle)")
    print(DG.round(3).to_string(index=False))

    def g(o, spec, h):
        z = RS[(RS.outcome == o) & (RS.spec == spec) & (RS.h == h)]
        return z.iloc[0] if len(z) else None

    print("\nResponses at 12 and 24 months (90% bands)")
    for o in OUTC:
        for spec in ("six_est", "six_est_adjusted", "six_equal", "six_equal_adjusted", "west_est", "all_est"):
            cells = []
            for h in HS:
                z = g(o, spec, h)
                cells.append(f"h={h}: {z.estimate:.3f} [{z.lo90:.3f}, {z.hi90:.3f}]" if z is not None else "")
            print(f"  {o:8s} {spec:20s} " + "   ".join(cells))

    # ------------------------------------------------------------------ tables
    def wtab(name, caption, label, colspec, header, rows, note, sep=None):
        t = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + r"\label{" + label + "}}", r"\footnotesize"]
        t += [r"\setlength{\tabcolsep}{" + sep + "}"] if sep else []
        t += [r"\begin{tabular}{" + colspec + "}", r"\toprule"] + header + [r"\midrule"] + rows + [
            r"\bottomrule", r"\end{tabular}", r"\par\smallskip\parbox{0.95\textwidth}{\footnotesize " + note + "}",
            r"\end{table}"]
        open(os.path.join(OUTP, name), "w").write("\n".join(t) + "\n")
        print(name, "written")

    def head(o, ncol, first):
        return ([] if first else [r"\addlinespace"]) + [r"\multicolumn{" + str(ncol) + "}{l}{" + LAB[o] + r"} \\"]

    def dg(o, wname, k):
        z = DG[(DG.outcome == o) & (DG.weights == wname)]
        return z.iloc[0][k] if len(z) else np.nan

    # weights and diagnostics (replaces the table of ak_3_exhibits.py)
    SCR = pd.read_csv(os.path.join(SRC, "0_screen.csv"))
    scr = SCR[SCR["sample"] == "pre2020"]
    rows = []
    for c in SIX:
        vals = []
        for o in OUTC:
            z = WT[(WT.outcome == o) & (WT.state == c)]
            vals.append(f2(z.weight.iloc[0]) if len(z) else "")
        rows.append(f"{c} & " + " & ".join(vals) + r" \\")
    rows.append(r"\midrule")
    rows.append("Sum & " + " & ".join(f2(dg(o, "est", "sum_w")) for o in OUTC) + r" \\")
    rows.append(r"\midrule")
    rows.append("Common share & " + " & ".join(f2(dg(o, "est", "share_pc1")) for o in OUTC) + r" \\")
    rows.append(r"Reproduced, estimated & " + " & ".join(f2(dg(o, "est", "ratio_pc1")) for o in OUTC) + r" \\")
    rows.append(r"Reproduced, equal & " + " & ".join(f2(dg(o, "equal", "ratio_pc1")) for o in OUTC) + r" \\")

    def pdes(o, spec, ctrl):
        z = scr[(scr.outcome == o) & (scr.spec == spec) & (scr.controls == ctrl)].p
        return f2(z.iloc[0]) if len(z) else ""
    rows.append(r"\midrule")
    rows.append(r"$p$, design & " + " & ".join(pdes(o, "estimated", "jm_emp") for o in OUTC) + r" \\")
    rows.append(r"$p$, Alaska & " + " & ".join(pdes(o, "alaska_alone", "none") for o in OUTC) + r" \\")
    wtab("tab_alaska_weights.tex", "Weights on the donor states, reproduction of the common movement and tests of the design",
         "tab:ak-weights", "l" + "c" * len(OUTC), [" & " + " & ".join(SHORT[o] for o in OUTC) + r" \\"], rows,
         r"Columns: outcomes of Alaska. Non-negative weights from the regression of the change in the Alaska outcome on the "
         r"changes of the six donor states, with a constant and the current value and twelve lags of the dividend and of the "
         r"change in the oil price, January 1976 (payroll series January 1990) to December 2019. Common share: share of the variance of the changes of the 50 other states explained by their first principal "
         r"component, the main common movement. Reproduced: ratio of the loading of the weighted donors (estimated or equal weights) to that of Alaska on that component "
         r"(Proposition \ref{prop:wchanges}); one means that the weights leave none of it in the cycle. $p$, design: test of "
         r"Section \ref{sec:test} with estimated weights (null: a stochastic trend specific to Alaska). $p$, Alaska: the same "
         r"test on the Alaska outcome without controls.", sep="4pt")

    # adjusted responses (main text)
    rows = []
    for i, o in enumerate(OUTC):
        rows += head(o, 5, i == 0)
        for h in HS:
            zs = [g(o, s, h) for s in ("six_est", "six_est_adjusted", "six_equal", "six_equal_adjusted")]
            rows.append(f"{h} & " + " & ".join(f"{f2(z.estimate)}{star(z.lo90, z.hi90)}" if z is not None else ""
                                               for z in zs) + r" \\")
            rows.append(" & " + " & ".join(band(z.lo90, z.hi90) if z is not None else "" for z in zs) + r" \\")
    wtab("tab_alaska_adjusted.tex", "Responses adjusted for the common movement left in the cycle", "tab:ak-adjusted",
         "lcccc", [r" & \multicolumn{2}{c}{Estimated weights} & \multicolumn{2}{c}{Equal weights} \\",
                   r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Months & CVAR & Adjusted & CVAR & Adjusted \\"], rows,
         r"Response to a dividend of \$1,000 per person with the six donors, 90 percent bands below the estimates. CVAR: as "
         r"in Table \ref{tab:ak-main}. Adjusted: response of Alaska relative to the weighted donors in the VAR with both, "
         r"minus $(1-r)/r$ times the response of the weighted donors, with $r$ the reproduced share in Table "
         r"\ref{tab:ak-weights} (Proposition \ref{prop:wchanges}), computed in each bootstrap draw with $r$ and the weights "
         r"held at their estimates. $^{*}$: the 90 percent band excludes zero.")

    # robustness pools (supplement)
    rows = []
    for i, o in enumerate(OUTC):
        rows += head(o, 4, i == 0)
        rows.append("Reproduced & " + " & ".join(f2(dg(o, w_, "ratio_pc1")) for w_ in ("est", "west_est", "all_est")) + r" \\")
        for h in HS:
            zs = [g(o, s, h) for s in ("six_est", "west_est", "all_est")]
            rows.append(f"{h} months & " + " & ".join(f"{f2(z.estimate)}{star(z.lo90, z.hi90)}" if z is not None else ""
                                                      for z in zs) + r" \\")
            rows.append(" & " + " & ".join(band(z.lo90, z.hi90) if z is not None else "" for z in zs) + r" \\")
    wtab("tab_alaska_pools.tex", "Alaska: larger donor pools", "tab:ak-pools", "lccc",
         [r" & Six donors & West & All states \\"], rows,
         r"CVAR of Table \ref{tab:ak-main} with weights estimated in changes from the six donors of the main specification, "
         r"from the 12 other states of the Census West region, and from all 50 other states and the District of Columbia, "
         r"with 90 percent bands below the estimates. Reproduced: share of the main common movement of Alaska reproduced "
         r"by the weighted donors, as in Table \ref{tab:ak-weights}. $^{*}$: the 90 percent band excludes zero.")
    print("\nwritten to", SRC, "and", OUTP)
