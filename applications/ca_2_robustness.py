"""ca_2_robustness.py: pass-through of the LCFS cost to the California retail gasoline price.

Outcome: California retail regular gasoline, cents per gallon, monthly average of EIA weekly prices,
calendar-month effects removed from the changes. Controls: Gulf Coast; PADDs 1 to 4; Texas and
Florida (equal weights, differenced branch: the design test does not reject).
Policy (main, 'spliced'): LCFS cost per gallon, CARB formula before the first CEC month and the
refiners' reported cost (CEC) from then on, in monthly changes; the change at the splice month is set
to zero; the sample ends at the last CEC month. Also 'tv' (CARB formula only) and 'const' (constant
factor) in section B. Requires ca_1_cec.py.
Sections (results/california/):
  B_main.csv     control sets x policy measures (spliced, spliced with a January dummy, tv, tv with a
                 January dummy, const): CVAR and VAR without controls, percentile (90 and 68 percent),
                 Hall and asymptotic (delta-method) bands, width ratios; design test p-values
  C_lags.csv     both models at every lag order 1 to 12 (Gulf, PADDs 1 to 4)
  D_samples.csv  2014-2019, 2014-2024, 2017 onward, 2020 onward, full, and full with 2020-2022 dummied
  E_controls.csv PADDs 1 to 4 leaving one region out
  F_lp.csv       local projections (Newey-West): linear, increases and decreases, high and low credit price
Usage: py ca_2_robustness.py [--B 999] [--sections B C D E F]
"""
from common import *

a = args(sections=dict(nargs="*", default=list("BCDEF")))
OUT = outdir("california")
H = 24
HOR = (0, 1, 2, 3, 6, 12, 24)
D, FIRST_CEC = load_california()
MAIN = "spliced"
SETS = {k: v[0] for k, v in globals()["SETS"].items()}
print(f"sample {D.index[0]:%Y-%m} to {D.index[-1]:%Y-%m}; CEC cost from {FIRST_CEC:%Y-%m}", flush=True)


def series(df, cc, measure):
    return ca_series(df, cc, measure, FIRST_CEC)


# ------------------------------------------------------------------ estimation helpers
def asymptotic(Z, p, exog, cum):
    from statsmodels.tsa.api import VAR
    res = VAR(Z, exog=exog).fit(p, trend="c")
    ir = res.irf(H)
    if cum:
        pt, se = ir.orth_cum_effects[:, 1, 0], ir.cum_effect_stderr(orth=True)[:, 1, 0]
    else:
        pt, se = ir.orth_irfs[:, 1, 0], ir.stderr(orth=True)[:, 1, 0]
    s = ir.orth_irfs[0, 0, 0]
    return pt / s, (pt - 1.645 * se) / s, (pt + 1.645 * se) / s


def fits(y, C, dose, pc, pv, ve, branch):
    w = np.ones(C.shape[1]) / C.shape[1]
    if branch == "levels":
        rc = cvar.CVAR(p=pc, q=12).fit(y, C, dose, var_exog=ve)
    else:
        rc = cvar.CVAR(p=pc).fit(y, C, dose, var_exog=ve, weights=w, difference=True)
    rv = cvar.CVAR(p=pv).fit(y, C, dose, var_exog=ve, weights=w, difference=True)
    return rc, rv


def estimate(y, C, dose, pc, pv, B, ve=None, branch="differenced", seed=1, asy=True):
    rc, rv = fits(y, C, dose, pc, pv, ve, branch)
    bc = rc.bootstrap(B, H, seed=seed, no_controls=False)
    bv = rv.bootstrap(B, H, seed=seed + 1, no_controls=True)
    out = {}
    for nm, bt, key in [("cvar", bc, "cycle"), ("var", bv, "no_controls")]:
        lo, hi = bt.bands(key, 0.10, "percentile")
        hl, hh = bt.bands(key, 0.10, "hall")
        l6, h6 = bt.bands(key, 0.32, "percentile")
        out[nm] = dict(pt=bt.point[key][:, 0], lo=lo[:, 0], hi=hi[:, 0], hlo=hl[:, 0], hhi=hh[:, 0],
                       lo68=l6[:, 0], hi68=h6[:, 0])
    if asy:
        try:
            out["cvar"].update(zip(["apt", "alo", "ahi"],
                                   asymptotic(rc.Z, pc, rc.var_exog, cum=(branch == "differenced"))))
            out["var"].update(zip(["apt", "alo", "ahi"], asymptotic(rv._nc_fit()[0], pv, rv.var_exog, cum=True)))
        except Exception as e:
            print("  asymptotic bands failed:", str(e)[:80])
    return out


def rows(tag, est, **extra):
    r = []
    for h in HOR:
        row = dict(extra, spec=tag, h=h)
        for nm in ("cvar", "var"):
            e = est[nm]
            row.update({f"{nm}": e["pt"][h], f"{nm}_lo": e["lo"][h], f"{nm}_hi": e["hi"][h],
                        f"{nm}_hall_lo": e["hlo"][h], f"{nm}_hall_hi": e["hhi"][h],
                        f"{nm}_lo68": e["lo68"][h], f"{nm}_hi68": e["hi68"][h]})
            if "alo" in e:
                row.update({f"{nm}_asy_lo": e["alo"][h], f"{nm}_asy_hi": e["ahi"][h]})
        row["width_ratio_boot"] = (row["cvar_hi"] - row["cvar_lo"]) / (row["var_hi"] - row["var_lo"])
        if "cvar_asy_lo" in row and "var_asy_lo" in row:
            row["width_ratio_asy"] = (row["cvar_asy_hi"] - row["cvar_asy_lo"]) / (row["var_asy_hi"] - row["var_asy_lo"])
        row["cvar_sig_boot"] = row["cvar_lo"] > 0 or row["cvar_hi"] < 0
        row["var_sig_boot"] = row["var_lo"] > 0 or row["var_hi"] < 0
        row["cvar_sig68"] = row["cvar_lo68"] > 0 or row["cvar_hi68"] < 0
        row["var_sig68"] = row["var_lo68"] > 0 or row["var_hi68"] < 0
        r.append(row)
    return r


def rule_lags(y, C, dose, ve=None):
    rc, rv = fits(y, C, dose, 1, 1, ve, "differenced")
    return (lag_rule(rc.Z, rc.var_exog, [0, 1], 12), lag_rule(rv._nc_fit()[0], rv.var_exog, [0, 1], 12))


def design_p(y, C, dose, B):
    return cvar.design_test(y, C, dose[:, None], q=12, B=B, seed=2).pvalue_bootstrap


def plot_bands(path, est, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    hh = np.arange(H + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for k, (nm, ttl) in enumerate([("cvar", "CVAR"), ("var", "VAR without controls")]):
        e = est[nm]
        ax[k].fill_between(hh, e["lo"], e["hi"], alpha=0.18, label="bootstrap 90%")
        ax[k].fill_between(hh, e["lo68"], e["hi68"], alpha=0.30, label="bootstrap 68%")
        if "alo" in e:
            ax[k].plot(hh, e["alo"], "--", lw=0.9, color="k", label="asymptotic")
            ax[k].plot(hh, e["ahi"], "--", lw=0.9, color="k")
        ax[k].plot(hh, e["pt"])
        ax[k].axhline(0, lw=0.8, color="k")
        ax[k].axhline(1, lw=0.6, color="grey", ls=":")
        ax[k].set_title(f"{ttl}: {title}", fontsize=9)
        ax[k].set_xlabel("months")
    ax[0].set_ylabel("cents per 1 cent of LCFS cost")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



# ------------------------------------------------------------------ B. main
if "B" in a.sections:
    print("\nB. Main specifications", flush=True)
    R = []
    for cs in SETS:
        for meas in ["spliced", "spliced_jan", "tv", "tv_jan", "const"]:
            y, C, dose, _, idx = series(D, SETS[cs], meas.split("_")[0])
            ve = (idx.month == 1).astype(float)[:, None] if meas.endswith("_jan") else None
            pc, pv = rule_lags(y, C, dose, ve)
            dp = design_p(y, C, dose, a.B)
            est = estimate(y, C, dose, pc, pv, a.B, ve=ve)
            R += rows(f"{cs}_{meas}", est, controls=cs, measure=meas, p_cvar=pc, p_var=pv, design_p=dp)
            print(f"  {cs} {meas}: lags {pc}/{pv}, design p {dp:.3f}, h=0 CVAR {est['cvar']['pt'][0]:.2f} "
                  f"[{est['cvar']['lo'][0]:.2f}, {est['cvar']['hi'][0]:.2f}], h=12 {est['cvar']['pt'][12]:.2f}", flush=True)
            if cs == "padd1_4":
                plot_bands(os.path.join(OUT, f"B_{cs}_{meas}.png"), est, f"PADDs 1-4, {meas}")
    pd.DataFrame(R).to_csv(os.path.join(OUT, "B_main.csv"), index=False)

# ------------------------------------------------------------------ C. lags
if "C" in a.sections:
    print("\nC. Lag orders 1 to 12", flush=True)
    R = []
    for cs in ["gulf", "padd1_4"]:
        y, C, dose, _, idx = series(D, SETS[cs], MAIN)
        for p in range(1, 13):
            est = estimate(y, C, dose, p, p, max(199, a.B // 2), seed=p)
            R += rows(f"{cs}_p{p}", est, controls=cs, p=p)
            print(f"  {cs} p={p}: h=0 {est['cvar']['pt'][0]:.2f}, h=12 {est['cvar']['pt'][12]:.2f} "
                  f"[{est['cvar']['lo'][12]:.2f}, {est['cvar']['hi'][12]:.2f}]", flush=True)
    C_ = pd.DataFrame(R)
    C_.to_csv(os.path.join(OUT, "C_lags.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for k, h in enumerate([0, 3, 12]):
            for cs, mk in [("gulf", "o"), ("padd1_4", "s")]:
                z = C_[(C_.controls == cs) & (C_.h == h)]
                ax[k].errorbar(z.p + (0.1 if cs == "gulf" else -0.1), z.cvar,
                               yerr=[z.cvar - z.cvar_lo, z.cvar_hi - z.cvar], fmt=mk, ms=3, capsize=2, label=cs)
            ax[k].axhline(0, lw=0.8, color="k")
            ax[k].axhline(1, lw=0.6, color="grey", ls=":")
            ax[k].set_title(f"CVAR response at h = {h}", fontsize=9)
            ax[k].set_xlabel("lag order")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "C_lags.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

# ------------------------------------------------------------------ D. samples
if "D" in a.sections:
    print("\nD. Samples", flush=True)
    SAMPLES = {"2014_2019": ("2014-01-01", "2019-12-01"), "2014_2024": ("2014-01-01", "2024-12-01"),
               "2017_end": ("2017-01-01", None), "2020_end": ("2020-01-01", None), "full": (None, None)}
    R = []
    for cs in ["gulf", "padd1_4"]:
        for lab, (s0, s1) in SAMPLES.items():
            Ds = D.loc[s0:s1]
            y, C, dose, _, idx = series(Ds, SETS[cs], MAIN)
            dp = design_p(y, C, dose, a.B)
            branch = "levels" if dp <= 0.05 else "differenced"
            pc, pv = rule_lags(y, C, dose)
            for lt, (aa, bb) in [("rule", (pc, pv)), ("p2", (2, 2))]:
                est = estimate(y, C, dose, aa, bb, a.B, branch=branch)
                R += rows(f"{cs}_{lab}_{lt}", est, controls=cs, sample=lab, lags=lt, p_cvar=aa, p_var=bb,
                          T=len(Ds), design_p=dp, branch=branch)
            print(f"  {cs} {lab}: T {len(Ds)}, design p {dp:.3f} ({branch}), rule lags {pc}/{pv}, "
                  f"h=0 {est['cvar']['pt'][0]:.2f}", flush=True)
        # excluding 2020-2022: a dummy for every month of 2020-2022 in the VAR
        y, C, dose, _, idx = series(D, SETS[cs], MAIN)
        ex = ((idx >= "2020-01-01") & (idx <= "2022-12-01"))
        ve = np.column_stack([(idx == t).astype(float) for t in idx[ex]])
        est = estimate(y, C, dose, 2, 2, a.B, ve=ve, asy=False)
        R += rows(f"{cs}_dummies2020_22_p2", est, controls=cs, sample="full, 2020-22 dummied", lags="p2", p_cvar=2, p_var=2)
    pd.DataFrame(R).to_csv(os.path.join(OUT, "D_samples.csv"), index=False)

# ------------------------------------------------------------------ E. leave one control out
if "E" in a.sections:
    print("\nE. PADDs 1-4 leaving one out", flush=True)
    R = []
    for drop in SETS["padd1_4"]:
        cc = [c for c in SETS["padd1_4"] if c != drop]
        y, C, dose, _, idx = series(D, cc, MAIN)
        pc, pv = rule_lags(y, C, dose)
        est = estimate(y, C, dose, pc, pv, a.B)
        R += rows(f"without_{drop}", est, dropped=drop, p_cvar=pc, p_var=pv)
        print(f"  without {drop}: h=0 {est['cvar']['pt'][0]:.2f}, h=12 {est['cvar']['pt'][12]:.2f}", flush=True)
    pd.DataFrame(R).to_csv(os.path.join(OUT, "E_controls.csv"), index=False)

# ------------------------------------------------------------------ F. local projections, nonlinearity
if "F" in a.sections:
    print("\nF. Local projections", flush=True)
    import statsmodels.api as sm
    R = []
    for cs in ["gulf", "padd1_4"]:
        y, C, dose, lev, idx = series(D, SETS[cs], MAIN)
        g = y - C.mean(1)
        cr = D.credit.loc[idx].values
        cred_hi = pd.Series(cr > np.median(cr)).shift(1).fillna(False).values.astype(float)
        regs = {"linear": [dose],
                "increase_decrease": [np.maximum(dose, 0), np.minimum(dose, 0)],
                "high_low_credit": [dose * cred_hi, dose * (1 - cred_hi)]}
        for outcome, s in [("cvar_cycle", g), ("no_controls", y)]:
            ds = np.r_[np.nan, np.diff(s)]
            for p in [2, 6]:
                lagX = np.column_stack([pd.Series(ds).shift(j).values for j in range(1, p + 1)] +
                                       [pd.Series(dose).shift(j).values for j in range(1, p + 1)])
                for kind, X0 in regs.items():
                    for h in HOR:
                        lhs = pd.Series(s).shift(-h).values - pd.Series(s).shift(1).values
                        X = np.column_stack(X0 + [lagX])
                        ok = ~np.isnan(lhs) & ~np.isnan(X).any(1)
                        m = sm.OLS(lhs[ok], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
                        for j, nm in enumerate(["coef1", "coef2"][:len(X0)]):
                            b, se = m.params[1 + j], m.bse[1 + j]
                            R.append(dict(controls=cs, outcome=outcome, p=p, kind=kind, term=nm, h=h, coef=b,
                                          lo=b - 1.645 * se, hi=b + 1.645 * se, n=int(ok.sum())))
        print(f"  {cs} done", flush=True)
    F_ = pd.DataFrame(R)
    F_.to_csv(os.path.join(OUT, "F_lp.csv"), index=False)
    print("  terms: linear coef1 = cost change; increase_decrease coef1 = increases, coef2 = decreases; "
          "high_low_credit coef1 = high credit price, coef2 = low")

print("\nwritten to", OUT)
