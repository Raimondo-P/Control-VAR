"""eb_4_states.py: every eligible EB state in the differenced branch. Cumulated response of the
unemployment rate (percentage points) to a ten-week EB innovation (EUC ordered first), CVAR against
the VAR without controls (both in changes), 90 percent percentile bands from the residual
bootstrap; each model with its rule lags ('rule') and both at the larger order ('common').
Requires eb_3_lags.py. Output: results/eb/4_states.csv
Usage: py eb_4_states.py [--B 999] [--workers N]"""
from common import *
from multiprocessing import Pool

UR, EB, EUC, elig = load_eb()
H = 36


def job(item):
    s, pc, pv, B = item
    rows = []
    for spec, a_, b_ in [("rule", pc, pv), ("common", max(pc, pv), max(pc, pv))]:
        bc = fit_eb_diff(UR, EB, EUC, s, a_).bootstrap(B, H, seed=11, no_controls=False)
        bv = fit_eb_diff(UR, EB, EUC, s, b_).bootstrap(B, H, seed=12, no_controls=True)
        lc, hc = bc.bands("cycle")
        lv, hv = bv.bands("no_controls")
        for h in (0, 6, 12, 24, 36):
            rows.append(dict(st=s, spec=spec, h=h, p_cvar=a_, p_var=b_,
                             cvar=bc.point["cycle"][h, 0], cvar_lo=lc[h, 0], cvar_hi=hc[h, 0],
                             ice=bc.point["ice"][h, 0],
                             var=bv.point["no_controls"][h, 0], var_lo=lv[h, 0], var_hi=hv[h, 0],
                             width_ratio=(hc[h, 0] - lc[h, 0]) / (hv[h, 0] - lv[h, 0])))
    return rows


if __name__ == "__main__":
    a = args()
    L = pd.read_csv(os.path.join(outdir("eb"), "3_lags.csv")).set_index("st")
    with Pool(a.workers) as P:
        res = P.map(job, [(s, int(L.p_cvar[s]), int(L.p_var[s]), a.B) for s in elig if s in L.index])
    o = pd.DataFrame([r for rr in res for r in rr])
    o.to_csv(os.path.join(outdir("eb"), "4_states.csv"), index=False)
    for spec in ("rule", "common"):
        x = o[o.spec == spec]
        print(f"\n{spec}: CVAR / VAR band-width ratio by horizon")
        print(x.groupby("h").width_ratio.describe()[["50%", "min", "max"]].round(2))
        print("share of states with narrower CVAR bands:", x.assign(n=x.width_ratio < 1).groupby("h").n.mean().round(2).to_dict())
