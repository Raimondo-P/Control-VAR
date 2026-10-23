"""ak_2_checks.py: two checks of the main Alaska specification (pre-2020, six-state donors, weights estimated
in changes, oil ordered first, 12 lags).

A. Sensitivity to the exogeneity assumption (Proposition prop:bounds). From the VAR with Alaska and the
   weighted controls (Proposition prop:placebo), bands for [gamma^G - lam |gamma^S|, gamma^G + lam |gamma^S|],
   where the bias specific to Alaska is at most lam times the common-trend bias measured by the controls.
   Reported for lam = 0, 0.25, 0.5, 1, with two break-even values:
     breakdown_lambda : smallest lam at which the band contains the response of Alaska in the VAR with the
                        controls added (the package's definition, as in the EB application)
     zero_lambda      : smallest lam at which the band contains zero, i.e. how large the Alaska-specific
                        bias must be, relative to the common-trend bias, to make the effect insignificant
   Weights are held at their estimates in these draws.
B. Seasonal adjustment. The main specification uses seasonally adjusted series, which remove the average
   October effect. This check re-estimates with (i) the adjusted series as in the main specification,
   (ii) the adjusted series plus month dummies, (iii) the unadjusted series (LAUS LAUST, CES SMU) plus month
   dummies, in the weights regression and in the VAR. If the negative response of leisure and hospitality on
   impact is an artifact of the adjustment in October, it should disappear in (iii).
   Bands: residual bootstrap with the weights held at their estimates (the same draws in every version).

Requires ak_0_screen.py and ak_1_application.py in the same folder and the cached data; the unadjusted series
are downloaded on the first run (data/alaska/raw/bls_nsa.csv).
Outputs (results/alaska/): 2_bounds.csv, 2_seasonal.csv
   (iv) and (v): the unadjusted series with a seasonal pattern that changes over time, month dummies plus month
   dummies times a linear trend, and month dummies by decade. Alaska's seasonal swings (fishing, tourism) changed
   a lot over the sample, which fixed month dummies cannot absorb.
Usage: py ak_2_checks.py [--B 999] [--workers N] [--refresh] [--outcomes leisure nonfarm] [--skip_bounds]
"""
from common import *
from multiprocessing import Pool
import ak_0_screen as s0
from ak_1_application import w_nnls, col, lags, Q

H = 36
HOR = (0, 1, 2, 3, 6, 12, 24, 36)
OUTC = ["epop", "ur", "lfpr", "nonfarm", "retail", "leisure"]
END = "2019-12-01"
CC = s0.SETS_AK["jm_emp"]
P = 12


def laus_nsa(st, m):
    return f"LAUST{s0.FIPS[st]}00000000000{m}"


def ces_nsa(st, ind):
    return f"SMU{s0.FIPS[st]}00000{ind}01"


def load_nsa(refresh):
    f = os.path.join(s0.RAW, "bls_nsa.csv")
    if refresh or not os.path.exists(f):
        ids = ([laus_nsa(s, s0.LAUS[k]) for s in s0.STATES for k in ("ur", "epop", "lfpr")]
               + [ces_nsa(s, i) for s in s0.STATES for i in s0.CES.values()])
        print(f"downloading {len(ids)} unadjusted BLS series", flush=True)
        s0.bls_fetch(ids, 1976, pd.Timestamp.today().year).to_csv(f)
    Bn = pd.read_csv(f, index_col=0, parse_dates=True)

    def colx(i):
        return Bn[i] if i in Bn.columns else pd.Series(np.nan, index=Bn.index)
    out = {}
    for nm in ("epop", "ur", "lfpr"):
        out[nm] = s0.fill_gaps(pd.DataFrame({s: colx(laus_nsa(s, s0.LAUS[nm])) for s in s0.STATES}), nm + " (NSA)")
    for nm, ind in s0.CES.items():
        out[nm] = 100 * np.log(pd.DataFrame({s: colx(ces_nsa(s, ind)) for s in s0.STATES}))
    for nm, df in out.items():
        miss = [s for s in s0.STATES if df[s].notna().sum() == 0]
        if miss:
            print(f"  {nm} (NSA): no data for {miss}", flush=True)
    return out


def month_dummies(idx):
    m = np.asarray(idx.month)
    return np.column_stack([(m == k).astype(float) for k in range(1, 12)])


def season_terms(idx, kind):
    """Deterministic seasonal terms for the weights regression and the VAR.
    fixed : 11 month dummies (constant seasonal pattern)
    trend : month dummies plus month dummies times a linear trend (seasonal pattern changing smoothly)
    decade: month dummies separately in each decade, plus decade dummies (seasonal pattern changing by decade)"""
    if kind is None:
        return None
    M = month_dummies(idx)
    if kind == "fixed":
        return M
    if kind == "trend":
        t = np.linspace(0.0, 1.0, len(idx))[:, None]
        return np.column_stack([M, M * t])
    if kind == "decade":
        dec = np.asarray(idx.year) // 10
        ds = sorted(set(dec))
        cols = [M * (dec == d)[:, None] for d in ds]
        cols += [(dec == d).astype(float)[:, None] for d in ds[1:]]
        return np.column_stack(cols)
    raise ValueError(kind)


def design_m(df, cc, M=None):
    """Weights regression in changes, as in ak_1_application.py, optionally with month dummies."""
    dy = np.diff(df["y"].values)
    dX = np.diff(df[cc].values, axis=0)
    pol = np.column_stack([df["doil"].values, df["W"].values])[1:]
    parts = [np.ones((len(dy), 1)), lags(pol, range(0, Q + 1))]
    if M is not None:
        parts.append(M[1:])
    F = np.hstack(parts)
    ok = ~(np.isnan(F).any(1) | np.isnan(dX).any(1))
    return dy[ok], dX[ok], F[ok]


def fit2(df, cc, w, p, M=None):
    return cvar.CVAR(p=p).fit(df["y"].values, df[cc].values, df["W"].values, other=df["doil"].values,
                              var_exog=M, names=["AK"], control_names=cc,
                              weights=np.asarray(w, dtype=float), difference=True)


def job_bounds(item):
    o, df, B = item
    try:
        dy, X, F = design_m(df, CC)
        r = fit2(df, CC, w_nnls(dy, X, F), P)
        cr = r.controls_response(H, B=B, seed=13)
        bd = cvar.bounds(cr, lams=(0.0, 0.25, 0.5, 1.0), horizons=(3, 6, 12, 24))
        G, S = cr.draws["difference"][:, :, 0], np.abs(cr.draws["controls"][:, :, 0])
        grid = np.round(np.arange(0, 3.001, 0.01), 2)
        zl = []
        for h in bd.h:
            g, s = G[:, int(h)], S[:, int(h)]
            hit = [l for l in grid if np.quantile(g - l * s, 0.05) <= 0 <= np.quantile(g + l * s, 0.95)]
            zl.append(hit[0] if hit else np.nan)
        bd["zero_lambda"] = zl
        bd["outcome"] = o
        return bd
    except Exception as e:
        return f"bounds {o}: {type(e).__name__}: {e}"


def job_seas(item):
    o, version, df, kind, B = item
    try:
        M = season_terms(df.index, kind)
        dy, X, F = design_m(df, CC, M)
        w = w_nnls(dy, X, F)
        r = fit2(df, CC, w, P, M)
        n = r.U.shape[0]
        idx = np.random.default_rng(21).integers(0, n, (B, n))
        D = np.array([col(r._draw(ix, H, no_controls=False)["cycle"]) for ix in idx])
        Dv = np.array([col(r._nc_draw(ix, H)) for ix in idx])
        rows = []
        for spec, pt, dr in (("cvar", col(r.irf(H)["cycle"]), D), ("var", col(r.irf_no_controls(H)), Dv)):
            lo, hi = np.quantile(dr, [0.05, 0.95], axis=0)
            for h in HOR:
                rows.append(dict(outcome=o, version=version, spec=spec, h=h, estimate=pt[h], lo90=lo[h], hi90=hi[h],
                                 zero_in_90=bool(lo[h] <= 0 <= hi[h]), sum_weights=float(w.sum()),
                                 weights=", ".join(f"{c} {v:.2f}" for c, v in sorted(zip(CC, w), key=lambda z: -z[1]) if v > 0.005),
                                 start=f"{df.index[0]:%Y-%m}", end=f"{df.index[-1]:%Y-%m}"))
        return rows
    except Exception as e:
        return f"seasonal {o} {version}: {type(e).__name__}: {e}"


if __name__ == "__main__":
    a = args(refresh=dict(action="store_true"), outcomes=dict(nargs="*", default=OUTC),
             skip_bounds=dict(action="store_true"))
    OUT = outdir("alaska")
    B_, F_ = s0.load_raw(False)
    out, W, doil = s0.build(B_, F_)
    out_n = load_nsa(a.refresh)

    items_b = [(o, s0.arrays(out, W, doil, o, CC, END), a.B) for o in a.outcomes]
    items_s = []
    for o in a.outcomes:
        dsa = s0.arrays(out, W, doil, o, CC, END)
        dn = s0.arrays(out_n, W, doil, o, CC, END)
        items_s += [(o, "SA", dsa, None, a.B), (o, "SA + months", dsa, "fixed", a.B),
                    (o, "NSA + months", dn, "fixed", a.B), (o, "NSA + months x trend", dn, "trend", a.B),
                    (o, "NSA + months x decade", dn, "decade", a.B)]
    print(f"{len(items_b)} bounds and {len(items_s)} seasonal specifications, B = {a.B}", flush=True)
    if a.skip_bounds:
        items_b = []
    with Pool(a.workers) as Pp:
        rb = Pp.map(job_bounds, items_b)
        rs = Pp.map(job_seas, items_s)
    for r in rb + rs:
        if isinstance(r, str):
            print("FAILED", r)

    okb = [r for r in rb if not isinstance(r, str)]
    BD = pd.concat(okb, ignore_index=True) if okb else pd.DataFrame()
    if len(BD):
        BD.to_csv(os.path.join(OUT, "2_bounds.csv"), index=False)
    SS = pd.DataFrame([x for r in rs if not isinstance(r, str) for x in r])
    SS.to_csv(os.path.join(OUT, "2_seasonal.csv"), index=False)

    print("\nA. Sensitivity bounds (pre-2020, six-state donors, p = 12); 90% bands")
    for _, r in BD.iterrows():
        txt = "  ".join(f"lam {l:g}: [{r[f'lam{l:g}_lower']:.3f}, {r[f'lam{l:g}_upper']:.3f}]" for l in (0.0, 0.25, 0.5, 1.0))
        print(f"  {r.outcome:8s} h={int(r.h):2d}  CVAR {r.difference:7.3f}  controls {r.controls:7.3f}  "
              f"VAR with controls {r.treated_response:7.3f}  {txt}  zero at lam {r.zero_lambda}  VAR at lam {r.breakdown_lambda}")

    print("\nB. Seasonal adjustment: CVAR response (90% band); weights held at their estimates")
    for o in a.outcomes:
        print(f"  {o}")
        for v in ("SA", "SA + months", "NSA + months", "NSA + months x trend", "NSA + months x decade"):
            g = SS[(SS.outcome == o) & (SS.version == v) & (SS.spec == "cvar")]
            if not len(g):
                continue
            txt = "  ".join(f"h={int(z.h)}: {z.estimate:.3f} [{z.lo90:.3f}, {z.hi90:.3f}]"
                            for _, z in g[g.h.isin([0, 1, 3, 12, 24])].iterrows())
            print(f"    {v:22s} sum w {g.sum_weights.iloc[0]:.2f}  {txt}")
    print("\nwritten to", OUT)
