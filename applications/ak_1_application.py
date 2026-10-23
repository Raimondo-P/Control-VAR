"""ak_1_application.py: the Alaska Permanent Fund Dividend, CVAR in changes with weights estimated in changes.

Design (fixed in advance, following Jones and Marinescu 2022 and Kueng 2018):
  outcomes   epop (employment-population ratio, main outcome of Jones and Marinescu), ur, lfpr (LAUS, from
             1976); nonfarm, retail, leisure (100 x log employment, CES, from 1990; retail and leisure are
             nontradable, where the spending response of Kueng should show)
  controls   jm_emp (UT, WY, WA, NV, MT, MN: donors of the synthetic Alaska for employment) is the main set;
             jm_pt (NV, WY) and wa (Washington, the control of Kueng) are robustness sets
  policies   oil (100 x change in log WTI) ordered first; W, the dividend paid in the month in thousands of
             2024 dollars per person (zero in other months). Responses are to a unit innovation in W, i.e.
             to $1,000 per person.
  weights    estimated for each outcome and control set from the regression in changes
               dY1_t = c + b' dY0_t + sum_{j=0..12} k_j' (oil, W)_{t-j} + e_t,   b >= 0,
             with HAC tests of sum(b) = 1 and b = 1/K; equal weights reported beside them
  VAR        (oil, W, change in the cycle), responses of the cycle cumulated; p = 12 (main) and the
             Ljung-Box rule of the other applications
  bands      90 and 68 percent, residual bootstrap with the same draws for every specification:
               equal     : equal weights
               est_fixed : estimated weights held fixed
               est_full  : weights re-estimated in each draw on a moving-block resample of the rows of the
                           weights regression, then a residual draw of the VAR on those weights (bands to quote)
               var       : VAR without controls (change in the Alaska outcome)
  objects    cycle (response of the outcome), ice (impulse causal effect, Theorem thm:direct: later dividends
             held fixed), policy (response of the dividend to its own innovation), and the response of the
             weighted controls in the VAR with both (Proposition prop:placebo; the controls never receive the
             dividend, so a response there is bias)
  samples    full; pre2020 (to December 2019, before the pandemic); pre2016 (to December 2015, before the
             legislature set the amounts); pre2008 (to December 2007, before the 2008 dividend, rebate and recession)
Requires ak_0_screen.py in the same folder and its cached data (data/alaska/raw, data/alaska/pfd_amounts.csv).
Outputs (results/alaska/): 1_responses.csv, 1_loading.csv, 1_weights.csv
Usage: py ak_1_application.py [--B 999] [--workers N] [--block 12] [--outcomes epop ur] [--sets jm_emp]
       [--samples full] [--pmodes p12]
"""
from common import *
from multiprocessing import Pool
from scipy.optimize import nnls
from scipy.stats import chi2, norm
import statsmodels.api as sm
from ak_0_screen import load_raw, build, arrays, SETS_AK

H = 36
HOR = (0, 1, 2, 3, 6, 12, 24, 36)
Q = 12
OUTCOMES_A = ["epop", "ur", "lfpr", "nonfarm", "retail", "leisure"]
SAMPLES_A = {"full": None, "pre2020": "2019-12-01", "pre2016": "2015-12-01", "pre2008": "2007-12-01"}
KEYS = ("cycle", "ice", "policy")


def col(x):
    return np.asarray(x, dtype=float).reshape(H + 1, -1)[:, 0]


def lags(M, js):
    M = np.asarray(M, dtype=float).reshape(len(M), -1)
    T = len(M)
    out = []
    for j in js:
        Z = np.full_like(M, np.nan)
        if j >= 0:
            Z[j:] = M[:T - j]
        else:
            Z[:T + j] = M[-j:]
        out.append(Z)
    return np.hstack(out)


def design(df, cc):
    dy = np.diff(df["y"].values)
    dX = np.diff(df[cc].values, axis=0)
    pol = np.column_stack([df["doil"].values, df["W"].values])[1:]
    F = np.hstack([np.ones((len(dy), 1)), lags(pol, range(0, Q + 1))])
    ok = ~(np.isnan(F).any(1) | np.isnan(dX).any(1))
    return dy[ok], dX[ok], F[ok]


def w_nnls(dy, X, F):
    P = np.linalg.pinv(F)
    b, _ = nnls(X - F @ (P @ X), dy - F @ (P @ dy))
    return b


def w_ols(dy, X, F):
    res = sm.OLS(dy, np.column_stack([X, F])).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    k = X.shape[1]
    return np.asarray(res.params)[:k], np.asarray(res.cov_params())[:k, :k], float(res.rsquared)


def loading_tests(b, V):
    K = len(b)
    one = np.ones(K)
    s, se = float(b.sum()), float(np.sqrt(one @ V @ one))
    d = b - 1.0 / K
    W = float(d @ np.linalg.pinv(V) @ d)
    return s, se, float(2 * norm.sf(abs((s - 1) / se))), float(chi2.sf(W, K))


def mbb(rng, M, ell, B):
    nb = int(np.ceil(M / ell))
    st = rng.integers(0, M - ell + 1, (B, nb))
    return (st[:, :, None] + np.arange(ell)[None, None, :]).reshape(B, -1)[:, :M]


def fit(df, cc, w, p):
    return cvar.CVAR(p=p).fit(df["y"].values, df[cc].values, df["W"].values, other=df["doil"].values,
                              names=["AK"], control_names=cc, weights=np.asarray(w, dtype=float), difference=True)


def band_rows(base, spec, obj, pt, draws):
    rows = []
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
    l6, h6 = np.quantile(draws, [0.16, 0.84], axis=0)
    for h in HOR:
        rows.append(dict(base, spec=spec, object=obj, h=h, estimate=pt[h], lo90=lo[h], hi90=hi[h], lo68=l6[h],
                         hi68=h6[h], zero_in_90=bool(lo[h] <= 0 <= hi[h]), zero_in_68=bool(l6[h] <= 0 <= h6[h])))
    return rows


def job(item):
    outcome, sample, cs, cc, df, pmode, B, block = item
    base = dict(outcome=outcome, sample=sample, controls=cs, T=len(df),
                start=f"{df.index[0]:%Y-%m}", end=f"{df.index[-1]:%Y-%m}")
    try:
        K = len(cc)
        dy, X, F = design(df, cc)
        w_hat = w_nnls(dy, X, F)
        b1, V1, r2 = w_ols(dy, X, F)
        t1 = loading_tests(b1, V1)
        rows_b = mbb(np.random.default_rng(22), len(dy), block, B)
        Wd = np.array([w_nnls(dy[r], X[r], F[r]) for r in rows_b])
        s_lo, s_hi = np.quantile(Wd.sum(1), [0.05, 0.95])
        eq = np.ones(K) / K
        p = 12 if pmode == "p12" else lag_rule(fit(df, cc, w_hat, 1).Z, None, [1, 2], 12)
        base.update(pmode=pmode, p=p)

        r_eq, r_w = fit(df, cc, eq, p), fit(df, cc, w_hat, p)
        n = r_w.U.shape[0]
        idx = np.random.default_rng(21).integers(0, n, (B, n))
        D = {s: {k: np.zeros((B, H + 1)) for k in KEYS} for s in ("equal", "est_fixed", "est_full")}
        Dv = np.zeros((B, H + 1))
        for b in range(B):
            for s, rr in (("equal", r_eq), ("est_fixed", r_w)):
                d = rr._draw(idx[b], H, no_controls=False)
                for k in KEYS:
                    D[s][k][b] = col(d[k])
            d = fit(df, cc, Wd[b], p)._draw(idx[b], H, no_controls=False)
            for k in KEYS:
                D["est_full"][k][b] = col(d[k])
            Dv[b] = col(r_w._nc_draw(idx[b], H))
        pe, pw = r_eq.irf(H), r_w.irf(H)
        P = {"equal": {k: col(pe[k]) for k in KEYS}, "est_fixed": {k: col(pw[k]) for k in KEYS}}
        P["est_full"] = P["est_fixed"]
        rows = []
        for s in D:
            for k in KEYS:
                rows += band_rows(base, s, k, P[s][k], D[s][k])
        rows += band_rows(base, "var", "cycle", col(r_w.irf_no_controls(H)), Dv)
        try:
            cr = r_w.controls_response(H, B=min(B, 499), seed=13)
            for k in ("treated", "controls", "difference"):
                rows += band_rows(base, "controls_response", k, col(cr.point[k]), cr.draws[k].reshape(cr.draws[k].shape[0], H + 1, -1)[:, :, 0])
        except Exception as e:
            print(f"  controls response skipped for {outcome} {sample} {cs}: {e}", flush=True)

        load = dict(base, K=K, r2=r2, sum_nnls=float(w_hat.sum()), sum_nnls_lo90=s_lo, sum_nnls_hi90=s_hi,
                    sum_ols=t1[0], se_ols=t1[1], p_sum1=t1[2], p_equal=t1[3],
                    n_positive=int((w_hat > 0.005).sum()), max_weight=float(w_hat.max()))
        wt = pd.DataFrame(dict(outcome=outcome, sample=sample, controls=cs, pmode=pmode, control=cc,
                               w_nnls=w_hat, w_ols=b1, w_ols_se=np.sqrt(np.diag(V1)), equal=eq))
        return dict(rows=rows, load=load, wt=wt)
    except Exception as e:
        return dict(error=f"{outcome} {sample} {cs} {pmode}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    a = args(block=dict(type=int, default=12), outcomes=dict(nargs="*", default=OUTCOMES_A),
             sets=dict(nargs="*", default=list(SETS_AK)), samples=dict(nargs="*", default=list(SAMPLES_A)),
             pmodes=dict(nargs="*", default=["p12", "rule"]))
    OUT = outdir("alaska")
    B_, F_ = load_raw(False)
    out, W, doil = build(B_, F_)
    items = []
    for o in a.outcomes:
        for smp in a.samples:
            for cs in a.sets:
                cc = SETS_AK[cs]
                df = arrays(out, W, doil, o, cc, SAMPLES_A[smp])
                for pm in a.pmodes:
                    items.append((o, smp, cs, cc, df, pm, a.B, a.block))
    print(f"{len(items)} specifications, B = {a.B}", flush=True)
    with Pool(a.workers) as P:
        res = P.map(job, items)
    for r in res:
        if "error" in r:
            print("FAILED", r["error"])
    res = [r for r in res if "error" not in r]
    R = pd.DataFrame([x for r in res for x in r["rows"]])
    L = pd.DataFrame([r["load"] for r in res])
    R.to_csv(os.path.join(OUT, "1_responses.csv"), index=False)
    L.to_csv(os.path.join(OUT, "1_loading.csv"), index=False)
    pd.concat([r["wt"] for r in res], ignore_index=True).to_csv(os.path.join(OUT, "1_weights.csv"), index=False)

    print("\nLoading of Alaska on the controls, regression in changes (p = 12 rows)")
    print(L[L.pmode == "p12"][["outcome", "sample", "controls", "K", "n_positive", "max_weight", "sum_nnls",
                               "sum_nnls_lo90", "sum_nnls_hi90", "p_sum1", "p_equal"]].round(3).to_string(index=False))

    def fmt(g, h):
        z = g[g.h == h]
        if not len(z):
            return ""
        z = z.iloc[0]
        star = "**" if not z.zero_in_90 else ("*" if not z.zero_in_68 else "")
        return f"{z.estimate:.3f}{star} [{z.lo90:.3f}, {z.hi90:.3f}]"

    main = R[(R.controls == "jm_emp") & (R.pmode == "p12")]
    for smp in a.samples:
        print(f"\nMain specification (six-state donors, p = 12, {smp}), response to $1,000 per person; "
              "** 90% band excludes zero, * 68% band excludes zero")
        for o in a.outcomes:
            g = main[(main.outcome == o) & (main["sample"] == smp)]
            if not len(g):
                continue
            print(f"  {o}")
            for lab, s, k in (("CVAR, estimated weights", "est_full", "cycle"), ("CVAR, equal weights", "equal", "cycle"),
                              ("impulse causal effect", "est_full", "ice"), ("VAR without controls", "var", "cycle"),
                              ("weighted controls", "controls_response", "controls")):
                gg = g[(g.spec == s) & (g.object == k)]
                print(f"    {lab:26s} " + "  ".join(f"h={h}: {fmt(gg, h)}" for h in (0, 1, 3, 12, 24)))
    print("\nwritten to", OUT)
