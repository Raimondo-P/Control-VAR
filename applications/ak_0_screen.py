"""ak_0_screen.py: screening of the Alaska Permanent Fund Dividend design before building the application.

Question: does Alaska share its stochastic trends with the control states the literature uses? The test of
the design (Section 5) is run for each outcome, control set and sample. Only designs where it rejects
support the levels branch with estimated weights.

Control sets (from the two papers):
  wa      Washington (Kueng 2018; the difference-in-differences check of Jones and Marinescu 2022)
  jm_emp  Utah, Wyoming, Washington, Nevada, Montana, Minnesota (synthetic Alaska for the employment rate,
          Jones and Marinescu 2022, Table A.9)
  jm_pt   Nevada, Wyoming (synthetic Alaska for the part-time rate)
Outcomes (monthly, seasonally adjusted):
  LAUS (from 1976): epop employment-population ratio (main outcome of Jones and Marinescu), ur unemployment
        rate, lfpr participation rate
  CES (from 1990): nonfarm, retail, leisure, 100 x log of employment (retail and leisure are nontradable)
Policies in the weights regression (current value and 12 lags):
  W    the dividend paid in the month, thousands of 2024 dollars per person (zero in other months), from
       data/alaska/pfd_amounts.csv (Alaska Department of Revenue, PFD Division)
  oil  100 x monthly change in log WTI (FRED WTISPLC), ordered first
Specifications of the test:
  estimated         estimated weights (DOLS, non-negative, Johansen rank step), no level shifts
  estimated_shifts  the same with automatic level shifts
  equal             equal weights (known-weights statistic on Alaska minus the mean of the controls)
  alaska_alone      Alaska with no controls: a unit-root test of the outcome itself
Samples: full (1976 or 1990 to the last month) and pre2016 (to December 2015, before the legislature set
the amounts).

Data: BLS public API (LAUS and CES; set the environment variable BLS_API_KEY for the faster v2 limits) and
FRED (WTI, CPI). Downloads are cached in data/alaska/raw; --refresh downloads again.
Outputs (results/alaska/): 0_screen.csv, 0_weights.csv
Usage: py ak_0_screen.py [--B 999] [--workers N] [--refresh] [--no_extra] [--outcomes epop ur]
"""
from common import *
import json
import time
import urllib.request
from multiprocessing import Pool

AKD = os.path.join(DATA, "alaska")
RAW = os.path.join(AKD, "raw")
FIPS = {"AK": "02", "WA": "53", "UT": "49", "WY": "56", "NV": "32", "MT": "30", "MN": "27"}
SETS_AK = {"wa": ["WA"], "jm_emp": ["UT", "WY", "WA", "NV", "MT", "MN"], "jm_pt": ["NV", "WY"]}
STATES = list(FIPS)
LAUS = {"ur": "03", "emp": "05", "epop": "07", "lfpr": "08", "pop": "09"}
CES = {"nonfarm": "00000000", "retail": "42000000", "leisure": "70000000"}
OUTCOMES = ["epop", "ur", "lfpr", "nonfarm", "retail", "leisure"]
SAMPLES = {"full": None, "pre2020": "2019-12-01", "pre2016": "2015-12-01"}
SPECS = ["estimated", "estimated_shifts", "equal"]
Q = 12


def laus_id(st, m):
    return f"LASST{FIPS[st]}00000000000{m}"


def ces_id(st, ind):
    return f"SMS{FIPS[st]}00000{ind}01"


# ------------------------------------------------------------------ downloads
def bls_fetch(ids, y0, y1):
    key = os.environ.get("BLS_API_KEY")
    per, span = (50, 20) if key else (25, 10)
    rows = []
    for i in range(0, len(ids), per):
        chunk = ids[i:i + per]
        for a in range(y0, y1 + 1, span):
            b = min(a + span - 1, y1)
            payload = {"seriesid": chunk, "startyear": str(a), "endyear": str(b)}
            if key:
                payload["registrationkey"] = key
            req = urllib.request.Request("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as f:
                res = json.loads(f.read().decode())
            if res.get("status") != "REQUEST_SUCCEEDED" or res.get("message"):
                print(f"  BLS {a}-{b}: {res.get('status')} {res.get('message')}", flush=True)
            for s in res.get("Results", {}).get("series", []):
                for d in s.get("data", []):
                    if d["period"].startswith("M") and d["period"] != "M13":
                        rows.append(dict(id=s["seriesID"], date=f"{d['year']}-{d['period'][1:]}-01",
                                         value=d["value"]))
            time.sleep(1.0)
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="id", values="value", aggfunc="last").sort_index()


def fred(sid):
    d = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", na_values=".")
    return pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values,
                     index=pd.to_datetime(d.iloc[:, 0]), name=sid)


def load_raw(refresh):
    os.makedirs(RAW, exist_ok=True)
    fb, ff = os.path.join(RAW, "bls.csv"), os.path.join(RAW, "fred.csv")
    if refresh or not os.path.exists(fb):
        ids = [laus_id(s, m) for s in STATES for m in LAUS.values()] + [ces_id(s, i) for s in STATES for i in CES.values()]
        print(f"downloading {len(ids)} BLS series", flush=True)
        bls_fetch(ids, 1976, pd.Timestamp.today().year).to_csv(fb)
    if refresh or not os.path.exists(ff):
        print("downloading FRED: WTISPLC, CPIAUCSL", flush=True)
        pd.concat([fred("WTISPLC"), fred("CPIAUCSL")], axis=1).to_csv(ff)
    B = pd.read_csv(fb, index_col=0, parse_dates=True)
    F = pd.read_csv(ff, index_col=0, parse_dates=True)
    return B, F


def build(B, F, use_extra=True):
    """Outcome panels (columns: states), the dividend series W and the oil change."""
    def col(i):
        return B[i] if i in B.columns else pd.Series(np.nan, index=B.index)
    out = {}
    for nm in ("epop", "ur", "lfpr"):
        out[nm] = pd.DataFrame({s: col(laus_id(s, LAUS[nm])) for s in STATES})
    implied = pd.DataFrame({s: 100 * col(laus_id(s, LAUS["emp"])) / col(laus_id(s, LAUS["pop"])) for s in STATES})
    out["epop"] = out["epop"].fillna(implied)
    gap = (out["epop"] - implied).abs().max().max()
    print(f"check: largest gap between the LAUS ratio and employment/population x 100: {gap:.3f}", flush=True)
    for nm, ind in CES.items():
        out[nm] = 100 * np.log(pd.DataFrame({s: col(ces_id(s, ind)) for s in STATES}))
    for nm, df in out.items():
        miss = [s for s in STATES if df[s].notna().sum() == 0]
        if miss:
            print(f"  {nm}: no data for {miss}", flush=True)
        out[nm] = fill_gaps(df, nm)
    F = F.resample("MS").mean()
    cpi = fill_gaps(F[["CPIAUCSL"]], "CPI")["CPIAUCSL"]
    base = cpi.loc["2024"].mean() if cpi.loc["2024"].notna().any() else cpi.dropna().iloc[-1]
    doil = 100 * np.log(F["WTISPLC"]).diff()
    P = pd.read_csv(os.path.join(AKD, "pfd_amounts.csv"), comment="#")
    W = pd.Series(0.0, index=F.index)
    lost = []
    for _, r in P.iterrows():
        t = pd.Timestamp(int(r.year), int(r.pay_month), 1)
        if t in W.index and not np.isnan(cpi.get(t, np.nan)):
            amt = r.dividend + (r.extra if use_extra else 0.0)
            W[t] = amt * base / cpi[t] / 1000.0
        else:
            lost.append(int(r.year))
    if lost:
        print(f"  WARNING: dividend not placed (month outside the data or CPI missing) for {lost}", flush=True)
    return out, W, doil


def fill_gaps(df, label, limit=2):
    """Linear interpolation of isolated missing months inside each series (at most `limit` in a row, never
    at the start or end), e.g. October 2025, which BLS did not publish because of the federal shutdown."""
    filled = df.interpolate(limit=limit, limit_area="inside")
    new = (df.isna() & filled.notna()).any(axis=1)
    if new.any():
        print(f"  {label}: interpolated {[str(t.date())[:7] for t in df.index[new]]}", flush=True)
    return filled


def contiguous(df):
    """Longest block of consecutive months with no missing value."""
    ok = np.r_[df.notna().all(axis=1).values, False]
    best, start = (0, 0), None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return df.iloc[best[0]:best[1]]


def arrays(out, W, doil, outcome, cc, end):
    Y = out[outcome]
    df = pd.concat([Y["AK"].rename("y")] + [Y[c].rename(c) for c in cc] + [W.rename("W"), doil.rename("doil")],
                   axis=1).loc[:end]
    span = df.dropna()
    blk = contiguous(df)
    if len(span) and len(blk) < len(span):
        print(f"  WARNING {outcome} {cc}: internal gap, kept {blk.index[0]:%Y-%m} to {blk.index[-1]:%Y-%m} "
              f"({len(blk)} of {len(span)} complete months)", flush=True)
    return blk


# ------------------------------------------------------------------ tests
def job(item):
    outcome, sample, cs, cc, spec, df, B = item
    base = dict(outcome=outcome, sample=sample, controls=cs, spec=spec, T=len(df),
                start=str(df.index[0].date())[:7] if len(df) else "", end=str(df.index[-1].date())[:7] if len(df) else "")
    try:
        y = df["y"].values
        C = df[cc].values if cc else np.zeros((len(df), 0))
        W, doil = df["W"].values, df["doil"].values
        pol = np.column_stack([doil, W])
        if spec in ("estimated", "estimated_shifts"):
            r = cvar.CVAR(q=Q, breaks="auto" if spec == "estimated_shifts" else None).fit(
                y, C, W, other=doil, names=["AK"], control_names=cc, dates=df.index)
            t = r.design_test(B=B, seed=7)[0]
            w = r.weights()
            sm = r.summary().iloc[0]
            base.update(stat=t.statistic, p=t.pvalue_bootstrap, sum_weights=float(w.weight.sum()),
                        weights=sm["weights"], rank=sm["coint_relations_among_controls"],
                        shifts=sm["level_shifts"])
            base["_w"] = w.assign(outcome=outcome, sample=sample, controls=cs, spec=spec)
        elif spec == "equal":
            g = y - C.mean(1)
            t = cvar.design_test(g, np.zeros((len(g), 0)), pol, q=Q, B=B, seed=5)
            base.update(stat=t.statistic, p=t.pvalue_bootstrap)
        else:                                              # Alaska alone
            t = cvar.design_test(y, np.zeros((len(y), 0)), pol, q=Q, B=B, seed=5)
            base.update(stat=t.statistic, p=t.pvalue_bootstrap)
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"
    return base


if __name__ == "__main__":
    a = args(refresh=dict(action="store_true"), no_extra=dict(action="store_true"),
             outcomes=dict(nargs="*", default=OUTCOMES))
    OUT = outdir("alaska")
    B_, F_ = load_raw(a.refresh)
    out, W, doil = build(B_, F_, use_extra=not a.no_extra)
    print(f"dividend months: {(W > 0).sum()}, from {W[W > 0].index.min():%Y-%m} to {W[W > 0].index.max():%Y-%m}", flush=True)

    items = []
    for o in a.outcomes:
        for smp, end in SAMPLES.items():
            df0 = arrays(out, W, doil, o, [], end)
            items.append((o, smp, "none", [], "alaska_alone", df0, a.B))
            for cs, cc in SETS_AK.items():
                df = arrays(out, W, doil, o, cc, end)
                if len(df) < 120:
                    print(f"  skipped {o} {smp} {cs}: {len(df)} months", flush=True)
                    continue
                for sp in SPECS:
                    items.append((o, smp, cs, cc, sp, df, a.B))
    print(f"{len(items)} tests, B = {a.B}", flush=True)
    with Pool(a.workers) as P:
        res = P.map(job, items)

    wts = [r.pop("_w") for r in res if "_w" in r]
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "0_screen.csv"), index=False)
    if wts:
        pd.concat(wts, ignore_index=True).to_csv(os.path.join(OUT, "0_weights.csv"), index=False)
    if "error" in R:
        for _, r in R[R["error"].notna()].iterrows():
            print(f"FAILED {r.outcome} {r['sample']} {r.controls} {r.spec}: {r.error}")
    ok = R[R["p"].notna()] if "p" in R else R
    print("\nBootstrap p-values of the test of the design (null: Alaska has a stochastic trend of its own)")
    print(ok.pivot_table(index=["outcome", "controls"], columns=["sample", "spec"], values="p").round(3).to_string())
    est = ok[ok.spec == "estimated"]
    if len(est):
        print("\nEstimated weights (no level shifts):")
        print(est[["outcome", "sample", "controls", "rank", "sum_weights", "weights"]].to_string(index=False))
    print("\nwritten to", OUT)
