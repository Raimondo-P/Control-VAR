r"""st_screen_vars.py: is the stationary gap with the neighbours rare only for employment? The screen of
st_screen.py (no policy terms) for other state variables, and for consumer prices across regions and metros.

Variables (each state against its bordering states, equal and estimated weights; the state alone):
  epop      employment-population ratio (LAUS), monthly from 1976
  lfpr      participation rate = epop / (1 - unemployment rate / 100) (LAUS), monthly from 1976
  payroll   100 x log total nonfarm employment (CES, seasonally adjusted), monthly from 1990
  activity  100 x log Philadelphia Fed state coincident index (FRED {ST}PHCI), monthly from 1979
  income    100 x log total personal income (BEA via FRED {ST}OTOT), quarterly from 1948
  hpi       100 x log FHFA all-transactions house price index (FRED {ST}STHPI), quarterly from 1975
Consumer prices: 100 x log CPI-U (seasonally adjusted where FRED has it), each of the four Census regions
  against the other three, and New York, Chicago and Los Angeles against each other.
Samples end in December 2019 (fourth quarter for quarterly series). With 49 states, two or three
rejections at 5 percent are expected by chance.
Usage: python st_screen_vars.py (defaults: the cvar package in ../applications, the data in ../applications/data/states/raw) [--B 499]
Outputs: results/st_vars.csv (one row per unit, variable, test), results/st_vars_summary.csv
"""
import argparse
import glob
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_screen import FIPS, NB  # noqa: E402

OUT = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "data", "fred_states")
END = pd.Timestamp("2019-12-01")
CPI_REGIONS = {"Northeast": ["CUSR0100SA0", "CUUR0100SA0"], "Midwest": ["CUSR0200SA0", "CUUR0200SA0"],
               "South": ["CUSR0300SA0", "CUUR0300SA0"], "West": ["CUSR0400SA0", "CUUR0400SA0"]}
CPI_METROS = {"New York": ["CUURA101SA0"], "Chicago": ["CUURA207SA0"], "Los Angeles": ["CUURA421SA0"]}


def fred(sid):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{sid}.csv")
    if not os.path.exists(f):
        for t in range(3):
            try:
                r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                open(f, "w").write(r.text)
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        else:
            return None
    s = pd.read_csv(f, na_values=".")
    s = s.set_index(pd.to_datetime(s.iloc[:, 0])).iloc[:, 1]
    return pd.to_numeric(s, errors="coerce").dropna()


def first_fred(ids):
    for sid in ids:
        s = fred(sid)
        if s is not None and len(s) > 100:
            return s, sid
    return None, None


def bls_panel(raw):
    B = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(glob.glob(os.path.join(raw, "bls_*.csv")))], axis=1)
    B = B.T.assign(_n=B.notna().sum().values).sort_values("_n", ascending=False).drop(columns="_n")
    return B[~B.index.duplicated(keep="first")].T.sort_index()


def state_series(B):
    """{variable: (DataFrame by state, frequency, start)}"""
    get = lambda code: pd.DataFrame({st: B.get(code.format(f=f)) for st, f in FIPS.items()})
    epop = get("LASST{f}0000000000007")
    ur = get("LASST{f}0000000000003")
    out = {"epop": (epop, "M", 1976), "lfpr": (epop / (1 - ur / 100), "M", 1976),
           "payroll": (100 * np.log(get("SMS{f}000000000000001")), "M", 1990)}
    for v, pat, fq, y0 in [("activity", "{st}PHCI", "M", 1979), ("income", "{st}OTOT", "Q", 1948), ("hpi", "{st}STHPI", "Q", 1975)]:
        cols = {}
        for st in FIPS:
            s = fred(pat.format(st=st))
            if s is not None:
                cols[st] = 100 * np.log(s)
        out[v] = (pd.DataFrame(cols), fq, y0)
        print(f"{v}: {len(cols)} states from FRED", flush=True)
    return out


def tests(cvar, y, C, B, seeds=(5, 5, 7)):
    z = np.zeros((len(y), 0))
    return dict(p_alone=cvar.design_test(y, z, None, q=0, B=B, seed=seeds[0]).pvalue_bootstrap,
                p_equal=cvar.design_test(y - C.mean(1), z, None, q=0, B=B, seed=seeds[1]).pvalue_bootstrap,
                p_estimated=cvar.design_test(y, C, None, q=0, B=B, seed=seeds[2]).pvalue_bootstrap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=os.path.join(HERE, "..", "applications"), help="folder with the cvar package")
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "applications", "data", "states", "raw"), help="all-state BLS files and fred.csv")
    ap.add_argument("--B", type=int, default=499)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.pkg))
    import cvar  # noqa: E402
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for v, (X, fq, y0) in state_series(bls_panel(a.raw)).items():
        end = END if fq == "M" else pd.Timestamp("2019-10-01")
        X = X[(X.index >= pd.Timestamp(f"{y0}-01-01")) & (X.index <= end)]
        for st, nbs in NB.items():
            cs = [c for c in nbs.split() if c in X.columns]
            if st not in X.columns or not cs:
                continue
            D = X[[st] + cs].dropna()
            if len(D) < (120 if fq == "M" else 60):
                continue
            r = dict(variable=v, unit=st, frequency=fq, start=f"{D.index[0]:%Y-%m}", T=len(D), controls=" ".join(cs))
            try:
                r.update(tests(cvar, D[st].values, D[cs].values, a.B))
            except Exception as e:  # noqa: BLE001
                r["error"] = str(e)[:100]
            rows.append(r)
            print(f"{v:8s} {st}: alone {r.get('p_alone', np.nan):.3f}  equal {r.get('p_equal', np.nan):.3f}  "
                  f"estimated {r.get('p_estimated', np.nan):.3f}", flush=True)
    # consumer prices across regions and metros
    for group, ids in [("cpi_region", CPI_REGIONS), ("cpi_metro", CPI_METROS)]:
        S = {}
        for name, cand in ids.items():
            s, sid = first_fred(cand)
            if s is not None:
                S[name] = 100 * np.log(s); print(f"{group} {name}: {sid}", flush=True)
        X = pd.DataFrame(S)
        X = X[X.index <= END]
        for u in X.columns:
            cs = [c for c in X.columns if c != u]
            D = X[[u] + cs].dropna()
            r = dict(variable=group, unit=u, frequency="M", start=f"{D.index[0]:%Y-%m}", T=len(D), controls=", ".join(cs))
            try:
                r.update(tests(cvar, D[u].values, D[cs].values, a.B))
            except Exception as e:  # noqa: BLE001
                r["error"] = str(e)[:100]
            rows.append(r)
            print(f"{group} {u}: alone {r.get('p_alone', np.nan):.3f}  equal {r.get('p_equal', np.nan):.3f}  "
                  f"estimated {r.get('p_estimated', np.nan):.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "st_vars.csv"), index=False)
    summ = R.groupby("variable").agg(units=("unit", "size"), T=("T", "median"),
                                     alone_rejects=("p_alone", lambda p: (p <= 0.05).mean()),
                                     equal_rejects=("p_equal", lambda p: (p <= 0.05).mean()),
                                     estimated_rejects=("p_estimated", lambda p: (p <= 0.05).mean()))
    ok = R.p_alone > 0.10
    summ["gap_stationary_and_alone_I1"] = R[ok].groupby("variable").apply(
        lambda g: ((g.p_equal <= 0.05) | (g.p_estimated <= 0.05)).sum()).reindex(summ.index).fillna(0).astype(int)
    summ.to_csv(os.path.join(OUT, "st_vars_summary.csv"))
    print("\nshare of units rejecting at 5 percent (alone: the series itself looks stationary):")
    print(summ.round(2).to_string())
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
