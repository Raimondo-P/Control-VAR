r"""st_screen.py: where is a CVAR in levels possible at all? A screen of the outcomes, with no policy.

For each of the 48 contiguous states and the District of Columbia, the test of the design (Section 5 of
the paper) on the employment rate (LAUS, seasonally adjusted) against its bordering states:
  alone      the state without controls (is its employment rate I(1)? a non-rejection is needed)
  equal      the state minus the average of its neighbours
  estimated  weights on the neighbours estimated inside the test
No policy terms enter: the screen asks only whether the gap between a state and its neighbours is
stationary. A stationary policy adds a stationary component and does not change the null; the full
test with a policy's terms, run once under a pre-analysis plan, decides the branch for any candidate.
Samples: 1976-2019 and 1990-2019 (before the pandemic).
With 49 states, a test at 5 percent rejects in two or three states by chance alone: the candidates are
the states where the gap rejects in both samples while the state alone does not.
Usage: python st_screen.py (defaults: the cvar package in ../applications, the data in ../applications/data/states/raw) [--B 499]
Outputs: results/st_screen.csv, printed list of candidates
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIPS = {"AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
        "GA": "13", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
        "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32",
        "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41",
        "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51",
        "WA": "53", "WV": "54", "WI": "55", "WY": "56"}
NB = {"AL": "FL GA MS TN", "AZ": "CA CO NM NV UT", "AR": "LA MO MS OK TN TX", "CA": "AZ NV OR",
      "CO": "AZ KS NE NM OK UT WY", "CT": "MA NY RI", "DE": "MD NJ PA", "DC": "MD VA", "FL": "AL GA",
      "GA": "AL FL NC SC TN", "ID": "MT NV OR UT WA WY", "IL": "IN IA KY MO WI", "IN": "IL KY MI OH",
      "IA": "IL MN MO NE SD WI", "KS": "CO MO NE OK", "KY": "IL IN MO OH TN VA WV", "LA": "AR MS TX", "ME": "NH",
      "MD": "DE PA VA WV DC", "MA": "CT NH NY RI VT", "MI": "IN OH WI", "MN": "IA ND SD WI", "MS": "AL AR LA TN",
      "MO": "AR IL IA KS KY NE OK TN", "MT": "ID ND SD WY", "NE": "CO IA KS MO SD WY", "NV": "AZ CA ID OR UT",
      "NH": "ME MA VT", "NJ": "DE NY PA", "NM": "AZ CO OK TX UT", "NY": "CT MA NJ PA VT", "NC": "GA SC TN VA",
      "ND": "MN MT SD", "OH": "IN KY MI PA WV", "OK": "AR CO KS MO NM TX", "OR": "CA ID NV WA",
      "PA": "DE MD NJ NY OH WV", "RI": "CT MA", "SC": "GA NC", "SD": "IA MN MT NE ND WY",
      "TN": "AL AR GA KY MS MO NC VA", "TX": "AR LA NM OK", "UT": "AZ CO ID NV NM WY", "VT": "MA NH NY",
      "VA": "KY MD NC TN WV DC", "WA": "ID OR", "WV": "KY MD OH PA VA", "WI": "IL IA MI MN",
      "WY": "CO ID MT NE SD UT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=os.path.join(HERE, "..", "applications"), help="folder with the cvar package")
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "applications", "data", "states", "raw"), help="all-state BLS files and fred.csv")
    ap.add_argument("--B", type=int, default=499)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.pkg))
    import cvar  # noqa: E402
    os.makedirs(OUT, exist_ok=True)
    B = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(glob.glob(os.path.join(a.raw, "bls_*.csv")))], axis=1)
    B = B.T.assign(_n=B.notna().sum().values).sort_values("_n", ascending=False).drop(columns="_n")
    B = B[~B.index.duplicated(keep="first")].T.sort_index()
    ser = lambda st: B.get(f"LASST{FIPS[st]}0000000000007")
    rows = []
    for st, nbs in NB.items():
        cs = [c for c in nbs.split() if ser(c) is not None]
        if ser(st) is None or not cs:
            print(f"  {st}: series missing"); continue
        for samp, y0 in [("1976_2019", 1976), ("1990_2019", 1990)]:
            idx = pd.date_range(f"{y0}-01-01", "2019-12-01", freq="MS")
            D = pd.concat([ser(st).rename("y")] + [ser(c).rename(c) for c in cs], axis=1).reindex(idx).dropna()
            y, C = D["y"].values, D[cs].values
            z = np.zeros((len(y), 0))
            r = dict(state=st, sample=samp, T=len(D), neighbours=" ".join(cs))
            try:
                r["p_alone"] = cvar.design_test(y, z, None, q=0, B=a.B, seed=5).pvalue_bootstrap
                r["p_equal"] = cvar.design_test(y - C.mean(1), z, None, q=0, B=a.B, seed=5).pvalue_bootstrap
                r["p_estimated"] = cvar.design_test(y, C, None, q=0, B=a.B, seed=7).pvalue_bootstrap
            except Exception as e:  # noqa: BLE001
                r["error"] = str(e)[:100]
            rows.append(r)
            print(f"{st} {samp}: alone {r.get('p_alone', np.nan):.3f}  equal {r.get('p_equal', np.nan):.3f}  "
                  f"estimated {r.get('p_estimated', np.nan):.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "st_screen.csv"), index=False)
    W = R.pivot_table(index="state", columns="sample", values=["p_alone", "p_equal", "p_estimated"])
    ok_alone = (W["p_alone"] > 0.10).all(axis=1)
    cand = {w: W.index[ok_alone & (W[f"p_{w}"] <= 0.05).all(axis=1)].tolist() for w in ("equal", "estimated")}
    print("\nshare of states rejecting at 5 percent, by sample:")
    print(R.groupby("sample")[["p_alone", "p_equal", "p_estimated"]].apply(lambda d: (d <= 0.05).mean()).round(2).to_string())
    print(f"\ncandidates (alone does not reject at 10 percent in either sample; the gap rejects at 5 percent in both):")
    for w, l in cand.items():
        print(f"  {w} weights: {', '.join(l) if l else 'none'}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
