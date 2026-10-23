r"""mc_screen_power.py: size and power of the test of the design in panels built like the state screens.

DGP (monthly, burn-in 200):
  F_t = F_{t-1} + f_t, f_t = 0.5 f_{t-1} + N(0,1)              common stochastic trend
  Y_kt = F_t + v_kt, v_kt = 0.5 v_k,t-1 + 0.7 N(0,1)            K controls (the neighbours)
  Y_1t = F_t + g_t                                               the treated unit
    null (own trend):   g_t = g_{t-1} + s N(0,1)                 s = 0.25: an own trend weaker than the common one
    alternative:        g_t = rho g_{t-1} + N(0,1), rho in {0.95, 0.98, 0.99, 0.995}
  T in {180, 300, 528} (15, 25 and 44 years of monthly data); K = 4.
Bootstraps: the difference-based sieve of Proposition 6 (_boot) and the residual-based sieve (_resid), which
  takes the innovations from the unrestricted autoregression in levels (needs the updated designtest.py).
Tests, exactly as in st_screen.py (no policy terms, q = 0): alone (Y_1 without controls, always I(1) here:
  its rejection rate is the size of that test), equal weights (Y_1 minus the mean of the controls), estimated
  weights (the controls in the test regression). Each with the sieve-bootstrap p-value (B draws) and, for
  alone and equal weights, ERS's asymptotic 5 percent critical value 3.26 as an independent check.
Also reported: c = T (rho - 1), and the half-life of the gap in years. The ERS power envelope at c = -7 is
  one half by construction.
Usage: python mc_screen_power.py (default: the cvar package in ../applications) [--R 300] [--B 199] [--workers 6]
Output: results/mc_screen_power.csv and a printed table of rejection rates at 5 percent.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
TS = [180, 300, 528]
RHOS = ["null", 0.95, 0.98, 0.99, 0.995]
K = 4


def simulate(T, rho, rng, burn=200):
    n = T + burn
    f = np.zeros(n); v = np.zeros((n, K)); g = np.zeros(n)
    ef, ev, eg = rng.standard_normal(n), 0.7 * rng.standard_normal((n, K)), rng.standard_normal(n)
    for t in range(1, n):
        f[t] = 0.5 * f[t - 1] + ef[t]
        v[t] = 0.5 * v[t - 1] + ev[t]
        g[t] = g[t - 1] + 0.25 * eg[t] if rho == "null" else rho * g[t - 1] + eg[t]
    F = np.cumsum(f)
    Y0 = F[:, None] + v
    Y1 = F + g
    return Y1[burn:], Y0[burn:]


def one(args):
    pkg, T, rho, r, B = args
    sys.path.insert(0, pkg)
    import cvar
    rng = np.random.default_rng(1_000_003 * TS.index(T) + 10_007 * RHOS.index(rho) + r)
    y, C = simulate(T, rho, rng)
    z = np.zeros((T, 0))
    out = dict(T=T, rho=rho, rep=r)
    for spec, yy, CC in [("alone", y, z), ("equal", y - C.mean(1), z), ("estimated", y, C)]:
        t = cvar.design_test(yy, CC, None, q=0, B=B, seed=r)
        out[f"{spec}_boot"] = t.pvalue_bootstrap <= 0.05
        tr = cvar.design_test(yy, CC, None, q=0, B=B, seed=r, bootstrap="residual")
        out[f"{spec}_resid"] = tr.pvalue_bootstrap <= 0.05
        if spec != "estimated":
            out[f"{spec}_ers"] = t.statistic < 3.26
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=os.path.join(HERE, "..", "applications"), help="folder with the cvar package")
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--B", type=int, default=199)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pkg = os.path.abspath(a.pkg)
    if not os.path.isfile(os.path.join(pkg, "cvar", "designtest.py")):
        sys.exit(f"No cvar package in {pkg}: --pkg must be the folder that contains the cvar folder "
                 f"(on Windows, e.g. ..\\applications from the simulations folder)")
    jobs = [(pkg, T, rho, r, a.B) for T in TS for rho in RHOS for r in range(a.R)]
    print(f"{len(jobs)} replications, B = {a.B}, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=10))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "mc_screen_power.csv"), index=False)
    S = D.groupby(["T", "rho"], sort=False).mean(numeric_only=True).drop(columns="rep")
    S = S.reset_index()
    S["c"] = [np.nan if r == "null" else round(t * (r - 1), 1) for t, r in zip(S["T"], S["rho"])]
    S["half_life_years"] = [np.nan if r == "null" else round(np.log(0.5) / np.log(r) / 12, 1) for r in S["rho"]]
    cols = ["T", "rho", "c", "half_life_years", "alone_boot", "alone_resid", "alone_ers", "equal_boot", "equal_resid",
            "equal_ers", "estimated_boot", "estimated_resid"]
    print("\nrejection rates at 5 percent (null: own trend, so these are sizes; otherwise power):")
    print(S[cols].round(3).to_string(index=False))
    S[cols].to_csv(os.path.join(OUT, "mc_screen_power_summary.csv"), index=False)


if __name__ == "__main__":
    main()
