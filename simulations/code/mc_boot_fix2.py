#!/usr/bin/env python3
"""
Confirmation run for the lag rule of the sieve bootstrap, p_T = ceil(12 (T/200)^(1/3)) = 16 at T = 400,
with 20 lags for comparison; cbar = -7 (the ERS choice used in the paper); T = 400 only.
Based on mc_boot_fix.py (12 and 24 lags were run there).
Needs, in the same folder: mc_adaptive.py, robust_po.py (and the files they import: mc_competitors.py,
mc_bootstrap_cvar.py, po_critical_values.py, mc_power.py, mc_envelope.py).

Design: the CVAR design of mc_adaptive.sim_cvar (persistent idiosyncratic components, two policies with
feedback, controls K = 3 with r = 3 and K = 6 with r = 5), correlation delta in {0, 0.6}, T in {200, 400},
g in {0, 10} (size and power).

Variants of the sieve-bootstrap test (statistic: robust_po.po_maic with cbar = -7):
  order of the autoregression in the bootstrap p in {16, 20}.
Also reported, for reference: the test with the asymptotic ERS critical value (cbar = -7: 3.26).

Output in --out (default mc_bootfix2): results.csv, one row per (T, K, delta, g), saved after each cell.
Columns BOOT_c7_p16, BOOT_c7_p20 are rejection rates at 5%
(size when g = 0, power when g = 10); ASY_c7 uses the ERS critical value.
Usage: py mc_boot_fix2.py               (1000 replications, 199 bootstrap draws)
"""
import argparse, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
import warnings; warnings.filterwarnings("ignore")
import mc_adaptive as A
import robust_po as RP

CBARS = [-7.0]
PS = [16, 20]
ERS_CV5 = 3.26

def null_model(y, X):
    """Null regression in differences: dy on a constant and the changes of the controls and policy terms."""
    dy = np.diff(y); dX = np.diff(X, axis=0); keep = np.abs(dX).sum(0) > 1e-12
    Z = np.column_stack([np.ones(len(dy)), dX[:, keep]])
    g = np.linalg.lstsq(Z, dy, rcond=None)[0]; fit = Z @ g
    return fit, dy - fit

def ar_fit(e, p):
    Zp = np.column_stack([e[p - j:len(e) - j] for j in range(1, p + 1)])
    phi = np.linalg.lstsq(Zp, e[p:], rcond=None)[0]; res = e[p:] - Zp @ phi
    return phi, res - res.mean()

def one(args):
    T, K, r, g, delta, seed, B = args
    rng = np.random.default_rng(seed)
    y, x, W = A.sim_cvar(T, K, r, g, delta, rng)
    X = np.column_stack([np.ones(T), x, W])
    stats = {c: RP.po_maic(y, X, c) for c in CBARS}
    out = {"ASY_c7": float(stats[-7.0] < ERS_CV5)}
    fit, e = null_model(y, X)
    for p in PS:
        phi, res = ar_fit(e, p)
        cnt = {c: 0 for c in CBARS}
        for _ in range(B):
            es = lfilter([1.0], np.r_[1.0, -phi], res[rng.integers(0, len(res), len(e) + 200)])[200:]
            ys = np.r_[y[0], y[0] + np.cumsum(fit + es)]
            for c in CBARS:
                cnt[c] += RP.po_maic(ys, X, c) <= stats[c]
        for c in CBARS:
            out[f"BOOT_c{int(-c)}_p{p}"] = float((cnt[c] + 1) / (B + 1) <= 0.05)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=199)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_bootfix2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, "results.csv")
    if os.path.exists(path): os.remove(path)
    rows = []; t0 = time.time(); ci = 0
    with Pool(a.workers) as pool:
        for T in [400]:
            for K, r in [(3, 3), (6, 5)]:
                for delta in [0.0, 0.6]:
                    for g in [0.0, 10.0]:
                        args = [(T, K, r, g, delta, 3_500_000_000 + 10_000_000 * ci + i, a.boot) for i in range(a.reps)]
                        d = pd.DataFrame(pool.map(one, args, chunksize=1)); ci += 1
                        row = dict(T=T, K=K, r=r, delta=delta, g=g); row.update({c: round(d[c].mean(), 3) for c in d.columns})
                        rows.append(row); pd.DataFrame(rows).to_csv(path, index=False)
                        print(f"T={T} K={K} delta={delta} g={g} done ({time.time()-t0:.0f}s)", flush=True)
    print("Wrote", path)

if __name__ == "__main__":
    main()
