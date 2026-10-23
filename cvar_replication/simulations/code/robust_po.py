#!/usr/bin/env python3
"""
Size control of the point-optimal GLS test for the CVAR under serial correlation of the cycle.
Needs, in the same folder: mc_power.py, mc_bootstrap_cvar.py, mc_envelope.py, po_critical_values.py.

All tests have the null of a unit root in the treated unit (CVAR invalid) and reject for small values.
cbar = -7 throughout, the value of the paper (earlier versions used -12).

Tests
  PO_BART   original statistic: Bartlett long-run variance from the regression in differences
            (critical value simulated from the canonical null with the same regressors).
  PO_MAIC   same numerator; long-run variance from the ADF regression on the GLS-detrended residual,
            with the lag chosen by the modified AIC of Ng and Perron (2001); same critical value logic.
  PO_SIEVE  PO_MAIC with a sieve bootstrap p-value (Park 2003): the null model in differences,
            dy_t = dX_t g + e_t with e_t an AR(p) chosen by BIC, is resampled and cumulated, with the
            regressors held fixed. B bootstrap draws per replication.
  DFGLS_sa, ADF_sa  residual-based tests, size-adjusted within each design (for comparison of power).

Designs: canonical (iid cycle innovations) and the CVAR design of mc_bootstrap_cvar.py (persistent
idiosyncratic components, two policies, feedback), with 3 and 6 controls, T = 200 and 400,
c in {0, -5, -10, -20}.

Results are appended to OUT/robust.csv after each design, so an interrupted run keeps what finished.
Usage: python robust_po.py                 (500 replications, 199 bootstrap draws)
       python robust_po.py --reps 1000 --boot 399
"""
import argparse, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
import warnings; warnings.filterwarnings("ignore")
import mc_power as P
import po_critical_values as PC
import mc_envelope as E

CBAR = -7.0   # the value of the paper (Theorem po, ERS 1996); earlier versions used -12
CS = [0.0, -5.0, -10.0, -20.0]

def gls_resid(y, X, alpha):
    yd = np.r_[y[0], y[1:] - alpha * y[:-1]]
    Xd = np.vstack([X[0], X[1:] - alpha * X[:-1]])
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    return y - X @ b

def maic_lrv(u, pmax=12):
    """Ng-Perron (2001): ADF regression on the GLS-detrended series without deterministic terms,
    lag by the modified AIC, long-run variance s2 / (1 - sum of lag coefficients)^2."""
    du = np.diff(u); n = len(du) - pmax; best = None; lrv = {}
    for p in range(0, pmax + 1):
        Y = du[pmax:]
        Z = np.column_stack([u[pmax:-1]] + [du[pmax - j:len(du) - j] for j in range(1, p + 1)])
        b = np.linalg.lstsq(Z, Y, rcond=None)[0]; e = Y - Z @ b; s2 = e @ e / n
        tau = b[0] ** 2 * (u[pmax:-1] @ u[pmax:-1]) / s2
        crit = np.log(s2) + 2 * (tau + p) / n
        lrv[p] = s2 / (1 - b[1:].sum()) ** 2 if p > 0 else s2
        if best is None or crit < best[0]: best = (crit, p)
    return lrv[best[1]]

def po_numerator(y, X, cbar):
    T = len(y); a = 1 + cbar / T
    return PC.ssr_qd(y, X, a) - a * PC.ssr_qd(y, X, 1.0)

def po_maic(y, X, cbar=CBAR):
    T = len(y)
    return po_numerator(y, X, cbar) / maic_lrv(gls_resid(y, X, 1 + cbar / T))

def sieve_pvalue(y, X, B, rng, cbar=CBAR, pmax=12):
    stat = po_maic(y, X, cbar)
    dy = np.diff(y); dX = np.diff(X, axis=0); keep = np.abs(dX).sum(0) > 1e-12
    Z = np.column_stack([np.ones(len(dy)), dX[:, keep]])
    g = np.linalg.lstsq(Z, dy, rcond=None)[0]; fit = Z @ g; e = dy - fit
    best = None
    for p in range(0, pmax + 1):
        Y = e[pmax:]
        if p == 0:
            phi = np.zeros(0); r = Y
        else:
            ZZ = np.column_stack([e[pmax - j:len(e) - j] for j in range(1, p + 1)])
            phi = np.linalg.lstsq(ZZ, Y, rcond=None)[0]; r = Y - ZZ @ phi
        bic = np.log(r @ r / len(Y)) + p * np.log(len(Y)) / len(Y)
        if best is None or bic < best[0]: best = (bic, phi, r)
    _, phi, res = best; res = res - res.mean()
    count = 0
    for _ in range(B):
        eps = res[rng.integers(0, len(res), len(e) + 100)]
        es = lfilter([1.0], np.r_[1.0, -phi], eps)[100:]
        ys = np.r_[y[0], y[0] + np.cumsum(fit + es)]
        count += po_maic(ys, X, cbar) <= stat
    return (count + 1) / (B + 1)

def one(args):
    design, T, r, K, c, seed, B = args
    rng = np.random.default_rng(seed)
    if design == "cvar":
        y, X, u, m, dx = P.sim_cvar(T, K, r, c, rng)
    else:
        y, X, u, m, dx = P.sim_canonical(T, r, c, rng)
    uh = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(PO_BART=PC.po_stat(y, X, CBAR), PO_MAIC=po_maic(y, X),
                PO_SIEVE_p=sieve_pvalue(y, X, B, rng),
                DFGLS=E.adf_t(uh, gls_alpha=1 - 7.0 / T), ADF=E.adf_t(uh))

def null_cv(pool, T, m, reps):
    """5% critical values of PO_BART and PO_MAIC from the canonical null with m control regressors."""
    res = pool.map(one, [("canonical", T, m, m, 0.0, 3_000_000 + 1000 * m + i, 0) for i in range(reps)])
    return {k: np.quantile([o[k] for o in res], 0.05) for k in ["PO_BART", "PO_MAIC"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--boot", type=int, default=199)
    ap.add_argument("--cvreps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_robust")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, "robust.csv")
    if os.path.exists(path): os.remove(path)
    designs = [("canonical", 3, 3), ("cvar", 3, 3), ("cvar", 5, 6)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for T in [200, 400]:
            cvs = {m: null_cv(pool, T, m, a.cvreps) for m in sorted({K for _, _, K in designs})}
            for di, (design, r, K) in enumerate(designs):
                res = {}
                for ci, c in enumerate(CS):
                    args = [(design, T, r, K, c, 50_000_000 + 10_000_000 * di + 100_000 * ci + i, a.boot)
                            for i in range(a.reps)]
                    res[c] = pd.DataFrame(pool.map(one, args, chunksize=1))
                sa = {k: np.quantile(res[0.0][k], 0.05) for k in ["PO_BART", "PO_MAIC", "DFGLS", "ADF"]}
                rows = []
                for c in CS:
                    d = res[c]
                    rows.append(dict(T=T, design=design, r=r, K=K, c=c,
                        PO_BART=round(np.mean(d.PO_BART < cvs[K]["PO_BART"]), 3),
                        PO_MAIC=round(np.mean(d.PO_MAIC < cvs[K]["PO_MAIC"]), 3),
                        PO_SIEVE=round(np.mean(d.PO_SIEVE_p <= 0.05), 3),
                        PO_MAIC_sa=round(np.mean(d.PO_MAIC < sa["PO_MAIC"]), 3),
                        DFGLS_sa=round(np.mean(d.DFGLS < sa["DFGLS"]), 3),
                        ADF_sa=round(np.mean(d.ADF < sa["ADF"]), 3)))
                pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                print(f"T={T} {design} K={K} done ({time.time()-t0:.0f}s)", flush=True)
    print("Wrote", path)

if __name__ == "__main__":
    main()
