#!/usr/bin/env python3
"""
Critical values of the point-optimal GLS test for the Control VAR (CVAR).

Statistic (rejects the null for SMALL values):
    P_T = [ S(alpha_bar) - alpha_bar * S(1) ] / omega2_hat,   alpha_bar = 1 + cbar/T
where S(alpha) is the sum of squared residuals of the quasi-differenced regression
    y_t - alpha*y_{t-1}  on  d_t - alpha*d_{t-1},  x_t - alpha*x_{t-1},  w_t - alpha*w_{t-1}
with d_t the deterministic terms (constant, or constant and trend), x_t the r trend
directions of the controls and w_t the stationary policy terms (current value and
12 lags of the policy), and omega2_hat a Bartlett long-run variance of the residual of
the regression in first differences (alpha = 1).
cbar = -7 with a constant, cbar = -13.5 with a constant and trend (as in ERS 1996).

Null hypothesis: y_t has a unit root of its own (the treated unit does not share the
controls' trends; the CVAR is not valid). Rejection supports the CVAR design.

Simulated null: y_t a random walk independent of x_t (r independent random walks)
and of the policy. The policy is either absent, or an AR(1) with coefficient 0.5
that reacts to the innovations of the trends (as in the CVAR, where the policy reacts
to common shocks), entered with its current value and 12 lags.

Output (in --out, default po_cv):
  critical_values.csv : quantiles (1%, 2.5%, 5%, 10%) by deterministic terms, r,
                        policy terms, and T (T = 2000 approximates the limit)
  critical_values.tex : the 5% table for the paper
Usage:
  python po_critical_values.py --quick
  python po_critical_values.py            # 20,000 replications per cell
"""
import argparse, os, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter

Q = [0.01, 0.025, 0.05, 0.10]
TS = [100, 200, 400, 2000]
RS = list(range(0, 7))
NLAG = 12

def lagmat(x, lags):
    T = x.shape[0]; x = x.reshape(T, -1); out = []
    for j in lags:
        z = np.full(x.shape, np.nan)
        z[j:] = x[:T - j] if j > 0 else x
        out.append(z)
    return np.hstack(out)

def ssr_qd(y, X, alpha):
    yd = np.r_[y[0], y[1:] - alpha * y[:-1]]
    Xd = np.vstack([X[0], X[1:] - alpha * X[:-1]])
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]; e = yd - Xd @ b
    return e @ e

def lrv(e):
    T = len(e); l = int(np.floor(4 * (T / 100) ** (2 / 9)))
    e = e - e.mean(); s = e @ e / T
    for j in range(1, l + 1):
        s += 2 * (1 - j / (l + 1)) * (e[j:] @ e[:-j]) / T
    return s

def po_stat(y, X, cbar):
    T = len(y); a = 1 + cbar / T
    num = ssr_qd(y, X, a) - a * ssr_qd(y, X, 1.0)
    # long-run variance from the regression in first differences (alpha = 1)
    dy = np.diff(y); dX = np.diff(X, axis=0)
    keep = np.abs(dX).sum(0) > 1e-12          # drop the constant, which differences to zero
    Z = np.column_stack([np.ones(len(dy)), dX[:, keep]])
    e = dy - Z @ np.linalg.lstsq(Z, dy, rcond=None)[0]
    return num / lrv(e)

def one(args):
    T, r, det, policy, seed = args
    rng = np.random.default_rng(seed)
    n = T + NLAG + 50
    y = np.cumsum(rng.standard_normal(n))
    eta = rng.standard_normal((n, r)) if r > 0 else np.zeros((n, 0))
    x = np.cumsum(eta, axis=0)
    cols = [np.ones(n)]
    if det == "trend": cols.append(np.arange(n, dtype=float))
    if r > 0: cols.append(x)
    if policy:
        shock = (0.5 * eta.sum(1) / np.sqrt(max(r, 1)) if r > 0 else 0) + rng.standard_normal(n)
        w = lfilter([1.0], [1.0, -0.5], shock)
        cols.append(lagmat(w, range(0, NLAG + 1)))
    X = np.column_stack(cols)
    sl = slice(n - T, n)
    cbar = -7.0 if det == "const" else -13.5
    return po_stat(y[sl], X[sl], cbar)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="po_cv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.reps = 500
    os.makedirs(a.out, exist_ok=True); rows = []; t0 = time.time()
    cells = list(itertools.product(["const", "trend"], RS, [False, True], TS))
    with Pool(a.workers) as pool:
        for ci, (det, r, policy, T) in enumerate(cells):
            args = [(T, r, det, policy, 1_000_003 * (ci + 1) + i) for i in range(a.reps)]   # reproducible seeds
            st = np.array(pool.map(one, args, chunksize=max(1, a.reps // (8 * a.workers))))
            row = dict(det=det, r=r, policy=policy, T=T)
            row.update({f"q{int(q*1000)/10:g}": round(float(np.quantile(st, q)), 3) for q in Q})
            rows.append(row)
            print(f"{det:5s} r={r} policy={policy!s:5s} T={T:4d}  5% cv = {row['q5']:.3f}   ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "critical_values.csv"), index=False)
    # LaTeX table of 5% critical values, T = 200 and the limit (T = 2000), with policy terms
    lines = [r"\begin{tabular}{c cc cc}", r"\toprule",
             r" & \multicolumn{2}{c}{Constant} & \multicolumn{2}{c}{Constant and trend}\\",
             r"$r$ & $T=200$ & $T\to\infty$ & $T=200$ & $T\to\infty$\\", r"\midrule"]
    for r in RS:
        v = lambda det, T: df[(df.det == det) & (df.r == r) & (df.policy) & (df["T"] == T)]["q5"].iloc[0]
        lines.append(f"{r} & {v('const',200):.2f} & {v('const',2000):.2f} & {v('trend',200):.2f} & {v('trend',2000):.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "critical_values.tex"), "w").write("\n".join(lines))
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
