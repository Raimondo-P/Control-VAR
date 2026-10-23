#!/usr/bin/env python3
"""
Power curves for the point-optimal GLS test of the CVAR design.
Needs, in the same folder: mc_bootstrap_cvar.py, mc_envelope.py, po_critical_values.py.

Null hypothesis of every test: the treated unit has a unit root of its own (CVAR invalid).
Alternatives: rho = 1 + c/T < 1 (stationary cycle, CVAR valid).

Designs
  canonical  y = 0.5 + beta'x + tau(L)W + u, u_t = rho u_{t-1} + eps_t, eps iid N(0,1),
             x = r random walks, W reacts to the trends' innovations (as in mc_envelope.py).
  corr       as canonical, but eps_t has correlation 0.5 with the innovation of the first
             trend (the case in which estimating the weights is costly).
  cvar       the CVAR data generating process of mc_bootstrap_cvar.py (persistent
             idiosyncratic components, two policies, feedback 0.3, controls possibly
             cointegrated among themselves), plus an extra component with coefficient rho.

Tests
  PO7   point-optimal GLS statistic, cbar = -7, critical value from the response surface
        3.32 + 14.1/T - 9.0 m/T (m = number of control regressors): ACTUAL size and power.
  PO12  same with cbar = -12, critical value simulated from the canonical null in this script.
  POX7, POX12  the point-optimal statistic with one lead and lag of the trends' changes as
        covariates and an AR spectral long-run variance (robust to correlation with the trends'
        innovations and to serial correlation); critical values simulated from the canonical null.
  *_sa  the same tests, size-adjusted within each design.
  DFGLS_sa, ADF_sa  DF-GLS and ADF on the residual of the CVAR weights regression,
        size-adjusted (no valid tables exist for them).
  ENV, ENV_INV      power envelopes with known weights and invariant to the weights
        (canonical and corr designs only). In the corr design ENV_INV ignores the information
        in the correlation and is not the relevant envelope (see the note).

Usage:
  python mc_power.py --quick
  python mc_power.py                 # 2,000 replications per point
Output (in --out, default mc_power): power.csv and power_curves.png
"""
import argparse, os, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
import warnings; warnings.filterwarnings("ignore")
import mc_bootstrap_cvar as M
import mc_envelope as E
import po_critical_values as PC

CS = [0, -2.5, -5, -7.5, -10, -15, -20, -30]
NLAG = 12

def surface_cv(T, m):
    return 3.32 + 14.1 / T - 9.0 * m / T

def lrv_ar(e, pmax=8):
    """AR spectral density at frequency zero, lag order by BIC (as in ERS 1996)."""
    e = e - e.mean(); n = len(e); best = None
    for p in range(0, pmax + 1):
        Y = e[pmax:]
        if p == 0:
            s2 = Y @ Y / len(Y); phi = np.zeros(0)
        else:
            Z = np.column_stack([e[pmax - j:n - j] for j in range(1, p + 1)])
            phi = np.linalg.lstsq(Z, Y, rcond=None)[0]; r = Y - Z @ phi; s2 = r @ r / len(Y)
        bic = np.log(s2) + p * np.log(len(Y)) / len(Y)
        if best is None or bic < best[0]: best = (bic, s2 / (1 - phi.sum()) ** 2)
    return best[1]

def pox_stat(y, X, dx, cbar):
    """Point-optimal GLS statistic with leads and lags of the trends' changes as covariates
    (corrects for correlation between the cycle and the trends' innovations) and an
    AR spectral estimator of the long-run variance (robust to serial correlation)."""
    T = len(y); a = 1 + cbar / T
    L = np.column_stack([E.lag(dx[:, k], j) for k in range(dx.shape[1]) for j in (-1, 0, 1)]) if dx.shape[1] else np.zeros((T, 0))
    XX = np.column_stack([X, L]); ok = ~np.isnan(XX).any(1)
    yy, XX = y[ok], XX[ok]
    num = PC.ssr_qd(yy, XX, a) - a * PC.ssr_qd(yy, XX, 1.0)
    dy = np.diff(yy); dX = np.diff(XX, axis=0); keep = np.abs(dX).sum(0) > 1e-12
    Z = np.column_stack([np.ones(len(dy)), dX[:, keep]]); e = dy - Z @ np.linalg.lstsq(Z, dy, rcond=None)[0]
    return num / lrv_ar(e)

def stats_from(y, X, u_true, T, dx=None):
    """All test statistics for one sample. X = [constant, controls, policy terms]."""
    out = {}
    out["PO7"] = PC.po_stat(y, X, -7.0)
    out["PO12"] = PC.po_stat(y, X, -12.0)
    out["POX7"] = pox_stat(y, X, dx, -7.0)
    out["POX12"] = pox_stat(y, X, dx, -12.0)
    # residual-based tests on the OLS (static + policy terms) weights regression residual
    uh = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    out["DFGLS"] = E.adf_t(uh, gls_alpha=1 - 7.0 / T)
    out["ADF"] = E.adf_t(uh)
    if u_true is not None:
        for c in CS[1:]:
            out[f"np_{c}"] = E.po_stat(u_true, c, 1.0)
            out[f"inv_{c}"] = E.po_inv(y, X, c, 1.0)
    return out

def sim_canonical(T, r, c, rng, corr=0.0):
    burn = 200; n = T + burn + NLAG
    eta = rng.standard_normal((n, r)); x = np.cumsum(eta, 0)
    w = lfilter([1.0], [1.0, -0.5], 0.5 * eta.sum(1) / np.sqrt(r) + rng.standard_normal(n))
    Eff = lfilter([0.2], [1.0, -0.8], w)
    z = rng.standard_normal(n)
    eps = corr * eta[:, 0] + np.sqrt(1 - corr ** 2) * z
    u = lfilter([1.0], [1.0, -(1 + c / T)], eps)
    y = 0.5 + x @ (np.ones(r) / r) + Eff + u
    X = np.column_stack([np.ones(n), x, PC.lagmat(w, range(0, NLAG + 1))])
    sl = slice(n - T, n)
    return y[sl], X[sl], u[sl] - u[n - T], r, np.diff(x, axis=0, prepend=x[:1])[sl]

def sim_cvar(T, K, r, c, rng):
    d = dict(K=K, r=r, feedback=0.3, c_eta=0.5, theta0=0.2, constrained=True, boundary=False, lagrule="12")
    s = M.simulate(d, T + NLAG, rng)
    u = lfilter([1.0], [1.0, -(1 + c / T)], rng.standard_normal(T + NLAG))
    y = s["Y1"] + u
    X = np.column_stack([np.ones(T + NLAG), s["Y0"], PC.lagmat(np.column_stack([s["R"], s["W"]]), range(0, NLAG + 1))])
    sl = slice(NLAG, T + NLAG)
    m0, D1, H = M.rank_step(s["Y0"])
    dx = np.diff(s["Y0"] @ D1, axis=0, prepend=(s["Y0"] @ D1)[:1])
    return y[sl], X[sl], None, K, dx[sl]

def one(args):
    design, T, r, K, c, seed = args
    rng = np.random.default_rng(seed)
    if design == "canonical": y, X, u, m, dx = sim_canonical(T, r, c, rng)
    elif design == "corr":    y, X, u, m, dx = sim_canonical(T, r, c, rng, corr=0.5)
    else:                     y, X, u, m, dx = sim_cvar(T, K, r, c, rng)
    out = stats_from(y, X, u, T, dx); out["m"] = m
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_power")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.reps = 100
    os.makedirs(a.out, exist_ok=True); rows = []; t0 = time.time()
    designs = [("canonical", 1, 1), ("canonical", 3, 3), ("corr", 1, 1), ("corr", 3, 3), ("cvar", 3, 3), ("cvar", 5, 6)]
    with Pool(a.workers) as pool:
        for T in [200, 400]:
            # critical value of PO12 from the canonical null with matching number of regressors
            cv12 = {}
            for m in sorted({K for _, _, K in designs}):
                null = pool.map(one, [("canonical", T, m, m, 0.0, 7_000_000 + 1000 * m + i) for i in range(max(a.reps, 2000))])
                cv12[m] = {k: np.quantile([o[k] for o in null], 0.05) for k in ["PO12", "POX7", "POX12"]}
            for di, (design, r, K) in enumerate(designs):
                res = {}
                for ci, c in enumerate(CS):
                    args = [(design, T, r, K, c, 10_000_000 * (di + 1) + 100_000 * ci + i) for i in range(a.reps)]
                    res[c] = pd.DataFrame(pool.map(one, args, chunksize=max(1, a.reps // (8 * a.workers))))
                m = int(res[0]["m"].iloc[0]); cv7 = surface_cv(T, m)
                sa = {k: np.quantile(res[0][k], 0.05) for k in ["PO7", "PO12", "POX7", "POX12", "DFGLS", "ADF"]}
                for c in CS:
                    d = res[c]
                    row = dict(T=T, design=design, r=r, K=K, c=c, rho=round(1 + c / T, 4),
                               PO7=round(np.mean(d.PO7 < cv7), 3), PO12=round(np.mean(d.PO12 < cv12[m]["PO12"]), 3),
                               POX7=round(np.mean(d.POX7 < cv12[m]["POX7"]), 3), POX12=round(np.mean(d.POX12 < cv12[m]["POX12"]), 3))
                    for k in ["PO7", "PO12", "POX7", "POX12", "DFGLS", "ADF"]:
                        row[k + "_sa"] = round(np.mean(d[k] < sa[k]), 3)
                    if design != "cvar" and c != 0:
                        row["ENV"] = round(np.mean(d[f"np_{c}"] < np.quantile(res[0][f"np_{c}"], 0.05)), 3)
                        row["ENV_INV"] = round(np.mean(d[f"inv_{c}"] < np.quantile(res[0][f"inv_{c}"], 0.05)), 3)
                    rows.append(row)
                print(f"T={T} {design} r={r} K={K} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "power.csv"), index=False)
    # plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(designs), figsize=(4 * len(designs), 7), sharey=True)
    for i, T in enumerate([200, 400]):
        for j, (design, r, K) in enumerate(designs):
            x = df[(df["T"] == T) & (df.design == design) & (df.r == r) & (df.K == K)]; cc = -x.c
            axx = ax[i, j]
            if "ENV_INV" in x and x.ENV_INV.notna().any():
                axx.plot(cc, x.ENV, "k-", lw=1, label="Envelope, weights known")
                axx.plot(cc, x.ENV_INV, "k--", lw=1, label="Envelope, invariant")
            axx.plot(cc, x.PO7, "b-o", ms=3, label="PO, cbar=-7")
            axx.plot(cc, x.PO12, "g-o", ms=3, label="PO, cbar=-12")
            axx.plot(cc, x.POX12, "c-s", ms=3, label="PO-X, cbar=-12")
            axx.plot(cc, x.DFGLS_sa, "r:", label="DF-GLS residual (size-adj.)")
            axx.plot(cc, x.ADF_sa, "m:", label="ADF residual (size-adj.)")
            axx.axhline(0.05, color="grey", lw=0.5)
            axx.set_title(f"{design}, r={r}, K={K}, T={T}", fontsize=10)
            if i == 1: axx.set_xlabel("-c")
    ax[0, 0].set_ylabel("Rejection rate"); ax[1, 0].set_ylabel("Rejection rate")
    ax[0, 0].legend(fontsize=7, loc="upper left"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "power_curves.png"), dpi=150)
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
