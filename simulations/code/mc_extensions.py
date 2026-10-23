#!/usr/bin/env python3
"""
Monte Carlo for the extensions of the Control VAR paper (appendix simulations).
Needs mc_bootstrap_cvar.py in the same folder (it reuses its estimator functions).

Experiments (choose with --exp, default all):
  spill   Spillovers to the controls with a known exposure mapping.
          K controls share one common trend with the treated unit; each control
          receives a share pi*x_k of the effect on the treated unit, with
          x_k = (d_1/d_k)^2 from distances d_k (quadratic distance decay).
          Estimators: synthetic CVAR ignoring spillovers; the response of the
          cycle with the nearest and with the most distant control; exposure
          estimator (intercept of the regression, horizon by horizon, of the
          responses g_k(h) of the single-control cycles, from one VAR in
          (W, G_1, ..., G_K), on b_k*x_k).
  nonlin  Effect non-linear in the policy innovation, f(u) = u + kappa*u^2.
          Estimators: linear CVAR response scaled to a surprise of size delta;
          dictionary local projection of the cycle on (u, u^2) and 12 lags.
  stat    Stationarity diagnostics for the estimated cycle (weights estimated
          by constrained dynamic OLS with policy terms). Under the null the
          counterfactual cycle is a stationary AR(rho), rho in {0.5, 0.9};
          under the alternative it has a unit root. Statistics: ADF and KPSS on
          the cycle and on the cycle net of the policy terms. Reports size at
          the standard critical values (Engle-Granger for ADF, 0.463 for KPSS),
          simulated 5% critical values (ADF from rho = 1, KPSS from rho = 0.9),
          and rejection rates at both.

Usage:
  python mc_extensions.py --quick                  # smoke test
  python mc_extensions.py --exp all --workers 8    # full run (about an hour)
Outputs in --out (default mc_ext): spill.csv, nonlin.csv, stat.csv, meta.json.
Send me the whole folder.
"""
import argparse, os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import nnls
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.adfvalues import mackinnoncrit
import warnings; warnings.filterwarnings("ignore")
import mc_bootstrap_cvar as M

P = 12
HS = [0, 1, 3, 6, 12]

# ---------------------------------------------------------------- spillovers
def sim_spill(T, K, pi, theta0, rng, burn=300):
    n = T + burn
    lam = np.linspace(0.8, 1.2, K)
    d = np.linspace(1.0, 4.0, K)              # distances, control 1 closest
    x = (d[0] / d) ** 2                        # quadratic distance decay, x_1 = 1
    eta = rng.standard_normal(n); F = np.zeros(n); dF = np.zeros(n)
    v = np.zeros((n, K)); e = np.zeros(n); W = np.zeros(n); E = np.zeros(n)
    ev = rng.standard_normal((n, K)); ee = rng.standard_normal(n); eW = rng.standard_normal(n)
    for t in range(1, n):
        dF[t] = 0.3 * dF[t-1] + eta[t]; F[t] = F[t-1] + dF[t]
        v[t] = 0.5 * v[t-1] + ev[t]; e[t] = 0.5 * e[t-1] + ee[t]
        W[t] = 0.5 * W[t-1] + 0.5 * eta[t] + eW[t]
        E[t] = 0.8 * E[t-1] + theta0 * W[t]
    Y1 = F + e + E
    Y0 = np.outer(F, lam) + v + pi * np.outer(E, x)
    sl = slice(burn, n)
    return Y1[sl], Y0[sl], W[sl], x, lam

def theta_spill(theta0, H=12):
    tau = theta0 * 0.8 ** np.arange(H + 1); psi = 0.5 ** np.arange(H + 1)
    return np.array([sum(tau[j] * psi[h - j] for j in range(h + 1)) for h in range(H + 1)])

def cvar_resp(Y1, Y0, W, D1):
    a, b = M.dols(Y1, Y0, np.zeros(len(W)), W, D1, True)   # no second policy: zeros
    G = Y1 - a - Y0 @ b
    Z = np.column_stack([W, G])
    A, U = M.var_fit(Z, P)
    S = np.linalg.cholesky(np.cov(U.T)); s = S[:, 0] / S[0, 0]
    n = 2; Al = [A[1 + j*n:1 + (j+1)*n].T for j in range(P)]
    R = np.zeros((13, n)); R[0] = s
    for h in range(1, 13): R[h] = sum(Al[j] @ R[h-j-1] for j in range(min(P, h)))
    return R[:, 1], b

def rep_spill(args):
    T, K, pi, theta0, seed = args
    rng = np.random.default_rng(seed)
    Y1, Y0, W, x, lam = sim_spill(T, K, pi, theta0, rng)
    m, D1, H = M.rank_step(Y0)
    syn, _ = cvar_resp(Y1, Y0, W, D1)
    # single-control cycles, then one VAR in (W, G_1, ..., G_K) so that all share the same policy innovation
    Gs = []; bk = []
    for k in range(K):
        a, b = M.dols(Y1, Y0[:, [k]], np.zeros(len(W)), W, np.eye(1), True)
        Gs.append(Y1 - a - Y0[:, k] * b[0]); bk.append(b[0])
    Z = np.column_stack([W] + Gs); n = K + 1
    A, U = M.var_fit(Z, P)
    S = np.linalg.cholesky(np.cov(U.T)); s0 = S[:, 0] / S[0, 0]
    Al = [A[1 + j*n:1 + (j+1)*n].T for j in range(P)]
    R = np.zeros((13, n)); R[0] = s0
    for h in range(1, 13): R[h] = sum(Al[j] @ R[h-j-1] for j in range(min(P, h)))
    g = R[:, 1:].T
    c = np.array(bk) * x
    Xr = np.column_stack([np.ones(K), c])
    expo = np.linalg.lstsq(Xr, g, rcond=None)[0][0]
    return dict(syn=syn, far=g[-1], near=g[0], expo=expo)

# --------------------------------------------------------------- non-linear
def sim_nonlin(T, kappa, theta0, rng, K=3, burn=300):
    n = T + burn
    Lam = np.eye(K) + 0.3; b = np.ones(K) / K; lam = Lam.T @ b
    eta = rng.standard_normal((n, K)); dF = np.zeros((n, K)); F = np.zeros((n, K))
    v = np.zeros((n, K)); e = np.zeros(n); W = np.zeros(n); E = np.zeros(n); u = np.zeros(n)
    ev = rng.standard_normal((n, K)); ee = rng.standard_normal(n); eW = rng.standard_normal(n)
    for t in range(1, n):
        dF[t] = 0.3 * dF[t-1] + eta[t]; F[t] = F[t-1] + dF[t]
        v[t] = 0.5 * v[t-1] + ev[t]; e[t] = 0.5 * e[t-1] + ee[t]
        u[t] = 0.5 * eta[t].sum() / np.sqrt(K) + eW[t]
        W[t] = 0.5 * W[t-1] + u[t]
        E[t] = 0.8 * E[t-1] + theta0 * (u[t] + kappa * u[t] ** 2)
    Y0 = F @ Lam.T + v; Y1 = F @ lam + e + E
    sl = slice(burn, n)
    return Y1[sl], Y0[sl], W[sl]

def rep_nonlin(args):
    T, kappa, theta0, seed = args
    rng = np.random.default_rng(seed)
    Y1, Y0, W = sim_nonlin(T, kappa, theta0, rng)
    m, D1, H = M.rank_step(Y0)
    lin, b = cvar_resp(Y1, Y0, W, D1)
    a, b = M.dols(Y1, Y0, np.zeros(len(W)), W, D1, True)
    G = Y1 - a - Y0 @ b; Z = np.column_stack([W, G])
    A, U = M.var_fit(Z, P); uh = np.r_[np.full(P, np.nan), U[:, 0]]
    out = {}
    lags = M.lagmat(Z, range(1, P + 1))
    for h in HS:
        y = np.r_[G[h:], np.full(h, np.nan)]
        X = np.column_stack([np.ones(len(G)), uh, uh ** 2, lags])
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        co = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        out[h] = (co[1], co[2])
    return dict(lin=lin, dic=out)

# ------------------------------------------------------------- stationarity
def rep_stat(args):
    T, K, rho, seed = args
    rng = np.random.default_rng(seed)
    d = dict(K=K, r=K, feedback=0.0, c_eta=0.5, theta0=0.2, constrained=True, boundary=False, lagrule="12")
    s = M.simulate(d, T, rng)
    # replace the idiosyncratic component of the treated unit with an AR(rho) (rho = 1: unit root)
    e = np.zeros(T); z = rng.standard_normal(T)
    for t in range(1, T): e[t] = rho * e[t-1] + z[t]
    Y1 = s["Y1"] + e
    m, D1, H = M.rank_step(s["Y0"])
    a, b = M.dols(Y1, s["Y0"], s["R"], s["W"], D1, True)
    G = Y1 - a - s["Y0"] @ b
    pol = np.column_stack([s["R"], s["W"]])
    X = np.column_stack([np.ones(T), M.lagmat(pol, range(13))]); ok = ~np.isnan(X).any(1)
    G0 = G[ok] - X[ok, 1:] @ np.linalg.lstsq(X[ok], G[ok], rcond=None)[0][1:]
    res = {}
    for nm, g in [("cycle", G), ("net", G0)]:
        res["adf_" + nm] = adfuller(g, maxlag=12, autolag="BIC")[0]
        res["kpss_" + nm] = kpss(g, regression="c", nlags="auto")[0]
    return res

# ------------------------------------------------------------------- driver
def run(fn, args, workers):
    if workers > 1:
        with Pool(workers) as p: return p.map(fn, args, chunksize=max(1, len(args) // (4 * workers)))
    return [fn(a) for a in args]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "spill", "nonlin", "stat"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_ext")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.reps = 8
    os.makedirs(a.out, exist_ok=True); meta = {}; t0 = time.time()
    if a.exp in ("all", "spill"):
        rows = []; th = theta_spill(0.2)
        for T in [200, 400]:
            for K in [3, 5]:
                for pi in [0.0, 0.5]:
                    res = run(rep_spill, [(T, K, pi, 0.2, 1000 + i) for i in range(a.reps)], a.workers)
                    for est in ["syn", "near", "far", "expo"]:
                        X = np.array([r[est] for r in res])
                        for h in HS:
                            rows.append(dict(T=T, K=K, pi=pi, estimator=est, h=h, truth=round(th[h], 4),
                                             bias=round(X[:, h].mean() - th[h], 4), rmse=round(np.sqrt(((X[:, h] - th[h]) ** 2).mean()), 4)))
                    print(f"spill T={T} K={K} pi={pi} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "spill.csv"), index=False)
    if a.exp in ("all", "nonlin"):
        rows = []; sd = np.sqrt(1.25)
        for T in [200, 400]:
            for kappa in [0.0, 0.5]:
                res = run(rep_nonlin, [(T, kappa, 0.2, 2000 + i) for i in range(a.reps)], a.workers)
                for delta in [-sd, sd, 2 * sd]:
                    for h in HS:
                        truth = 0.2 * 0.8 ** h * (delta + kappa * delta ** 2)
                        lin = np.array([r["lin"][h] * delta for r in res])
                        dic = np.array([r["dic"][h][0] * delta + r["dic"][h][1] * delta ** 2 for r in res])
                        for est, X in [("linear", lin), ("dictionary", dic)]:
                            rows.append(dict(T=T, kappa=kappa, delta=round(delta, 3), estimator=est, h=h, truth=round(truth, 4),
                                             bias=round(X.mean() - truth, 4), rmse=round(np.sqrt(((X - truth) ** 2).mean()), 4)))
                print(f"nonlin T={T} kappa={kappa} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "nonlin.csv"), index=False)
    if a.exp in ("all", "stat"):
        rows = []
        for T in [200, 400]:
            for K in [3, 6]:
                sims = {rho: pd.DataFrame(run(rep_stat, [(T, K, rho, 3000 + i) for i in range(a.reps)], a.workers)) for rho in [0.5, 0.9, 1.0]}
                eg5 = mackinnoncrit(N=min(K + 1, 6), regression="c", nobs=T)[1]
                for stat in ["adf_cycle", "adf_net", "kpss_cycle", "kpss_net"]:
                    adf = stat.startswith("adf")
                    # ADF: null = unit root (rho = 1), reject for small values; KPSS: null = stationarity, reject for large values.
                    std = eg5 if adf else 0.463
                    cv = np.quantile(sims[1.0][stat], 0.05) if adf else np.quantile(sims[0.9][stat], 0.95)
                    for rho in [0.5, 0.9, 1.0]:
                        x = sims[rho][stat]
                        rows.append(dict(T=T, K=K, rho=rho, stat=stat, null="unit root" if adf else "stationarity",
                                         std_cv=round(std, 3), sim_cv=round(cv, 3),
                                         reject_std_cv=round(float(np.mean(x < std) if adf else np.mean(x > std)), 3),
                                         reject_sim_cv=round(float(np.mean(x < cv) if adf else np.mean(x > cv)), 3)))
                print(f"stat T={T} K={K} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "stat.csv"), index=False)
    meta.update(reps=a.reps, seconds=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
