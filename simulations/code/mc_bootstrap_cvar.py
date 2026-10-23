#!/usr/bin/env python3
"""
Monte Carlo for the inference section of the Control VAR paper.

Question: do percentile bootstrap bands for the CVAR impulse response cover the
target when the weights on the controls are estimated, and which bootstrap
scheme is needed?

Estimator (Section 3 of the paper):
  1. Johansen trace test (5%) on the controls -> rank, trend directions D1.
  2. Dynamic OLS of Y1 on a constant, Y0, current and q lags of the policies
     (R, W), and L leads/lags of the changes in the trend directions D1'dY0.
     Optional constraint b >= 0 (solved exactly: free regressors partialled
     out, then NNLS).
  3. Cycle G = Y1 - a - b'Y0.
  4. VAR(p) in (R, W, G), p by AIC (max PMAX), Cholesky with R first, W
     second; response of G to a W innovation normalised to unit impact.

Bootstrap schemes compared (all residual bootstraps, percentile bands):
  plainvar: VAR without controls in (R, W, Y1), same lag length, for the
           comparison with the CVAR (bias against theta).
  oracle : weights known (pseudo-true), VAR bootstrap only. Benchmark.
  fixed  : estimated weights held fixed, VAR bootstrap only (old scheme).
  refit  : rebuild Y1* = a + b'Y0 + G* with the controls held at their
           observed values, re-estimate the weights in every draw (valid only
           when the first-step term is zero).
  joint  : bootstrap a VAR in (dX1, S, R, W, G), with the same lag length as the
           VAR of the CVAR, so the controls are resampled too; rebuild Y0* and Y1*,
           re-estimate the weights, without recentring.
  jointrc: the joint scheme with the draws recentred at the bootstrap population
           value of the response, computed on one long path of the fitted joint VAR
           (the paper's scheme, Theorem thm:joint).
  The rank and the trend directions are held at their sample estimates in
  every draw (inference conditional on the rank pre-test).

Targets (computed once per design from a long simulation):
  gamma_h : pseudo-true response of the cycle, E[G*_{t+h} u_t]/E[u_t^2], the
            probability limit of the estimator. Coverage of gamma checks the
            validity of the bootstrap.
  theta_h : causal effect of a policy surprise, E[Eff_{t+h} u_t]/E[u_t^2],
            with Eff = Y1 - Y1(0). gamma = theta when Assumption 4 holds.

DGP:
  dF_t = 0.3 dF_{t-1} + eta_t                 (r common trends)
  Y0_t = Lam F_t + v_t,  v AR(1) 0.5           (K controls; K - r cointegrating
                                                relations among them if r < K)
  Y1_t(0) = lam'F_t + e_t, e AR(1) 0.5, lam = Lam' b_true
  R_t = 0.7 R_{t-1} + c_eta * sum(eta_t)/sqrt(r) + eps_R          (EUC-like)
  W_t = 0.5 W_{t-1} + c_eta * sum(eta_t)/sqrt(r) + 0.3 R_t
        + feedback * Gtrue_{t-1} + eps_W                           (EB-like)
  Eff_t = 0.8 Eff_{t-1} + theta0 W_t      (theta0 = 0.2 by default; the loop
        policy -> effect -> cycle -> policy is stable iff theta0*feedback < 0.1)
  Y1_t = Y1_t(0) + Eff_t,   Gtrue_t = Y1_t - b_true'Y0_t
  With r < K and feedback > 0 the policy reacts to the cycle built with
  b_true, not with the pseudo-true weights b*, so condition (ii) of
  Proposition 2 fails as well: this is where the first step matters most.

Usage:
  python mc_bootstrap_cvar.py --quick                 # smoke test, 1 design
  python mc_bootstrap_cvar.py --grid --workers 8      # full grid (hours)
  python mc_bootstrap_cvar.py --T 200 --K 6 --r 5 --feedback 0.3 --reps 500
Outputs (in --out, default ./mc_out):
  summary_<design>.csv   coverage, width, bias, RMSE by scheme and horizon
  meta_<design>.json     rank selection frequencies, lag lengths, truths
  all_summaries.csv      everything stacked (send me this file)
"""
import argparse, json, os, time, itertools
import numpy as np
from multiprocessing import Pool
from scipy.optimize import nnls
from scipy.linalg import null_space

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
except ImportError as exc:
    raise SystemExit("statsmodels is required: pip install statsmodels") from exc

HMAX = 24
H_REPORT = [0, 1, 2, 3, 6, 12, 24]
PMAX = 12          # max VAR lag for AIC
Q_POL = 12         # lags of the policies in the DOLS regression
L_DOLS = 1         # leads/lags of the trend directions in DOLS
LEVEL = 0.90       # nominal level of the bands
BURN = 300
SCHEMES = ["oracle", "fixed", "refit", "joint", "jointrc", "plainvar"]


# ----------------------------------------------------------------------------
# DGP
# ----------------------------------------------------------------------------
def design_matrices(K, r, boundary):
    M = np.eye(K) + 0.3
    if r == K:
        Lam = M
    else:
        Lam = np.vstack([np.eye(r), np.ones((K - r, r)) / r]) + 0.3
    b_true = np.ones(K) / K
    if boundary:
        b_true[-1] = 0.0
        b_true[:-1] = 1.0 / (K - 1)
    lam = Lam.T @ b_true
    return Lam, b_true, lam


def simulate(d, T, rng, burn=BURN):
    K, r = d["K"], d["r"]
    Lam, b_true, lam = design_matrices(K, r, d["boundary"])
    n = T + burn
    eta = rng.standard_normal((n, r))
    ev = rng.standard_normal((n, K))
    ee = rng.standard_normal(n)
    eR = rng.standard_normal(n)
    eW = rng.standard_normal(n)
    dF = np.zeros((n, r)); v = np.zeros((n, K)); e = np.zeros(n)
    R = np.zeros(n); W = np.zeros(n); Eff = np.zeros(n)
    Gtrue = np.zeros(n)
    F = np.zeros((n, r))
    ceta = d["c_eta"]
    common = eta.sum(1) / np.sqrt(r)
    for t in range(1, n):
        dF[t] = 0.3 * dF[t - 1] + eta[t]
        F[t] = F[t - 1] + dF[t]
        v[t] = 0.5 * v[t - 1] + ev[t]
        e[t] = 0.5 * e[t - 1] + ee[t]
        R[t] = 0.7 * R[t - 1] + ceta * common[t] + eR[t]
        W[t] = (0.5 * W[t - 1] + ceta * common[t] + 0.3 * R[t]
                + d["feedback"] * Gtrue[t - 1] + eW[t])
        Eff[t] = 0.8 * Eff[t - 1] + d["theta0"] * W[t]
        y0 = Lam @ F[t] + v[t]
        y1 = lam @ F[t] + e[t] + Eff[t]
        Gtrue[t] = y1 - b_true @ y0
    Y0 = F @ Lam.T + v
    Y1 = F @ lam + e + Eff
    sl = slice(burn, n)
    return dict(Y1=Y1[sl], Y0=Y0[sl], R=R[sl], W=W[sl], Eff=Eff[sl])


# ----------------------------------------------------------------------------
# Estimation pieces
# ----------------------------------------------------------------------------
def lagmat(x, lags):
    """Columns x_{t-j} for j in lags (negative = lead); rows aligned to t."""
    T = x.shape[0]
    x = x.reshape(T, -1)
    out = []
    for j in lags:
        z = np.full_like(x, np.nan)
        if j >= 0:
            z[j:] = x[:T - j] if j > 0 else x
        else:
            z[:T + j] = x[-j:]
        out.append(z)
    return np.hstack(out)


def dols(Y1, Y0, R, W, D1, constrained):
    """Dynamic OLS with policy terms and leads/lags of D1'dY0. Returns a, b."""
    T, K = Y0.shape
    dX1 = np.vstack([np.full((1, D1.shape[1]), np.nan), np.diff(Y0, axis=0) @ D1])
    free = np.hstack([np.ones((T, 1)),
                      lagmat(np.column_stack([R, W]), range(0, Q_POL + 1)),
                      lagmat(dX1, range(-L_DOLS, L_DOLS + 1))])
    ok = ~np.isnan(free).any(1)
    y, X, Fm = Y1[ok], Y0[ok], free[ok]
    if not constrained:
        Z = np.hstack([Fm, X])
        coef = np.linalg.lstsq(Z, y, rcond=None)[0]
        a, b = coef[0], coef[Fm.shape[1]:]
        return a, b
    # Frisch-Waugh: partial out the free regressors, then NNLS on b
    Pinv = np.linalg.pinv(Fm)
    yt = y - Fm @ (Pinv @ y)
    Xt = X - Fm @ (Pinv @ X)
    b, _ = nnls(Xt, yt)
    g = Pinv @ (y - X @ b)
    return g[0], b


def var_design(Z, p):
    T, n = Z.shape
    X = np.hstack([np.ones((T - p, 1))] + [Z[p - j - 1:T - j - 1] for j in range(p)])
    return X, Z[p:]


def var_fit(Z, p):
    X, Y = var_design(Z, p)
    A = np.linalg.lstsq(X, Y, rcond=None)[0]
    U = Y - X @ A
    return A, U


def ic_lags(Z, pmax=PMAX):
    """Lag lengths chosen by AIC, HQ and BIC on a common sample."""
    T, n = Z.shape
    crit = {"aic": [], "hq": [], "bic": []}
    for p in range(1, pmax + 1):
        X, Y = var_design(Z[pmax - p:], p)
        A = np.linalg.lstsq(X, Y, rcond=None)[0]
        U = Y - X @ A
        Te = U.shape[0]
        ld = np.log(np.linalg.det(U.T @ U / Te))
        k = p * n * n
        crit["aic"].append(ld + 2 * k / Te)
        crit["hq"].append(ld + 2 * k * np.log(np.log(Te)) / Te)
        crit["bic"].append(ld + k * np.log(Te) / Te)
    return {c: int(np.argmin(v)) + 1 for c, v in crit.items()}


def aic_lag(Z, pmax=PMAX):
    return ic_lags(Z, pmax)["aic"]


def ljungbox_p(x, m=12):
    """Ljung-Box p-value at m lags for a univariate residual series."""
    from scipy.stats import chi2
    x = x - x.mean(); n = len(x); d = x @ x
    q = sum((x[k:] @ x[:-k] / d) ** 2 / (n - k) for k in range(1, m + 1)) * n * (n + 2)
    return 1 - chi2.cdf(q, m)


def lb_cycle(Z, p):
    _, U = var_fit(Z, p)
    return ljungbox_p(U[:, -1])


def choose_lag(Z, rule):
    """aic: AIC. paper: smallest of BIC/HQ/AIC whose cycle-equation residuals pass
    Ljung-Box(12) at 5%, else AIC. lbup: smallest p passing Ljung-Box, else PMAX.
    An integer: fixed lag length."""
    if rule.isdigit():
        return int(rule)
    if rule == "aic":
        return aic_lag(Z)
    if rule == "paper":
        ic = ic_lags(Z)
        for q in sorted(set(ic.values())):
            if lb_cycle(Z, q) > 0.05:
                return q
        return ic["aic"]
    if rule == "lbup":
        for q in range(1, PMAX + 1):
            if lb_cycle(Z, q) > 0.05:
                return q
        return PMAX
    raise ValueError(rule)


def irf(A, U, p, n, shock=1, resp=2, H=HMAX):
    S = np.linalg.cholesky(np.cov(U.T))
    s = S[:, shock] / S[shock, shock]
    Al = [A[1 + j * n:1 + (j + 1) * n].T for j in range(p)]
    Rr = np.zeros((H + 1, n)); Rr[0] = s
    for h in range(1, H + 1):
        Rr[h] = sum(Al[j] @ Rr[h - j - 1] for j in range(min(p, h)))
    return Rr[:, resp]


def var_sim(A, U, Z0, p, T, rng):
    n = U.shape[1]
    E = U[rng.integers(0, U.shape[0], T)] - U.mean(0)
    Z = np.zeros((T + p, n)); Z[:p] = Z0
    for t in range(p, T + p):
        x = np.concatenate([[1.0]] + [Z[t - j - 1] for j in range(p)])
        Z[t] = x @ A + E[t - p]
    return Z


def aic_lag_levels(Y, pmax=6):
    return aic_lag(Y, pmax)


def rank_step(Y0):
    """Johansen trace test at 5% on the controls. Returns rank, D1, H."""
    K = Y0.shape[1]
    k = max(aic_lag_levels(Y0) - 1, 1)
    jo = coint_johansen(Y0, 0, k)
    m = next((i for i in range(K) if jo.lr1[i] < jo.cvt[i, 1]), K)  # coint rank
    if m == 0:
        return 0, np.eye(K), np.zeros((K, 0))
    H = jo.evec[:, :m]
    D1 = null_space(H.T)
    return m, D1, H


# ----------------------------------------------------------------------------
# Truth
# ----------------------------------------------------------------------------
def truth(d, T_long=100_000, p_long=24, seed=987654):
    rng = np.random.default_rng(seed)
    s = simulate(d, T_long, rng)
    K, r = d["K"], d["r"]
    Lam, _, _ = design_matrices(K, r, d["boundary"])
    D1 = np.eye(K) if r == K else null_space(null_space(Lam.T).T)
    global L_DOLS
    Lsave = L_DOLS; L_DOLS = 6           # pseudo-true weights: long leads/lags
    a, b = dols(s["Y1"], s["Y0"], s["R"], s["W"], D1, d["constrained"])
    L_DOLS = Lsave
    G = s["Y1"] - a - s["Y0"] @ b
    # policy innovation: residual of W on R_t and p_long lags of (R, W, G)
    Z = np.column_stack([s["R"], s["W"], G])
    X = np.hstack([np.ones((T_long - p_long, 1)), s["R"][p_long:, None]]
                  + [Z[p_long - j - 1:T_long - j - 1] for j in range(p_long)])
    w = s["W"][p_long:]
    u = w - X @ np.linalg.lstsq(X, w, rcond=None)[0]
    Gs, Es = G[p_long:], s["Eff"][p_long:]
    n = len(u) - HMAX
    gam = np.array([Gs[h:h + n] @ u[:n] for h in range(HMAX + 1)]) / (u[:n] @ u[:n])
    the = np.array([Es[h:h + n] @ u[:n] for h in range(HMAX + 1)]) / (u[:n] @ u[:n])
    return dict(gamma=gam, theta=the, a_star=float(a), b_star=b.tolist())


# ----------------------------------------------------------------------------
# One replication
# ----------------------------------------------------------------------------
def one_rep(args):
    d, T, B, seed, tr = args
    rng = np.random.default_rng(seed)
    s = simulate(d, T, rng)
    Y1, Y0, R, W = s["Y1"], s["Y0"], s["R"], s["W"]
    K = Y0.shape[1]
    m, D1, H = rank_step(Y0)
    a, b = dols(Y1, Y0, R, W, D1, d["constrained"])
    G = Y1 - a - Y0 @ b
    Z = np.column_stack([R, W, G])
    p = choose_lag(Z, d["lagrule"])
    A, U = var_fit(Z, p)
    est = irf(A, U, p, 3)
    lb_rej = lb_cycle(Z, p) <= 0.05

    # oracle: pseudo-true weights known
    Go = Y1 - tr["a_star"] - Y0 @ np.array(tr["b_star"])
    Zo = np.column_stack([R, W, Go])
    Ao, Uo = var_fit(Zo, p)
    est_o = irf(Ao, Uo, p, 3)

    # VAR without controls, in (R, W, Y1), same lag length
    Zv = np.column_stack([R, W, Y1])
    Av, Uv = var_fit(Zv, p)
    est_v = irf(Av, Uv, p, 3)

    draws = {k: np.empty((B, HMAX + 1)) for k in SCHEMES}
    # joint VAR for the controls
    if m > 0:
        Dfull = np.hstack([D1, H])
    else:
        Dfull = np.eye(K)
    Xr = Y0 @ Dfull                      # rotated controls (X1, S)
    r_hat = D1.shape[1]
    dX1 = np.diff(Xr[:, :r_hat], axis=0)
    Sx = Xr[1:, r_hat:]
    J = np.column_stack([dX1, Sx, R[1:], W[1:], G[1:]])
    # lag length of the joint VAR: the same as the VAR of the CVAR (Theorem thm:joint) unless --joint-lags is set
    pj = p if d.get("joint_lags") is None else min(d["joint_lags"], p)
    Aj, Uj = var_fit(J, pj)
    Dinv_T = np.linalg.inv(Dfull)          # Xr = Y0 @ Dfull, so Y0 = Xr @ inv(Dfull) (was inv(Dfull.T), wrong when r < K)

    for i in range(B):
        # oracle
        Zs = var_sim(Ao, Uo, Zo[:p], p, T - p, rng)
        A2, U2 = var_fit(Zs, p); draws["oracle"][i] = irf(A2, U2, p, 3)
        # fixed weights
        Zs = var_sim(A, U, Z[:p], p, T - p, rng)
        A2, U2 = var_fit(Zs, p); draws["fixed"][i] = irf(A2, U2, p, 3)
        # refit with controls fixed
        Zs = var_sim(A, U, Z[:p], p, T - p, rng)
        y1s = a + Y0 @ b + Zs[:, 2]
        a2, b2 = dols(y1s, Y0, Zs[:, 0], Zs[:, 1], D1, d["constrained"])
        Z2 = np.column_stack([Zs[:, 0], Zs[:, 1], y1s - a2 - Y0 @ b2])
        A2, U2 = var_fit(Z2, p); draws["refit"][i] = irf(A2, U2, p, 3)
        # joint: resample the controls as well
        Js = var_sim(Aj, Uj, J[:pj], pj, T - 1 - pj, rng)
        X1s = np.vstack([Xr[:1, :r_hat], Xr[0, :r_hat] + np.cumsum(Js[:, :r_hat], 0)])
        Ss = np.vstack([Xr[:1, r_hat:], Js[:, r_hat:K]])
        Y0s = np.column_stack([X1s, Ss]) @ Dinv_T
        Rs = np.concatenate([[R[0]], Js[:, K]])
        Ws = np.concatenate([[W[0]], Js[:, K + 1]])
        Gs = np.concatenate([[G[0]], Js[:, K + 2]])
        y1s = a + Y0s @ b + Gs
        a2, b2 = dols(y1s, Y0s, Rs, Ws, D1, d["constrained"])
        Z2 = np.column_stack([Rs, Ws, y1s - a2 - Y0s @ b2])
        A2, U2 = var_fit(Z2, p); draws["joint"][i] = irf(A2, U2, p, 3)
        # VAR without controls
        Zs = var_sim(Av, Uv, Zv[:p], p, T - p, rng)
        A2, U2 = var_fit(Zs, p); draws["plainvar"][i] = irf(A2, U2, p, 3)

    # recentred joint scheme (Theorem thm:joint): the bootstrap population value of the response is
    # computed from one long path of the fitted joint VAR, and the draws are shifted by est - pop
    NL = int(d.get("recentre_mult") or 50) * (T - 1)
    JL = var_sim(Aj, Uj, J[:pj], pj, NL - pj, rng)
    X1L = np.vstack([Xr[:1, :r_hat], Xr[0, :r_hat] + np.cumsum(JL[:, :r_hat], 0)])
    SL = np.vstack([Xr[:1, r_hat:], JL[:, r_hat:K]])
    Y0L = np.column_stack([X1L, SL]) @ Dinv_T
    RL = np.concatenate([[R[0]], JL[:, K]])
    WL = np.concatenate([[W[0]], JL[:, K + 1]])
    GL = np.concatenate([[G[0]], JL[:, K + 2]])
    y1L = a + Y0L @ b + GL
    aL, bL = dols(y1L, Y0L, RL, WL, D1, d["constrained"])
    ZL = np.column_stack([RL, WL, y1L - aL - Y0L @ bL])
    AL, UL = var_fit(ZL, p)
    pop_joint = irf(AL, UL, p, 3)
    draws["jointrc"] = draws["joint"] + (est - pop_joint)

    al = (1 - LEVEL) / 2 * 100
    out = dict(rank_coint=m, p=p, pj=pj, lb_rej=lb_rej)
    for k in SCHEMES:
        point = est_o if k == "oracle" else (est_v if k == "plainvar" else est)
        lo, hi = np.percentile(draws[k], [al, 100 - al], axis=0)
        out[k] = dict(point=point, lo=lo, hi=hi)
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def tag(d, T):
    return (f"T{T}_K{d['K']}_r{d['r']}_fb{d['feedback']}"
            f"_{'nn' if d['constrained'] else 'uc'}{'_bd' if d['boundary'] else ''}"
            f"_lag-{d['lagrule']}")


def check_stable(d):
    """Policy-effect loop: W_t = 0.5 W_{t-1} + f E_{t-1} + ..., E_t = 0.8 E_{t-1} + theta0 W_t."""
    f, th = d["feedback"], d["theta0"]
    M = np.array([[0.5, f], [0.5 * th, 0.8 + th * f]])
    root = max(abs(np.linalg.eigvals(M)))
    if root >= 0.99:
        raise SystemExit(f"Design {d} is explosive or near unit root (largest root {root:.3f}); "
                         "lower --feedback or --theta0 (stability needs theta0*feedback < 0.1).")
    return root


def run_design(d, T, reps, B, workers, outdir, seed0):
    t0 = time.time()
    check_stable(d)
    tr = truth(d)
    args = [(d, T, B, seed0 + i, tr) for i in range(reps)]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(one_rep, args, chunksize=max(1, reps // (4 * workers)))
    else:
        res = [one_rep(a) for a in args]
    rows = []
    for k in SCHEMES:
        pts = np.array([r_[k]["point"] for r_ in res])
        lo = np.array([r_[k]["lo"] for r_ in res])
        hi = np.array([r_[k]["hi"] for r_ in res])
        for h in H_REPORT:
            for tname in ["gamma", "theta"]:
                tv = tr[tname][h]
                rows.append(dict(design=tag(d, T), T=T, K=d["K"], r=d["r"],
                    feedback=d["feedback"], constrained=d["constrained"],
                    boundary=d["boundary"], lagrule=d["lagrule"], scheme=k, h=h, target=tname,
                    truth=round(float(tv), 4),
                    coverage=round(float(np.mean((lo[:, h] <= tv) & (tv <= hi[:, h]))), 3),
                    width=round(float(np.mean(hi[:, h] - lo[:, h])), 4),
                    bias=round(float(np.mean(pts[:, h]) - tv), 4),
                    rmse=round(float(np.sqrt(np.mean((pts[:, h] - tv) ** 2))), 4)))
    ranks = np.array([r_["rank_coint"] for r_ in res])
    meta = dict(design=tag(d, T), reps=reps, B=B, level=LEVEL,
                true_coint_rank_controls=d["K"] - d["r"],
                freq_coint_rank={int(k): float(np.mean(ranks == k)) for k in np.unique(ranks)},
                lagrule=d["lagrule"],
                mean_p=float(np.mean([r_["p"] for r_ in res])),
                freq_p={int(k): float(np.mean(np.array([r_["p"] for r_ in res]) == k)) for k in np.unique([r_["p"] for r_ in res])},
                share_ljungbox_reject_cycle=float(np.mean([r_["lb_rej"] for r_ in res])),
                gamma=[round(x, 4) for x in tr["gamma"].tolist()],
                theta=[round(x, 4) for x in tr["theta"].tolist()],
                b_star=[round(x, 4) for x in tr["b_star"]],
                seconds=round(time.time() - t0, 1))
    os.makedirs(outdir, exist_ok=True)
    import csv
    with open(os.path.join(outdir, f"summary_{tag(d, T)}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, f"meta_{tag(d, T)}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{tag(d, T)}] done in {meta['seconds']}s; rank freq {meta['freq_coint_rank']}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--r", type=int, default=6)
    ap.add_argument("--feedback", type=float, default=0.3)
    ap.add_argument("--joint-lags", type=int, default=None, help="lag length of the joint VAR (default: same as the CVAR)")
    ap.add_argument("--c_eta", type=float, default=0.5, help="policy reaction to common shocks")
    ap.add_argument("--theta0", type=float, default=0.2)
    ap.add_argument("--unconstrained", action="store_true")
    ap.add_argument("--boundary", action="store_true", help="one true weight at zero (use with r = K)")
    ap.add_argument("--lagrule", default="aic",
                    help="aic | paper | lbup | an integer for a fixed lag length")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--boot", type=int, default=499)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_out")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--grid", action="store_true", help="run the full design grid")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 design, 4 reps, 49 draws")
    a = ap.parse_args()

    base = dict(c_eta=a.c_eta, theta0=a.theta0, constrained=not a.unconstrained,
                boundary=a.boundary, lagrule=a.lagrule, joint_lags=a.joint_lags)
    if a.quick:
        designs = [(dict(base, K=3, r=2, feedback=0.3), 200)]
        a.reps, a.boot = 4, 49
    elif a.grid:
        designs = []
        for T, (K, r), fb in itertools.product([200, 400], [(3, 3), (3, 2), (6, 6), (6, 5)], [0.0, 0.3]):
            designs.append((dict(base, K=K, r=r, feedback=fb), T))
        # boundary case for the constraint
        designs.append((dict(base, K=3, r=3, feedback=0.3, boundary=True), 200))
    else:
        designs = [(dict(base, K=a.K, r=a.r, feedback=a.feedback), a.T)]

    allrows = []
    for i, (d, T) in enumerate(designs):
        allrows += run_design(d, T, a.reps, a.boot, a.workers, a.out, a.seed + 100000 * i)
    import csv
    with open(os.path.join(a.out, "all_summaries.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("Wrote", os.path.join(a.out, "all_summaries.csv"))


if __name__ == "__main__":
    main()
