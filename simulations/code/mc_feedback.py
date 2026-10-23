#!/usr/bin/env python3
"""
Does the CVAR test survive a policy that reacts to the treated unit's own changes?
Compares the current statistic (PO) with the augmented statistic (APO) that adds lags of dY1 to both
quasi-differenced regressions. Needs, in the same folder: robust_po.py (and the files it imports:
mc_power.py, mc_envelope.py, mc_bootstrap_cvar.py, po_critical_values.py).

Data generating process (T in {200, 400}, K = 3 controls with r = 3 trends)
  common trends     F_t = F_{t-1} + eta_t,  eta_t ~ N(0, I_3)
  controls          Y0_t = Lam F_t + n_t,  Lam = I + 0.3,  n_kt AR(1) with coefficient 0.5
  treated           Y1_t = lam'F_t + e_t + E_t + u_t,  lam = Lam'b, b = (1/3, 1/3, 1/3), e_t AR(1) with 0.5
  own component     u_t = rho u_{t-1} + v_t,  v_t = phi_v v_{t-1} + eps_t,  u_0 = v_0 = 0,  rho = 1 - g/T
                    (g = 0: the treated unit has a trend of its own, null hypothesis; g > 0: CVAR valid)
  policies          R_t = 0.7 R_{t-1} + 0.5 mean(eta_t) + e_Rt
                    W_t = 0.5 W_{t-1} + 0.5 mean(eta_t) + 0.3 R_t + fb * (Y1_{t-1} - Y1_{t-13}) / sqrt(12)
                          + lev * u_{t-1} / 4 + e_Wt
                    (fb: reaction to the treated unit's own twelve-month change, as EB triggers;
                     lev: reaction to the level of its own component, which makes the policy share the
                     treated unit's trend under the null)
  effect            E_t = 0.8 E_{t-1} + 0.2 W_t
Designs: (fb, phi_v, lev) in {(0,0,0), (0,0.5,0), (0.5,0,0), (0.5,0.5,0), (0,0,0.3)}.
Local alternatives g in {0, 5, 10, 20}.

Statistics (cbar = -7; reject for SMALL values; regressors: constant, controls, current and 12 lags of R and W)
  PO      current test: long-run variance from the MAIC ADF regression on the GLS-detrended residual
          (robust_po.po_maic); critical value 3.26 (ERS 1996, constant, 5%).
  APO_p   augmented test: p lags of dY1 added, not quasi-differenced, to the regressions at alpha_bar and at 1;
          omega2 = s2 / (1 - sum of the lag coefficients)^2 from the alpha = 1 regression; p in {4, 12};
          critical value 3.26.
  PO_BOOT sieve-bootstrap p-value of PO (AR(12), controls and policies held fixed), at g in {0, 10} only.
Reported: actual rejection rates with the ERS critical value (size at g = 0), size-adjusted power, and the
bootstrap rejection rate.
Output in --out (default mc_feedback): results.csv (one row per design, T and g).
Usage: py mc_feedback.py              (1000 replications, 199 bootstrap draws)
"""
import argparse, os, time, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
import warnings; warnings.filterwarnings("ignore")
import robust_po as RP
import po_critical_values as PC

CBAR = -7.0
ERS_CV5 = 3.26
NLAG = 12
GS = [0.0, 5.0, 10.0, 20.0]
BOOT_GS = [0.0, 10.0]
DESIGNS = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.3)]

def simulate(T, g, fb, phi_v, lev, rng, burn=100):
    K = 3; r = 3; n = burn + NLAG + T; t0 = burn + NLAG
    Lam = np.eye(K) + 0.3; b = np.ones(K) / K; lam = Lam.T @ b
    eta = rng.standard_normal((n, r)); F = np.cumsum(eta, axis=0); ceta = eta.mean(1)
    noise = lfilter([1.0], [1.0, -0.5], rng.standard_normal((n, K)), axis=0)
    Y0 = F @ Lam.T + noise
    e = lfilter([1.0], [1.0, -0.5], rng.standard_normal(n))
    eps = np.zeros(n); eps[t0:] = rng.standard_normal(T)          # own component starts at the sample start
    v = lfilter([1.0], [1.0, -phi_v], eps)
    u = lfilter([1.0], [1.0, -(1 - g / T)], v)
    R = lfilter([1.0], [1.0, -0.7], 0.5 * ceta + rng.standard_normal(n))
    eW = rng.standard_normal(n); W = np.zeros(n); E = np.zeros(n); Y1 = np.zeros(n)
    base = F @ lam + e + u
    for t in range(n):
        react = 0.0
        if t >= 13: react = fb * (Y1[t - 1] - Y1[t - 13]) / np.sqrt(12)
        W[t] = (0.5 * W[t - 1] if t else 0.0) + 0.5 * ceta[t] + 0.3 * R[t] + react + lev * (u[t - 1] if t else 0.0) / 4 + eW[t]
        E[t] = (0.8 * E[t - 1] if t else 0.0) + 0.2 * W[t]
        Y1[t] = base[t] + E[t]
    P = PC.lagmat(np.column_stack([R, W]), range(0, NLAG + 1))
    sl = slice(t0, n)
    X = np.column_stack([np.ones(T), Y0[sl], P[sl]])
    return Y1[sl], X

def apo(y, X, p, cbar=CBAR):
    """Augmented point-optimal statistic: p lags of dy added (not quasi-differenced) at alpha_bar and at 1."""
    T = len(y); a = 1 + cbar / T
    dy = np.r_[np.nan, np.diff(y)]
    L = np.column_stack([np.r_[np.full(j, np.nan), dy[:-j]] for j in range(1, p + 1)])
    ok = ~np.isnan(L).any(1)
    yy, XX, LL = y[ok], X[ok], L[ok]
    def S(alpha):
        Ya = np.r_[yy[0], yy[1:] - alpha * yy[:-1]]
        Xa = np.vstack([XX[0], XX[1:] - alpha * XX[:-1]])
        Z = np.column_stack([Xa, LL])
        coef = np.linalg.pinv(Z) @ Ya; res = Ya - Z @ coef
        return res @ res, res, coef[-p:]
    S_a, _, _ = S(a)
    S_1, e1, phi = S(1.0)
    s2 = e1[1:] @ e1[1:] / (len(e1) - 1)
    w2 = s2 / (1 - phi.sum()) ** 2
    return (S_a - a * S_1) / w2

def boot_p(y, X, B, rng, psieve=12):
    stat = RP.po_maic(y, X, CBAR)
    dy = np.diff(y); dX = np.diff(X, axis=0); keep = np.abs(dX).sum(0) > 1e-12
    Z = np.column_stack([np.ones(len(dy)), dX[:, keep]])
    gcoef = np.linalg.lstsq(Z, dy, rcond=None)[0]; fit = Z @ gcoef; e = dy - fit
    Zp = np.column_stack([e[psieve - j:len(e) - j] for j in range(1, psieve + 1)])
    phi = np.linalg.lstsq(Zp, e[psieve:], rcond=None)[0]; res = e[psieve:] - Zp @ phi; res = res - res.mean()
    cnt = 0
    for _ in range(B):
        es = lfilter([1.0], np.r_[1.0, -phi], res[rng.integers(0, len(res), len(e) + 100)])[100:]
        ys = np.r_[y[0], y[0] + np.cumsum(fit + es)]
        cnt += RP.po_maic(ys, X, CBAR) <= stat
    return (cnt + 1) / (B + 1)

def one(args):
    T, g, fb, phi_v, lev, seed, B = args
    rng = np.random.default_rng(seed)
    y, X = simulate(T, g, fb, phi_v, lev, rng)
    out = dict(PO=RP.po_maic(y, X, CBAR), APO_4=apo(y, X, 4), APO_12=apo(y, X, 12))
    if B > 0: out["PO_BOOT_p"] = boot_p(y, X, B, rng)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=199)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="mc_feedback")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.csv")
    if os.path.exists(path): os.remove(path)
    t0 = time.time(); rows = []
    with Pool(a.workers) as pool:
        for T in [200, 400]:
            for di, (fb, phi_v, lev) in enumerate(DESIGNS):
                res = {}
                for gj, g in enumerate(GS):
                    B = a.boot if g in BOOT_GS else 0
                    args = [(T, g, fb, phi_v, lev, 2_000_000_000 + 100_000_000 * di + 10_000_000 * gj + 1_000_000 * (T == 400) + i, B)
                            for i in range(a.reps)]
                    res[g] = pd.DataFrame(pool.map(one, args, chunksize=1 if B else max(1, a.reps // 64)))
                null = res[0.0]
                for g in GS:
                    d = res[g]; row = dict(T=T, fb=fb, phi_v=phi_v, lev=lev, g=g)
                    for t in ["PO", "APO_4", "APO_12"]:
                        row[t] = round(float(np.mean(d[t] < ERS_CV5)), 3)
                        row[t + "_sa"] = round(float(np.mean(d[t] < np.quantile(null[t], 0.05))), 3)
                    if "PO_BOOT_p" in d: row["PO_BOOT"] = round(float(np.mean(d.PO_BOOT_p <= 0.05)), 3)
                    rows.append(row)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"T={T} fb={fb} phi_v={phi_v} lev={lev} done ({time.time()-t0:.0f}s)", flush=True)
    print("Wrote", path)

if __name__ == "__main__":
    main()
