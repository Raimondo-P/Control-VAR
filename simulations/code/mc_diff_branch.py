"""mc_diff_branch.py: Monte Carlo for the CVAR in changes with imposed weights (Theorem thm:diff).

Data generating process (monthly, T observations after a burn-in of 100):
  common trend      F_t = F_{t-1} + eta_t,                              eta_t ~ N(0, 1)
  controls          Y0_{kt} = F_t + v_{kt},  v_{kt} = 0.5 v_{k,t-1} + sigma_v * N(0, 1),  k = 1..K
  treated unit      Y1_t = F_t + e_t + sum_{j=0}^{2} tau_j W_{t-j},  tau = (0.5, 0.3, 0.1)
      own trend     e_t = e_{t-1} + sigma_e * N(0, 1)
      stationary    e_t = 0.8 e_{t-1} + sigma_e * N(0, 1)
  policy            W_t = 0.7 W_{t-1} + c * eta_t + eps_t,              eps_t ~ N(0, 1)
The policy reacts to the common shock with coefficient c. Its innovation is u_t = c eta_t + eps_t and its
response to its own innovation is psi_k = 0.7^k, so the true cumulated response is
  Gamma_h = sum_{j=0}^{min(h,2)} tau_j 0.7^(h-j),
and a VAR without controls adds the common-trend term c / (1 + c^2) at every horizon.

Designs:
  own_trend    e has a trend of its own, K = 4, sigma_v = 0.5, sigma_e = 0.3, c = 1
  stationary   e stationary,            K = 4, sigma_v = 0.5, sigma_e = 0.5, c = 1
  noisy_single own trend, one control with sigma_v = 2 (the controls add more variance than they remove)
  no_reaction  own trend, K = 4, sigma_v = 0.5, c = 0 (the VAR without controls is unbiased)
Estimators (VAR with p lags, policy ordered first):
  cvar_diff    CVAR in changes, equal weights (Theorem thm:diff), residual bootstrap, responses cumulated
  var_diff     VAR without controls in (W, change in Y1), cumulated
  cvar_levels  CVAR in levels with weights estimated by dynamic least squares, bootstrap re-estimating them
Reported for h = 0, 3, 6, 12: bias, RMSE, coverage of 90 percent percentile bands, mean band width.
Outputs: results/mc_diff/mc_diff.csv and tab_mc_diff.tex
Usage: py mc_diff_branch.py [--R 500] [--B 199] [--T 200] [--p 4] [--workers N]
"""
import os
import sys
import argparse
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "applications"))
import cvar  # noqa: E402

TAU = np.array([0.5, 0.3, 0.1])
RHO_W = 0.7
HOR = (0, 3, 6, 12)
H = 12
DESIGNS = {
    "own_trend": dict(trend=True, K=4, sv=0.5, se=0.3, c=1.0),
    "stationary": dict(trend=False, K=4, sv=0.5, se=0.5, c=1.0),
    "noisy_single": dict(trend=True, K=1, sv=2.0, se=0.3, c=1.0),
    "no_reaction": dict(trend=True, K=4, sv=0.5, se=0.3, c=0.0),
}
LABELS = {"own_trend": "Trend of the treated unit", "stationary": "Stationary cycle",
          "noisy_single": "One noisy control", "no_reaction": "No reaction to the common shock"}
EST = {"cvar_diff": "CVAR in changes", "var_diff": "VAR without controls", "cvar_levels": "CVAR in levels"}


def true_gamma():
    g = np.zeros(H + 1)
    for h in range(H + 1):
        g[h] = sum(TAU[j] * RHO_W ** (h - j) for j in range(min(h, 2) + 1))
    return g


def simulate(d, T, rng, burn=100):
    n = T + burn
    eta = rng.standard_normal(n)
    F = np.cumsum(eta)
    v = np.zeros((n, d["K"]))
    z = d["sv"] * rng.standard_normal((n, d["K"]))
    for t in range(1, n):
        v[t] = 0.5 * v[t - 1] + z[t]
    zeta = d["se"] * rng.standard_normal(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = (e[t - 1] if d["trend"] else 0.8 * e[t - 1]) + zeta[t]
    W = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        W[t] = RHO_W * W[t - 1] + d["c"] * eta[t] + eps[t]
    eff = np.zeros(n)
    for j, tj in enumerate(TAU):
        eff[j:] += tj * W[:n - j]
    Y1 = F + e + eff
    Y0 = F[:, None] + v
    return Y1[burn:], Y0[burn:], W[burn:]


def one(item):
    name, rep, T, B, p = item
    d = DESIGNS[name]
    rng = np.random.default_rng(100000 * list(DESIGNS).index(name) + rep)
    y, C, W = simulate(d, T, rng)
    w = np.ones(d["K"]) / d["K"]
    out = []
    try:
        r = cvar.CVAR(p=p).fit(y, C, W, weights=w, difference=True)
        bt = r.bootstrap(B, H, seed=rep, no_controls=True)
        for key, est in [("cycle", "cvar_diff"), ("no_controls", "var_diff")]:
            lo, hi = bt.bands(key, 0.10)
            out.append((est, bt.point[key][:, 0], lo[:, 0], hi[:, 0]))
    except Exception:
        pass
    try:
        r2 = cvar.CVAR(p=p, q=4, L=1).fit(y, C, W)
        b2 = r2.bootstrap(B, H, seed=rep + 7, no_controls=False)
        lo, hi = b2.bands("cycle", 0.10)
        # the levels CVAR responds in levels of the cycle, which is the cumulated response from t-1
        out.append(("cvar_levels", b2.point["cycle"][:, 0], lo[:, 0], hi[:, 0]))
    except Exception:
        pass
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--B", type=int, default=199)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "mc_diff"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = true_gamma()
    items = [(nm, rep, a.T, a.B, a.p) for nm in DESIGNS for rep in range(a.R)]
    with Pool(a.workers) as P:
        res = P.map(one, items, chunksize=4)
    rows = []
    for nm in DESIGNS:
        for est in EST:
            draws = [(pt, lo, hi) for n_, out in res if n_ == nm for e_, pt, lo, hi in out if e_ == est]
            if not draws:
                continue
            PT = np.array([x[0] for x in draws])
            LO = np.array([x[1] for x in draws])
            HI = np.array([x[2] for x in draws])
            for h in HOR:
                rows.append(dict(design=nm, estimator=est, h=h, true=g[h], reps=len(PT),
                                 mean=PT[:, h].mean(), bias=PT[:, h].mean() - g[h],
                                 rmse=np.sqrt(((PT[:, h] - g[h]) ** 2).mean()),
                                 coverage=((LO[:, h] <= g[h]) & (g[h] <= HI[:, h])).mean(),
                                 width=(HI[:, h] - LO[:, h]).mean()))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "mc_diff.csv"), index=False)
    print(R.round(3).to_string(index=False))

    def f(x):
        return f"{x:.2f}".replace("-", "$-$")
    lines = []
    for nm in DESIGNS:
        lines.append(r"\multicolumn{8}{l}{" + LABELS[nm] + r"} \\")
        for est in EST:
            z = R[(R.design == nm) & (R.estimator == est)].set_index("h")
            if z.empty:
                continue
            cells = [EST[est]] + [f(z.loc[h].bias) for h in (0, 6, 12)] + \
                    [f(z.loc[h].coverage) for h in (0, 6, 12)] + [f(z.loc[6].width)]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    tab = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{The CVAR in changes with imposed weights\label{tab:mc-diff}}", r"\small",
           r"\begin{tabular}{lccccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{Bias} & \multicolumn{3}{c}{Coverage} & Width \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
           r"Horizon & 0 & 6 & 12 & 0 & 6 & 12 & 6 \\", r"\midrule"] + lines + [
           r"\bottomrule", r"\end{tabular}",
           r"\par\smallskip\parbox{0.95\textwidth}{\footnotesize " +
           rf"{a.R} replications, $T={a.T}$, {a.p} lags, {a.B} bootstrap draws in each replication. Cumulated "
           r"response to a unit innovation in the policy; bias relative to the true response, coverage of 90 "
           r"percent percentile bands, and mean width of the band. Data generating process and designs in the text.}",
           r"\end{table}"]
    open(os.path.join(a.out, "tab_mc_diff.tex"), "w").write("\n".join(tab) + "\n")
    print("written to", a.out)


if __name__ == "__main__":
    main()
