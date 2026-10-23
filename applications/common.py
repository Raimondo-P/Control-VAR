"""common.py: paths, data loaders and the pre-specified rules shared by the EB and California scripts."""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cvar
from cvar.diagnostics import ljungbox_p

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "..", "results")          # the package-wide results folder, next to applications


def outdir(name):
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d


def args(defaultB=999, **extra):
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=defaultB, help="bootstrap draws")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    for k, v in extra.items():
        ap.add_argument(f"--{k}", **v)
    return ap.parse_args()


# ------------------------------------------------------------------ calendar adjustment
def sa_changes(x, groups, mask=None):
    """Remove calendar effects from the changes of a level series (for each array in `groups`, the
    group means of the changes minus the overall mean), estimated where `mask` is True; re-cumulate."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    keep = np.ones(len(dx), bool) if mask is None else np.asarray(mask, bool)[1:]
    adj = np.zeros_like(dx)
    for g in groups:
        g = np.asarray(g)[1:]
        s = pd.Series(dx - adj)
        eff = s[keep].groupby(g[keep]).mean() - s[keep].mean()
        adj = adj + pd.Series(g).map(eff).fillna(0.0).values
    return np.r_[x[0], x[0] + np.cumsum(dx - adj)]


# ------------------------------------------------------------------ lag rule
def lag_rule(Z, exog, cols, pmax, m=12, level=0.05):
    """Smallest order in 1..pmax with no Ljung-Box rejection (m lags, `level`) in the equations
    `cols` of a VAR in Z; pmax if no order passes."""
    for p in range(1, pmax + 1):
        _, U = cvar.var_fit(Z, p, exog)
        if all(ljungbox_p(U[:, i], m) > level for i in cols):
            return p
    return pmax


# ------------------------------------------------------------------ extended benefits
CTRL = ["SD", "ND", "IA", "NE", "WY", "UT", "OK"]          # never triggered EB, 2002-2019
NB = {"SD": ["ND", "MN", "IA", "NE", "WY", "MT"], "ND": ["MN", "SD", "MT"],
      "IA": ["MN", "WI", "IL", "MO", "NE", "SD"], "NE": ["SD", "IA", "MO", "KS", "CO", "WY"],
      "WY": ["MT", "SD", "NE", "CO", "UT", "ID"], "UT": ["ID", "WY", "CO", "AZ", "NV"],
      "OK": ["KS", "MO", "AR", "TX", "NM", "CO"]}            # neighbours of each control


def load_eb():
    Pn = pd.read_csv(os.path.join(DATA, "eb", "eb_euc_panel.csv"), parse_dates=["date"])
    UR = Pn.pivot(index="date", columns="st", values="ur")
    EB = Pn.pivot(index="date", columns="st", values="eb_weeks") / 10     # unit: ten weeks
    EUC = Pn.pivot(index="date", columns="st", values="euc_weeks") / 10
    months = (EB > 0).sum()
    elig = [s for s in months.index if months[s] >= 12 and s not in CTRL]
    return UR, EB, EUC, elig


def controls_of(s):
    """Never-triggered controls that do not border state s."""
    return [c for c in CTRL if s not in NB[c]]


def fit_eb_diff(UR, EB, EUC, s, p):
    """Differenced branch: equal imposed weights on the controls, cycle in changes, EUC ordered first."""
    cc = controls_of(s)
    return cvar.CVAR(p=p).fit(UR[s].values, UR[cc].values, EB[s].values, other=EUC[s].values,
                              names=[s], control_names=cc, weights=np.ones(len(cc)) / len(cc),
                              difference=True)


# ------------------------------------------------------------------ California
CA = os.path.join(DATA, "california")
CI_STD = {2014: 97.96, 2015: 97.96, 2016: 96.50, 2017: 95.02, 2018: 93.55, 2019: 93.23, 2020: 91.98,
          2021: 90.74, 2022: 89.50, 2023: 88.25, 2024: 87.01, 2025: 85.77, 2026: 84.52}
CI_CARBOB = {y: (99.78 if y <= 2018 else 100.82) for y in CI_STD}
ED_CARBOB = 119.53            # MJ per gallon of CARBOB
SHARE_CARBOB = 0.90           # E10
CONST_T_PER_GAL = 0.0018
SETS = {"gulf": (["gas_gulf"], "Gulf Coast (PADD 3)"),
        "padd1_4": (["gas_r10", "gas_r20", "gas_gulf", "gas_r40"], "PADDs 1 to 4"),
        "tx_fl": (["gas_tx", "gas_fl"], "Texas and Florida")}
REGION = {"gas_r10": "East Coast", "gas_r20": "Midwest", "gas_gulf": "Gulf Coast", "gas_r40": "Rocky Mountain"}


def t_per_gal(year):
    """LCFS deficit per retail gallon of E10 implied by CARB's formula with the year's gasoline benchmark."""
    y = min(max(year, 2014), 2026)
    return max(CI_CARBOB[y] - CI_STD[y], 0.0) * ED_CARBOB * SHARE_CARBOB / 1e6


def gas_monthly(name):
    d = pd.read_csv(os.path.join(CA, "ca_raw", f"{name}.csv"))
    s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0]))
    return s.dropna().resample("MS").mean()


def lcfs_credit():
    x = pd.read_excel(os.path.join(CA, "lcfs_credit_transfers.xlsx"), sheet_name="Monthly")
    mth = x["month"]
    mth = (pd.to_datetime(mth, unit="D", origin="1899-12-30") if np.issubdtype(mth.dtype, np.number)
           else pd.to_datetime(mth))
    s = pd.Series(pd.to_numeric(x["avg_price_usd_per_credit"], errors="coerce").values,
                  index=mth.dt.to_period("M").dt.to_timestamp()).groupby(level=0).last()
    return s.resample("MS").mean().interpolate()


def load_california(require_cec=True):
    """Monthly panel: retail gasoline prices (USD/gal), credit price, and the LCFS cost per gallon in
    cents: cost_const (constant factor), cost_tv (CARB formula), cost_cec (refiners, CEC) and
    cost_spliced (formula before the first CEC month, CEC from then on). Returns (D, first CEC month)."""
    GAS = ["gas_ca", "gas_gulf", "gas_r10", "gas_r20", "gas_r40", "gas_tx", "gas_fl"]
    D = pd.concat([pd.DataFrame({n: gas_monthly(n) for n in GAS}), lcfs_credit().rename("credit")], axis=1).dropna()
    D["cost_const"] = 100 * CONST_T_PER_GAL * D.credit
    D["cost_tv"] = 100 * D.credit * np.array([t_per_gal(t.year) for t in D.index])
    f = os.path.join(CA, "cec_lcfs_monthly.csv")
    if not os.path.exists(f):
        if require_cec:
            sys.exit("data/california/cec_lcfs_monthly.csv not found: run ca_1_cec.py first")
        return D, None
    cec = pd.read_csv(f, parse_dates=["date"]).set_index("date")["lcfs_cents"]
    first = cec.index.min()
    D = D.join(cec.rename("cost_cec"), how="left")
    D["cost_spliced"] = np.where(D.index < first, D.cost_tv, D.cost_cec)
    return D, first


def ca_series(df, cc, measure, first_cec=None):
    """Seasonally adjusted California and control prices (cents) and the monthly change of the cost
    measure, on the months where the measure exists; the change at the splice month is set to zero."""
    df = df[df[f"cost_{measure}"].notna()]
    mon = df.index.month
    y = sa_changes(100 * df.gas_ca.values, [mon])
    C = np.column_stack([sa_changes(100 * df[c].values, [mon]) for c in cc])
    lev = df[f"cost_{measure}"].values
    dose = np.r_[0.0, np.diff(lev)]
    if measure == "spliced" and first_cec is not None and first_cec in df.index and df.index.get_loc(first_cec) > 0:
        dose[df.index.get_loc(first_cec)] = 0.0
    return y, C, dose, lev, df.index
