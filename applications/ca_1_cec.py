"""ca_1_cec.py: the LCFS cost per gallon reported by refiners to the California Energy Commission
(SB 1322; data/california/Estimated_Gasoline_Price_Breakdown_and_Margins_*.xlsx, from
https://www.energy.ca.gov/estimated-gasoline-price-breakdown-and-margins), and its validation
against the cost implied by the CARB formula.
 1. Parse the spreadsheet (one row per price component, one column per month; a header-row layout is
    also handled). Writes data/california/cec_lcfs_monthly.csv, used by the other California scripts.
    If the wrong row is picked, set LCFS_COL / DATE_COL below.
 2. Over the overlapping months: correlations in levels and changes, and OLS (HAC) of the CEC cost
    on the formula cost and on the constant-factor cost.
Output: results/california/1_cec_validation.csv
Usage: py ca_1_cec.py"""
from common import *
import glob
import re
import statsmodels.api as sm

LCFS_COL = None
DATE_COL = None
files = sorted(glob.glob(os.path.join(CA, "Estimated_Gasoline_Price_Breakdown*.xls*")))
if not files:
    sys.exit("Save the CEC spreadsheet in data/california/ first.")
print("CEC file:", os.path.basename(files[-1]))


def to_cents(v):
    v = pd.to_numeric(pd.Series(v).astype(str).str.replace(r"[$,¢\s]", "", regex=True), errors="coerce")
    return v * 100 if v.abs().median() < 2 else v


def find_dates(df, cols):
    low = {c: str(c).strip().lower() for c in cols}
    if DATE_COL is not None:
        return pd.to_datetime(df[DATE_COL], errors="coerce")
    yc = [c for c in cols if low[c] == "year"]
    mc = [c for c in cols if low[c] == "month"]
    if yc and mc:
        y = pd.to_numeric(df[yc[0]], errors="coerce")
        m = df[mc[0]].astype(str).str.strip()
        mnum = pd.to_numeric(m, errors="coerce")
        mname = pd.to_datetime(m.str[:3], format="%b", errors="coerce").dt.month
        mm = mnum.fillna(mname)
        return pd.to_datetime(pd.DataFrame({"year": y.values, "month": mm.values, "day": 1}), errors="coerce")
    for c in cols:
        if any(k in low[c] for k in ("date", "month", "period")):
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().mean() > 0.5:
                return d
    # first column that parses as dates
    for c in cols:
        d = pd.to_datetime(df[c], errors="coerce")
        if d.notna().mean() > 0.5:
            return d
    return None


def parse_transposed(raw):
    """Layout with one row per series and one column per month (the CEC file since 2026): the
    first column holds the labels, the row labelled 'Date' holds Excel serial dates (or the rows
    'Year' and 'Month')."""
    lab = raw.iloc[:, 0].astype(str).str.strip()
    hit = lab.str.contains("low carbon|lcfs", case=False)
    if not hit.any():
        return None
    vals = raw.iloc[:, 1:]
    drow = lab.str.lower() == "date"
    if drow.any():
        dv = vals[drow.values].iloc[0]
        num = pd.to_numeric(dv, errors="coerce")
        dates = (pd.to_datetime(num, unit="D", origin="1899-12-30") if num.notna().mean() > 0.5
                 else pd.to_datetime(dv, errors="coerce"))
    else:
        y = pd.to_numeric(vals[(lab.str.lower() == "year").values].iloc[0], errors="coerce")
        m = pd.to_numeric(vals[(lab.str.lower() == "month").values].iloc[0], errors="coerce")
        dates = pd.to_datetime(pd.DataFrame({"year": y.values, "month": m.values, "day": 1}), errors="coerce")
    lrow = vals[hit.values].iloc[0]
    crow = lab.str.contains("cap.{0,5}trade", case=False, regex=True)
    out = pd.DataFrame({"date": np.asarray(dates), "lcfs_cents": to_cents(lrow.values).values,
                        "cap_trade_cents": to_cents(vals[crow.values].iloc[0].values).values if crow.any() else np.nan})
    return out.dropna(subset=["date", "lcfs_cents"]), lab[hit].iloc[0], (lab[crow].iloc[0] if crow.any() else None)


best = None
book = pd.read_excel(files[-1], sheet_name=None, header=None)
for sh, raw in book.items():
    raw = raw.dropna(how="all").reset_index(drop=True)
    tr = parse_transposed(raw) if raw.shape[1] > 1 else None
    if tr is not None and len(tr[0]) > 0:
        if best is None or len(tr[0]) > len(best[0]):
            best = (tr[0], sh, tr[1], tr[2])
        continue
    raw = raw.dropna(how="all").reset_index(drop=True)
    for i in range(min(len(raw), 40)):
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains("low carbon|lcfs").any():
            df = raw.iloc[i + 1:].copy()
            df.columns = [str(c).strip() for c in raw.iloc[i]]
            cols = list(df.columns)
            lc = LCFS_COL if LCFS_COL in cols else next(c for c in cols if re.search("low carbon|lcfs", c, re.I))
            cc = next((c for c in cols if re.search("cap.{0,5}trade", c, re.I)), None)
            d = find_dates(df, cols)
            if d is None:
                continue
            s = pd.DataFrame({"date": d.values, "lcfs_cents": to_cents(df[lc]).values,
                              "cap_trade_cents": to_cents(df[cc]).values if cc else np.nan}).dropna(subset=["date", "lcfs_cents"])
            if best is None or len(s) > len(best[0]):
                best = (s, sh, lc, cc)
            break
if best is None:
    sys.exit("No LCFS column found: open the file, then set LCFS_COL and DATE_COL at the top of this script.")
cec, sheet, lc, cc = best
cec["date"] = pd.to_datetime(cec["date"]).dt.to_period("M").dt.to_timestamp()
cec = cec.groupby("date").mean().sort_index()
cec.to_csv(os.path.join(CA, "cec_lcfs_monthly.csv"))
print(f"parsed sheet '{sheet}', LCFS column '{lc}', cap-and-trade column '{cc}': "
      f"{cec.index[0]:%Y-%m} to {cec.index[-1]:%Y-%m}, {len(cec)} months")
print(cec.round(2).to_string())


D, _ = load_california(require_cec=False)
D = D.drop(columns=[c for c in ["cost_cec", "cost_spliced"] if c in D.columns])
D = D.join(cec["lcfs_cents"].rename("cost_cec"), how="left")
ov = D.dropna(subset=["cost_cec"])
val = []
for m in ["cost_tv", "cost_const"]:
    r = sm.OLS(ov.cost_cec.values, sm.add_constant(ov[m].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rd = sm.OLS(np.diff(ov.cost_cec.values), sm.add_constant(np.diff(ov[m].values))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    val.append(dict(measure=m, months=len(ov), corr_levels=np.corrcoef(ov.cost_cec, ov[m])[0, 1],
                    corr_changes=np.corrcoef(np.diff(ov.cost_cec), np.diff(ov[m]))[0, 1],
                    slope_levels=r.params[1], slope_levels_se=r.bse[1], intercept=r.params[0],
                    slope_changes=rd.params[1], slope_changes_se=rd.bse[1],
                    mean_cec=ov.cost_cec.mean(), mean_measure=ov[m].mean()))
    pre = ov[ov.index < "2025-01-01"]
    if len(pre) > 3:
        val[-1].update(months_2024=len(pre), mean_abs_gap_2024=float(np.mean(np.abs(pre.cost_cec - pre[m]))))
V = pd.DataFrame(val)
V.to_csv(os.path.join(outdir("california"), "1_cec_validation.csv"), index=False)
print("\nCEC cost against the credit-price measures (cents per gallon):\n", V.round(3).to_string(index=False))
print("\nmonth by month:\n", ov[["credit", "cost_tv", "cost_cec"]].round(2).to_string())

fac = pd.DataFrame([dict(year=y, CI_benchmark=CI_STD[y], CI_CARBOB=CI_CARBOB[y], t_per_gal=t_per_gal(y),
                         cents_per_gal_at_100_usd=100 * 100 * t_per_gal(y)) for y in sorted(CI_STD)])
fac.to_csv(os.path.join(outdir("california"), "0_formula_factor.csv"), index=False)
print("\nCARB formula used for the cost before the CEC series (check against 17 CCR 95484):\n",
      fac.round(5).to_string(index=False))
