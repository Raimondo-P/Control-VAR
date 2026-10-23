import os
"""
download_california.py
----------------------
Run on YOUR computer (the cloud workspace cannot reach EIA / CARB).
    pip install pandas requests xlrd openpyxl
    py download_california.py                 # weekly gasoline + crude from EIA
    py download_california.py build           # after you saved the policy file(s): builds data_*.csv for screen_pair.py

Part 1 (automatic): EIA weekly retail gasoline, all grades, regular, $/gal, by region, plus WTI/Brent.
  Output: ./ca_raw/<name>.csv  with columns date,<name>
  Each URL is tried; failures are printed, not hidden. If the xls route fails, set EIA_API_KEY
  (free: https://www.eia.gov/opendata/register.php) and the API v2 fallback is used.

Part 2 (MANUAL, two small files, because CARB has no stable download URL). Save into ./ca_raw/ :
  (a) cap_trade_auction.csv   columns: date,price     (USD per allowance, auction settlement price, quarterly)
      Source: CARB "California-Quebec Joint Auction Results" summary (ww2.arb.ca.gov/our-work/programs/cap-and-trade-program/auction-information);
      or ICAP allowance price explorer. Date = auction date. Joint auctions since Nov 2012.
  (b) lcfs_credit.csv         columns: date,price     (USD per credit, MONTHLY average transfer price)
      Source: CARB "LCFS Credit Transfer Activity Reports" (monthly average price), ww2.arb.ca.gov/resources/documents/lcfs-credit-transfer-activity-reports
      Available from 2011 (programme started Jan 2011; fuels were covered from the start).
  Optional (c) cap_trade_secondary.csv  (same layout, if you have monthly secondary-market prices; better than quarterly auctions).

Part 3 (build): converts the policy into USD per gallon and writes data files:
  data_ca_us.csv        y = California, c = US excluding California (R5XCA proxy: West Coast ex-CA = R5XCA; US = NUS)
  data_ca_gulf.csv      y = California, c = Gulf Coast (R30)
  data_ca_r5xca.csv     y = California, c = West Coast excl. California
  w = policy cost per gallon, weekly (step function carried forward from the last observation).
  Conversion factors (EDIT at the top of the file if you disagree):
     cap-and-trade: allowance price (USD/t) x 0.008887 t CO2/gal (EPA, gasoline); applied from 2015-01-01, when fuels entered the cap.
     LCFS: credit price (USD/t) x LCFS_T_PER_GAL. The per-gallon pass-through is uncertain; the default 0.0018 t/gal is a
           rough placeholder (about a 10% carbon-intensity cut on ~0.009 t/gal, scaled by the energy-economy ratio). Check it against
           CARB's own cost-per-gallon estimates. The factor only rescales w, so it does not change the screening tests, only the units.
  Screening run:  py screen_pair.py data_ca_gulf_combined.csv --freq W --boot 299
"""
import io, os, sys, time
import pandas as pd
import requests

OUT = "ca_raw"; os.makedirs(OUT, exist_ok=True)
START = "2000-01-01"
KEY = os.environ.get("EIA_API_KEY", "")   # set EIA_API_KEY (free key from eia.gov/opendata)

CAP_T_PER_GAL = 0.008887
LCFS_T_PER_GAL = 0.0018        # <- edit (see header). Effective credits needed per gallon of gasoline, tonnes CO2e
CAP_START = pd.Timestamp("2015-01-01")

# EIA series ids: weekly retail gasoline, regular, all formulations
GAS = {  # name -> (xls area code, api series id)
    "gas_ca":    ("SCA",   "EMM_EPMRU_PTE_SCA_DPG"),
    "gas_us":    ("NUS",   "EMM_EPMRU_PTE_NUS_DPG"),
    "gas_r5xca": ("R5XCA", "EMM_EPMRU_PTE_R5XCA_DPG"),   # PADD5 excluding California
    "gas_gulf":  ("R30",   "EMM_EPMRU_PTE_R30_DPG"),
    "gas_r10":   ("R10",   "EMM_EPMRU_PTE_R10_DPG"),
    "gas_r20":   ("R20",   "EMM_EPMRU_PTE_R20_DPG"),
    "gas_r40":   ("R40",   "EMM_EPMRU_PTE_R40_DPG"),
    "gas_tx":    ("STX",   "EMM_EPMRU_PTE_STX_DPG"),
    "gas_fl":    ("SFL",   "EMM_EPMRU_PTE_SFL_DPG"),
}
CRUDE = {"wti": ("RWTC", "PET_RWTC_W"), "brent": ("RBRTE", "PET_RBRTE_W")}


def xls(url, sheet="Data 1"):
    r = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"}); r.raise_for_status()
    d = pd.read_excel(io.BytesIO(r.content), sheet_name=sheet, skiprows=2)
    d.columns = ["date", "v"] + list(d.columns[2:]); d = d[["date", "v"]]
    d["date"] = pd.to_datetime(d["date"]); d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna()


def _frame(j):
    d = pd.DataFrame(j)[["period", "value"]].rename(columns={"period": "date", "value": "v"})
    d["date"] = pd.to_datetime(d["date"]); d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna().sort_values("date")


def api(sid, area=None, kind="gas"):
    """EIA API v2. Tries several series-id spellings, then the facet route (duoarea/product/process)."""
    if not KEY: raise RuntimeError("no EIA_API_KEY")
    errs = []
    ids = [sid]
    if kind == "gas" and area:      # California etc. are 'EPMR' (regular, all formulations); some areas 'EPMRU'
        ids = [f"EMM_EPMRU_PTE_{area}_DPG", f"EMM_EPMR_PTE_{area}_DPG", f"EMM_EPMRR_PTE_{area}_DPG"]
    for i in ids:
        try:
            r = requests.get(f"https://api.eia.gov/v2/seriesid/PET.{i}.W", params={"api_key": KEY}, timeout=90)
            r.raise_for_status()
            return _frame(r.json()["response"]["data"])
        except Exception as e:
            errs.append(f"{i}:{str(e)[:25]}")
    if kind == "gas" and area:
        for prod in ["EPMR", "EPMRU", "EPM0"]:
            try:
                r = requests.get("https://api.eia.gov/v2/petroleum/pri/gnd/data/",
                                 params={"api_key": KEY, "frequency": "weekly", "data[0]": "value",
                                         "facets[duoarea][]": area, "facets[product][]": prod, "facets[process][]": "PTE",
                                         "length": 5000, "sort[0][column]": "period", "sort[0][direction]": "asc"}, timeout=90)
                r.raise_for_status()
                j = r.json()["response"]["data"]
                if j: return _frame(j)
            except Exception as e:
                errs.append(f"facet {prod}:{str(e)[:25]}")
    raise RuntimeError("; ".join(errs))


def get(name, area, sid, kind):
    if kind == "gas":
        urls = [f"https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_{area}_DPGw.xls"]
        # some areas use EPMRU (all formulations) in the file name
        urls.append(f"https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMRU_PTE_{area}_DPGw.xls")
    else:
        urls = [f"https://www.eia.gov/dnav/pet/hist_xls/{area}w.xls"]
    err = ""
    for u in urls:
        try:
            return xls(u), u
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:70]}"
    try:
        return api(sid, area, kind), "EIA API v2"
    except Exception as e:
        raise RuntimeError(err + " | " + str(e)[:200])


def download():
    rep = []
    for name, (area, sid) in {**GAS, **{k: v for k, v in CRUDE.items()}}.items():
        kind = "gas" if name.startswith("gas") else "crude"
        try:
            d, src = get(name, area, sid, kind)
            d = d[d.date >= START]
            d.rename(columns={"v": name}).to_csv(f"{OUT}/{name}.csv", index=False)
            print(f"OK      {name:10s} {d.date.min().date()} -> {d.date.max().date()} n={len(d)}  {src}")
            rep.append((name, "OK"))
        except Exception as e:
            print(f"FAILED  {name:10s} {e}")
            rep.append((name, "FAILED"))
        time.sleep(0.4)
    print("\nIf any gas series FAILED: open https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm, "
          "choose the region, 'Download Data (XLS)', and save the 'Data 1' sheet as CSV `date,<name>` into ca_raw/.")
    print("Now save cap_trade_auction.csv and lcfs_credit.csv (see header) and run:  py download_california.py build")


def weekly_step(df, idx):
    s = df.set_index("date")["price"].sort_index()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def build():
    def rd(n):
        return pd.read_csv(f"{OUT}/{n}.csv", parse_dates=["date"]).set_index("date").iloc[:, 0]
    ca = rd("gas_ca")
    pol = {}
    for f in ["cap_trade_auction", "cap_trade_secondary", "lcfs_credit"]:
        p = f"{OUT}/{f}.csv"
        if os.path.exists(p):
            pol[f] = pd.read_csv(p, parse_dates=["date"])
            print("found", f, len(pol[f]), "rows", pol[f].date.min().date(), "->", pol[f].date.max().date())
    if not pol:
        sys.exit("No policy file found in ca_raw/.")
    idx = ca.index
    w = {}
    cap_key = "cap_trade_secondary" if "cap_trade_secondary" in pol else ("cap_trade_auction" if "cap_trade_auction" in pol else None)
    if cap_key:
        c_ = weekly_step(pol[cap_key], idx) * CAP_T_PER_GAL
        c_[c_.index < CAP_START] = 0.0           # fuels covered only from 2015
        w["captrade"] = c_
    if "lcfs_credit" in pol:
        w["lcfs"] = weekly_step(pol["lcfs_credit"], idx) * LCFS_T_PER_GAL
    W = pd.DataFrame(w)
    W["combined"] = W.sum(axis=1, min_count=1)
    W.to_csv("policy_weekly.csv")
    for tag, cn in [("us", "gas_us"), ("r5xca", "gas_r5xca"), ("gulf", "gas_gulf")]:
        if not os.path.exists(f"{OUT}/{cn}.csv"): print("skip", tag); continue
        for pn in W.columns:
            d = pd.concat([ca.rename("y"), rd(cn).rename("c"), W[pn].rename("w")], axis=1).dropna()
            d.index.name = "date"
            d.to_csv(f"data_ca_{tag}_{pn}.csv")
            print(f"wrote data_ca_{tag}_{pn}.csv  n={len(d)}  {d.index.min().date()} -> {d.index.max().date()}")
    print("\nNote: U.S. average includes California (~10% weight); R5XCA is PADD 5 ex-California and shares West Coast refining, "
          "so it is the likelier spillover victim; Gulf Coast is the cleanest control.")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else download()
