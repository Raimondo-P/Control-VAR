"""ak_5_donor_pool.py: weights estimated in changes with larger donor pools (no instruments).

Least-squares weights in changes are shrunk by the idiosyncratic noise of the donors (the time-series counterpart of
the imperfect-fit bias of Ferman and Pinto 2021). The shrinking vanishes as the number of donors grows, because the
noise averages out (Ferman 2021). This script compares three pools:
  jm_emp   the six donors of Jones and Marinescu (2022): UT, WY, WA, NV, MT, MN (current main specification)
  west     the other states of the Census West region: WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, NM, HI
  all      every other state and DC (50 donors)
For each outcome, sample and pool it reports:
  weights    non-negative least squares in changes, with a constant and the current value and 12 lags of the dividend
             and of the change in the oil price (as in ak_1_application.py): number of positive weights, sum,
             largest weights
  factors    principal components of the changes of all 50 donors (after the same regressors are partialled out):
             share of their variance explained by the first three components, the loading of Alaska on each component
             and the loading of the weighted donors. The ratio (weighted donors / Alaska) is the share of each common
             movement of Alaska that the weighted donors reproduce: one means no shrinking, below one means the
             weights leave part of that movement in the cycle (Theorem thm:diff, part (c)).
  responses  CVAR point estimates at 3, 12 and 24 months (no bands) with each pool's weights, and with equal
             weights on the six donors.

Data: monthly, seasonally adjusted LAUS and CES series for all states from the BLS API (data/alaska/raw/bls_all_*.csv).
That is 306 series. With a free BLS API key in the environment variable BLS_API_KEY the download takes about 20
requests. Without a key the API allows 25 requests a day: the script saves each completed chunk, so running it again
on the next day continues where it stopped.
Requires ak_0_screen.py and ak_1_application.py in the same folder. Output: results/alaska/5_donor_pool.csv,
results/alaska/5_weights.csv
Usage: py ak_5_donor_pool.py [--samples pre2020 pre2016] [--outcomes epop nonfarm]
"""
from common import *
import glob
import ak_0_screen as s0
from ak_1_application import design, w_nnls, fit, col, lags, Q

FIPS_ALL = {"AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11",
            "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
            "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30",
            "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
            "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
            "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"}
DONORS_ALL = [s for s in FIPS_ALL if s != "AK"]
POOLS = {"jm_emp": s0.SETS_AK["jm_emp"],
         "west": ["WA", "OR", "CA", "ID", "NV", "UT", "AZ", "MT", "WY", "CO", "NM", "HI"],
         "all": DONORS_ALL}
LAUS_M = {"epop": "07", "ur": "03", "lfpr": "08"}
CES_I = {"nonfarm": "00000000", "leisure": "70000000", "retail": "42000000"}
OUTC = ["epop", "ur", "lfpr", "nonfarm", "leisure", "retail"]
SAMPLES = {"pre2020": "2019-12-01", "pre2016": "2015-12-01"}
P = 12
H = 36
NPC = 3


def laus_id(st, m):
    return f"LASST{FIPS_ALL[st]}00000000000{m}"


def ces_id(st, ind):
    return f"SMS{FIPS_ALL[st]}00000{ind}01"


def download_all():
    ids = [laus_id(s, m) for s in FIPS_ALL for m in LAUS_M.values()] + [ces_id(s, i) for s in FIPS_ALL for i in CES_I.values()]
    per = 50 if os.environ.get("BLS_API_KEY") else 25
    chunks = [ids[i:i + per] for i in range(0, len(ids), per)]
    os.makedirs(s0.RAW, exist_ok=True)
    for k, ch in enumerate(chunks):
        f = os.path.join(s0.RAW, f"bls_all_{per}_{k:02d}.csv")
        if os.path.exists(f):
            continue
        print(f"downloading chunk {k + 1} of {len(chunks)} ({len(ch)} series)", flush=True)
        try:
            d = s0.bls_fetch(ch, 1976, pd.Timestamp.today().year)
        except Exception as e:
            print(f"  chunk {k + 1} failed ({type(e).__name__}: {e}); run the script again later "
                  f"(without a key the BLS allows 25 requests a day)", flush=True)
            return False
        if d.shape[1] == 0:
            print(f"  chunk {k + 1} returned no data; run the script again later", flush=True)
            return False
        last = pd.Timestamp.today().year - 1
        missing_years = sorted(set(range(1976, last + 1)) - set(d.index.year))
        if missing_years:
            print(f"  chunk {k + 1} is incomplete (no data for {missing_years[0]}-{missing_years[-1]}): not saved; "
                  f"run the script again later", flush=True)
            return False
        d.to_csv(f)
    return True


def load_all():
    files = sorted(glob.glob(os.path.join(s0.RAW, "bls_all_*.csv")))
    B = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in files], axis=1)
    # if a series appears in more than one file, keep the copy with the most observations
    B = B.T.assign(_n=B.notna().sum().values).sort_values("_n", ascending=False).drop(columns="_n")
    B = B[~B.index.duplicated(keep="first")].T.sort_index()
    B.index = pd.to_datetime(B.index)

    def c(i):
        return B[i] if i in B.columns else pd.Series(np.nan, index=B.index)
    out = {}
    for nm, m in LAUS_M.items():
        out[nm] = s0.fill_gaps(pd.DataFrame({s: c(laus_id(s, m)) for s in FIPS_ALL}), nm)
    for nm, ind in CES_I.items():
        out[nm] = 100 * np.log(pd.DataFrame({s: c(ces_id(s, ind)) for s in FIPS_ALL}))
    for nm, df in out.items():
        miss = [s for s in FIPS_ALL if df[s].notna().sum() == 0]
        if miss:
            print(f"  {nm}: no data for {miss} (left out of the pools)", flush=True)
    return out


def complete(out, o, donors, end):
    """Donors with no missing month over the window where Alaska is observed."""
    Y = out[o].loc[:end]
    ak = Y["AK"].dropna()
    if not len(ak):
        return []
    win = Y.loc[ak.index[0]:ak.index[-1]]
    return [s for s in donors if s in win and win[s].notna().all()]


def partial_out(df, cols):
    """Changes of the columns with a constant and lags 0..Q of (oil, dividend) projected out."""
    pol = np.column_stack([df["doil"].values, df["W"].values])[1:]
    F = np.hstack([np.ones((len(df) - 1, 1)), lags(pol, range(0, Q + 1))])
    D = np.diff(df[cols].values, axis=0)
    ok = ~np.isnan(F).any(1)
    F, D = F[ok], D[ok]
    return D - F @ (np.linalg.pinv(F) @ D)


if __name__ == "__main__":
    a = args(samples=dict(nargs="*", default=list(SAMPLES)), outcomes=dict(nargs="*", default=OUTC))
    OUT = outdir("alaska")
    if not download_all():
        sys.exit("download incomplete: the chunks already saved are kept; run again to continue")
    out = load_all()
    B_, F_ = s0.load_raw(False)
    _, W, doil = s0.build(B_, F_)
    rows, wrows = [], []
    for o in a.outcomes:
        for smp in a.samples:
            end = SAMPLES[smp]
            pool_all = complete(out, o, DONORS_ALL, end)
            if len(pool_all) < NPC + 1:
                print(f"  skipped {o} {smp}: too few donors with complete data", flush=True)
                continue
            dfa = s0.arrays(out, W, doil, o, pool_all, end)
            Xa = partial_out(dfa, pool_all)
            ya = partial_out(dfa, ["y"])[:, 0]
            Z = (Xa - Xa.mean(0)) / Xa.std(0)
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            share = S ** 2 / (S ** 2).sum()
            PC = U[:, :NPC] * S[:NPC]
            lam_ak = np.linalg.lstsq(PC, ya, rcond=None)[0]
            lam_all = np.linalg.lstsq(PC, Xa, rcond=None)[0]          # NPC x K (loadings of each donor)
            idx = {s: i for i, s in enumerate(pool_all)}
            for pn, pool in POOLS.items():
                cc = [s for s in pool if s in idx]
                if len(cc) < 2:
                    continue
                df = s0.arrays(out, W, doil, o, cc, end)
                dy, X, F = design(df, cc)
                w = w_nnls(dy, X, F)
                lam_w = lam_all[:, [idx[s] for s in cc]] @ w
                row = dict(outcome=o, sample=smp, pool=pn, K=len(cc), T=len(df), start=f"{df.index[0]:%Y-%m}",
                           end=f"{df.index[-1]:%Y-%m}", n_positive=int((w > 0.005).sum()), sum_w=float(w.sum()),
                           max_w=float(w.max()),
                           top=", ".join(f"{s} {v:.2f}" for s, v in sorted(zip(cc, w), key=lambda z: -z[1])[:6] if v > 0.005))
                for j in range(NPC):
                    row[f"share_pc{j + 1}"] = float(share[j])
                    row[f"lam_ak_pc{j + 1}"] = float(lam_ak[j])
                    row[f"lam_w_pc{j + 1}"] = float(lam_w[j])
                    row[f"ratio_pc{j + 1}"] = float(lam_w[j] / lam_ak[j]) if abs(lam_ak[j]) > 1e-8 else np.nan
                try:
                    c = col(fit(df, cc, w, P).irf(H)["cycle"])
                    for h in (3, 12, 24):
                        row[f"cvar_h{h}"] = c[h]
                    if pn == "jm_emp":
                        ce = col(fit(df, cc, np.ones(len(cc)) / len(cc), P).irf(H)["cycle"])
                        for h in (3, 12, 24):
                            row[f"equal_h{h}"] = ce[h]
                except Exception as e:
                    print(f"  CVAR skipped for {o} {smp} {pn}: {e}", flush=True)
                rows.append(row)
                wrows += [dict(outcome=o, sample=smp, pool=pn, state=s, weight=float(v)) for s, v in zip(cc, w)]
            print(f"  done {o} {smp}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "5_donor_pool.csv"), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(OUT, "5_weights.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\nWeights by donor pool")
    print(R[["outcome", "sample", "pool", "K", "n_positive", "sum_w", "max_w", "top"]].round(3).to_string(index=False))
    print("\nCommon movements of the 50 donors: variance share of each component; loading of Alaska; share of it that the"
          " weighted donors reproduce (ratio, 1 = none left in the cycle)")
    cols = ["outcome", "sample", "pool"] + [f"share_pc{j}" for j in (1, 2, 3)] + [f"lam_ak_pc{j}" for j in (1, 2, 3)] \
        + [f"ratio_pc{j}" for j in (1, 2, 3)]
    print(R[cols].round(3).to_string(index=False))
    print("\nCVAR point estimates (no bands)")
    cols = ["outcome", "sample", "pool"] + [f"cvar_h{h}" for h in (3, 12, 24)] + [f"equal_h{h}" for h in (3, 12, 24)]
    print(R[[c for c in cols if c in R]].round(3).to_string(index=False))
    print("\nwritten to", OUT)
