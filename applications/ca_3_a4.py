"""ca_3_a4.py: does the LCFS cost react to the California-specific price cycle (Assumption 4)?
For each control set, policy measure (spliced, CARB formula), sample (full, 2017 onward) and lag
order (2, 6, 12):
  granger_cycle   : change in the cost on its own lags and lags of the change in the cycle
                    (California minus the average control); HAC Wald test on the cycle lags
  granger_gap     : the same with the lagged level of the gap; HAC Wald test, and the t-test on the gap
  var_causality   : Wald test that the cycle Granger-causes the policy in the VAR (policy, change in
                    the cycle) of the CVAR; this is the test to report (the HAC tests over-reject with
                    many restrictions in samples of this size)
  granger_controls: context, the policy on lags of the controls' own price changes
Requires ca_1_cec.py. Output: results/california/G_a4_tests.csv
Usage: py ca_3_a4.py"""
from common import *
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

OUT = outdir("california")
D, FIRST_CEC = load_california()


def lags(v, p):
    return np.column_stack([pd.Series(v).shift(j).values for j in range(1, p + 1)])


def wald(res, idx):
    R = np.zeros((len(idx), len(res.params)))
    for k, i in enumerate(idx):
        R[k, i] = 1.0
    w = res.wald_test(R, scalar=True)
    return float(w.statistic), float(w.pvalue)


rows = []
for cs, cc in {k: v[0] for k, v in SETS.items()}.items():
    for meas in ["spliced", "tv"]:
        for lab, start in [("full", None), ("2017_end", "2017-01-01")]:
            Ds = D.loc[start:]
            Ds = Ds[Ds[f"cost_{meas}"].notna()]
            mon = Ds.index.month
            y = sa_changes(100 * Ds.gas_ca.values, [mon])
            cbar = np.mean([sa_changes(100 * Ds[c].values, [mon]) for c in cc], axis=0)
            gap = y - cbar
            dg = np.r_[np.nan, np.diff(gap)]
            dc = np.r_[np.nan, np.diff(cbar)]
            dose = np.r_[np.nan, np.diff(Ds[f"cost_{meas}"].values)]
            if meas == "spliced" and FIRST_CEC in Ds.index and Ds.index.get_loc(FIRST_CEC) > 0:
                dose[Ds.index.get_loc(FIRST_CEC)] = 0.0
            for p in [2, 6, 12]:
                base = [lags(dose, p)]
                hac = dict(cov_type="HAC", cov_kwds={"maxlags": max(p, 3)})
                # 1. cycle lags
                X = np.column_stack(base + [lags(dg, p)])
                ok = ~np.isnan(dose) & ~np.isnan(X).any(1)
                r1 = sm.OLS(dose[ok], sm.add_constant(X[ok])).fit(**hac)
                s1, p1 = wald(r1, list(range(1 + p, 1 + 2 * p)))
                # 2. cycle lags and lagged gap level
                X2 = np.column_stack(base + [lags(dg, p), pd.Series(gap).shift(1).values])
                ok2 = ~np.isnan(dose) & ~np.isnan(X2).any(1)
                r2 = sm.OLS(dose[ok2], sm.add_constant(X2[ok2])).fit(**hac)
                s2, p2 = wald(r2, list(range(1 + p, 2 + 2 * p)))
                tgap, pgap = float(r2.tvalues[-1]), float(r2.pvalues[-1])
                # 3. VAR Granger causality in (policy, change in cycle)
                Z = np.column_stack([dose, dg])[1:]
                try:
                    vr = VAR(Z).fit(p, trend="c")
                    tc = vr.test_causality(caused=0, causing=[1], kind="wald")
                    s3, p3 = float(tc.test_statistic), float(tc.pvalue)
                except Exception as e:
                    s3, p3 = np.nan, np.nan
                # 4. controls' own changes (context)
                X4 = np.column_stack(base + [lags(dc, p)])
                ok4 = ~np.isnan(dose) & ~np.isnan(X4).any(1)
                r4 = sm.OLS(dose[ok4], sm.add_constant(X4[ok4])).fit(**hac)
                s4, p4 = wald(r4, list(range(1 + p, 1 + 2 * p)))
                rows.append(dict(controls=cs, measure=meas, sample=lab, p=p, n=int(ok.sum()),
                                 granger_cycle_stat=s1, granger_cycle_p=p1,
                                 granger_gap_stat=s2, granger_gap_p=p2, gap_t=tgap, gap_p=pgap,
                                 var_causality_stat=s3, var_causality_p=p3,
                                 granger_controls_stat=s4, granger_controls_p=p4))
                print(f"{cs:8s} {meas:5s} {lab:9s} p={p:2d}: cycle p {p1:.3f} | cycle+gap p {p2:.3f} "
                      f"(gap t {tgap:+.2f}) | VAR p {p3:.3f} | controls p {p4:.3f}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "G_a4_tests.csv"), index=False)
print("\nShare of tests rejecting at 5 percent:")
for c in ["granger_cycle_p", "granger_gap_p", "var_causality_p", "granger_controls_p"]:
    print(f"  {c}: {(R[c] <= 0.05).mean():.2f} of {R[c].notna().sum()}")
print("\nwritten to", OUT)
