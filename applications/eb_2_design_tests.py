"""eb_2_design_tests.py: test of the design (Section 5) for every eligible EB state.
(a) estimated weights (never-triggered controls minus neighbours), policy terms EUC and EB with
    q in {0, 3, 6, 12}, with and without automatic level shifts;
(b) imposed equal weights: the known-weights statistic on Y1 - mean(controls), with q = 12 and
    without policy terms.
Outputs: results/eb/2a_estimated_weights.csv, 2b_imposed_weights.csv
Usage: py eb_2_design_tests.py [--B 999] [--workers N]"""
from common import *
from multiprocessing import Pool

UR, EB, EUC, elig = load_eb()


def job_a(item):
    s, q, brk, B = item
    cc = controls_of(s)
    r = cvar.CVAR(q=q, breaks=brk).fit(UR[s].values, UR[cc].values, EB[s].values, other=EUC[s].values,
                                       names=[s], control_names=cc)
    t = r.design_test(B=B, seed=7)[0]
    return dict(st=s, q=q, shifts="auto" if brk else "none", n_shifts=len(r.units[0]["breaks"]),
                K=len(cc), P_T=t.statistic, p_boot=t.pvalue_bootstrap)


def job_b(item):
    s, B = item
    cc = controls_of(s)
    g = UR[s].values - UR[cc].values.mean(1)
    pol = np.column_stack([EUC[s].values, EB[s].values])
    t = cvar.design_test(g, np.zeros((len(g), 0)), pol, q=12, B=B, seed=5)
    t0 = cvar.design_test(g, np.zeros((len(g), 0)), None, q=0, B=B, seed=5)
    return dict(st=s, P_T=t.statistic, p_boot=t.pvalue_bootstrap, P_T_nopolicy=t0.statistic,
                p_boot_nopolicy=t0.pvalue_bootstrap)


if __name__ == "__main__":
    a = args()
    o = outdir("eb")
    with Pool(a.workers) as P:
        oa = pd.DataFrame(P.map(job_a, [(s, q, b, a.B) for s in elig for q in (12, 6, 3, 0) for b in (None, "auto")]))
        ob = pd.DataFrame(P.map(job_b, [(s, a.B) for s in elig]))
    oa.to_csv(os.path.join(o, "2a_estimated_weights.csv"), index=False)
    ob.to_csv(os.path.join(o, "2b_imposed_weights.csv"), index=False)
    print(oa.pivot_table(index="st", columns=["shifts", "q"], values="p_boot").round(3).to_string())
    print("\nRejections at 5% by specification:\n", oa.assign(r=oa.p_boot <= .05).groupby(["shifts", "q"]).r.sum())
    print("\nImposed equal weights:\n", ob.round(3).to_string(index=False))
