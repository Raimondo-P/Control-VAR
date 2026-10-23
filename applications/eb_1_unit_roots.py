"""eb_1_unit_roots.py: ERS point-optimal test (constant, cbar = -7) with the sieve bootstrap on the
unemployment rate of every eligible treated state and every control.
Output: results/eb/1_unit_roots.csv
Usage: py eb_1_unit_roots.py [--B 999] [--workers N]"""
from common import *
from multiprocessing import Pool


def job(item):
    name, x, B = item
    t = cvar.design_test(x, np.zeros((len(x), 0)), None, q=0, B=B, seed=1)
    return dict(state=name, P_T=t.statistic, p_boot=t.pvalue_bootstrap, reject5=t.pvalue_bootstrap <= 0.05)


if __name__ == "__main__":
    a = args()
    UR, EB, EUC, elig = load_eb()
    items = [(s, UR[s].values, a.B) for s in sorted(set(elig) | set(CTRL))]
    with Pool(a.workers) as P:
        o = pd.DataFrame(P.map(job, items))
    o.to_csv(os.path.join(outdir("eb"), "1_unit_roots.csv"), index=False)
    print(o.round(3).to_string(index=False))
    print("share rejecting at 5%:", round(o.reject5.mean(), 3))
