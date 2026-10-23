"""eb_5_average.py: average response across the EB states with the joint bootstrap (the same time
periods drawn for every state), one common lag order for all states and both models (the larger of
the median rule orders of the CVAR and of the VAR). Requires eb_3_lags.py.
Outputs: results/eb/5_average.csv, 5_states_common_lag.csv
Usage: py eb_5_average.py [--B 999]"""
from common import *

if __name__ == "__main__":
    a = args()
    UR, EB, EUC, elig = load_eb()
    L = pd.read_csv(os.path.join(outdir("eb"), "3_lags.csv")).set_index("st")
    sts = [s for s in elig if s in L.index]
    p = int(max(L.loc[sts].p_cvar.median(), L.loc[sts].p_var.median()))
    print("common lag order:", p)
    rs = [fit_eb_diff(UR, EB, EUC, s, p) for s in sts]
    per, avg = cvar.joint_bootstrap(rs, B=a.B, H=36, seed=21)
    ta = avg.table(horizons=(0, 6, 12, 24, 36))
    ta["p"] = p
    ta.to_csv(os.path.join(outdir("eb"), "5_average.csv"), index=False)
    pd.concat([b.table(horizons=(0, 6, 12, 24, 36)) for b in per]).to_csv(
        os.path.join(outdir("eb"), "5_states_common_lag.csv"), index=False)
    ta["width"] = ta.upper - ta.lower
    print(ta.round(3).to_string(index=False))
