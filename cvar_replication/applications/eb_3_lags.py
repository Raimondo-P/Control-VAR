"""eb_3_lags.py: lag order of the EB differenced branch by the pre-specified rule (smallest order
with no Ljung-Box rejection at 5 percent in the EB and outcome equations), separately for the CVAR
and the VAR without controls; AIC and BIC reported for reference.
Output: results/eb/3_lags.csv
Usage: py eb_3_lags.py"""
from common import *

if __name__ == "__main__":
    UR, EB, EUC, elig = load_eb()
    rows = []
    for s in elig:
        r = fit_eb_diff(UR, EB, EUC, s, 1)
        Zc, Zv = r.Z, r._nc_fit()[0]
        dc = cvar.lag_diagnostics(Zc, lags=range(1, 13))
        dv = cvar.lag_diagnostics(Zv, lags=range(1, 13))
        rows.append(dict(st=s, p_cvar=lag_rule(Zc, None, [1, 2], 12), p_var=lag_rule(Zv, None, [1, 2], 12),
                         aic_cvar=dc.AIC.idxmin(), bic_cvar=dc.BIC.idxmin(),
                         aic_var=dv.AIC.idxmin(), bic_var=dv.BIC.idxmin()))
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(outdir("eb"), "3_lags.csv"), index=False)
    print(o.to_string(index=False))
