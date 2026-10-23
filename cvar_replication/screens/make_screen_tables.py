r"""make_screen_tables.py: the two tables of the screen (Supplementary Appendix, Section S.7) from the outputs
of st_screen_vars.py (results/st_vars_summary.csv) and mc_screen_power.py (results/mc_screen_power_summary.csv),
written to ../results/paper. Usage: python make_screen_tables.py"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "paper")
HEAD1 = '\\begin{table}[htbp]\n\\centering\n\\caption{The test of the design on US states, regions and metropolitan areas\\label{tab:screen}}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.2cm}cccccc}\n\\toprule\n & Units & Observations & \\multicolumn{3}{c}{Share rejecting at 5 percent} & Gap stationary, \\\\\n\\cmidrule(lr){4-6}\n & & & Alone & Equal & Estimated & series I(1) \\\\\n\\midrule\n'
FOOT1 = '\\bottomrule\n\\end{tabular}\n\\par\\smallskip\\parbox{0.95\\textwidth}{\\footnotesize Test of Section \\ref{sec:test} with a constant and no policy terms, sieve-bootstrap p-values with 499 draws. The 48 contiguous states and the District of Columbia, each against the states it borders; consumer prices for each Census region against the other three, and for New York, Chicago and Los Angeles against the other two. Observations: median sample length, from the start of each series (1976 for the employment-population ratio and the participation rate, 1990 for payroll employment, 1979 for the coincident index, 1948 for personal income, 1975 for house prices, 1966 for the regions and 1914 for the metropolitan areas) to the end of 2019. Alone: the unit without controls. Equal: the unit minus the average of its neighbours. Estimated: weights on the neighbours estimated inside the test. Last column: number of units for which the unit alone does not reject at 10 percent and the gap with the neighbours rejects at 5 percent with equal or estimated weights.}\n\\end{table}'
HEAD2 = '\\begin{table}[htbp]\n\\centering\n\\caption{Size and power of the test of the design in panels built like the state data\\label{tab:screen-power}}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccccccc}\n\\toprule\nSample & Half-life & $c$ & \\multicolumn{3}{c}{Equal weights} & \\multicolumn{2}{c}{Estimated weights} \\\\\n\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n & (years) & & Bootstrap & Residual & ERS & Bootstrap & Residual \\\\\n\\midrule\n'
FOOT2 = '\\bottomrule\n\\end{tabular}\n\\par\\smallskip\\parbox{0.95\\textwidth}{\\footnotesize Rejection rates at 5 percent in 300 replications of monthly panels with one common stochastic trend, four controls that load on it with stationary noise, and a treated unit that follows the common trend plus a gap: a random walk (trend of its own: the rows give the size) or a stationary autoregression with the half-life shown, $c=T(\\rho-1)$. Bootstrap: the sieve bootstrap of Proposition \\ref{prop:sieve}, 199 draws. Residual: a sieve bootstrap that takes the innovations from the unrestricted autoregression in levels and imposes the unit root when rebuilding the series \\citep{PaparoditisPolitis2003,PalmSmeekesUrbain2008}. ERS: the asymptotic critical value of \\citet{ERS1996}, valid only without estimated weights.}\n\\end{table}'
ORDER = [("epop", "Employment-population ratio", "months"), ("lfpr", "Participation rate", "months"),
         ("payroll", "Payroll employment", "months"), ("activity", "Coincident index", "months"),
         ("income", "Personal income", "quarters"), ("hpi", "House prices", "quarters"),
         ("cpi_region", "Consumer prices, Census regions", "months"),
         ("cpi_metro", "Consumer prices, three metropolitan areas", "months")]


def f2(x):
    return "" if pd.isna(x) else (f"{x:.2f}" if x >= 0 else f"$-${-x:.2f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    S = pd.read_csv(os.path.join(HERE, "results", "st_vars_summary.csv")).set_index("variable")
    rows = []
    for v, name, unit in ORDER:
        if v not in S.index:
            continue
        r = S.loc[v]
        rows.append(f"{name} & {int(r.units)} & {int(round(r["T"]))} {unit} & {f2(r.alone_rejects)} & {f2(r.equal_rejects)} & "
                    f"{f2(r.estimated_rejects)} & {int(r.gap_stationary_and_alone_I1)} \\\\")
    open(os.path.join(OUT, "tab_screen.tex"), "w").write(HEAD1 + "\n".join(rows) + "\n" + FOOT1 + "\n")
    P = pd.read_csv(os.path.join(HERE, "results", "mc_screen_power_summary.csv"))
    years = {180: 15, 300: 25, 528: 44}
    rows = []
    for k, T in enumerate(sorted(P["T"].unique())):
        if k:
            rows.append("\\addlinespace")
        for r in P[P["T"] == T].itertuples():
            null = pd.isna(r.rho) or str(r.rho) == "null"   # the null rows: an own trend
            if null:
                lab = f"{years.get(T, T)} years & trend of its own & --"
            else:
                c = f"$-${-r.c:.1f}" if r.c < 0 else f"{r.c:.1f}"
                lab = f" & {r.half_life_years:.1f} & {c}"
            rows.append(f"{lab} & {f2(r.equal_boot)} & {f2(r.equal_resid)} & {f2(r.equal_ers)} & {f2(r.estimated_boot)} & "
                        f"{f2(r.estimated_resid)} \\\\")
    open(os.path.join(OUT, "tab_screen_power.tex"), "w").write(HEAD2 + "\n".join(rows) + "\n" + FOOT2 + "\n")
    print("tab_screen.tex and tab_screen_power.tex written to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
