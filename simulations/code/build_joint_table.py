"""build_joint_table.py: LaTeX table of the bootstrap schemes with estimated weights (step 'joint').
Reads results/mc_joint_*/all_summaries.csv and writes results/tex/tab_mc_joint.tex.
Usage: py build_joint_table.py [--res ../results]"""
import argparse, glob, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--res", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
a = ap.parse_args()
RUNS = [("mc_joint_fb03", r"$T=200$, $\phi=0.3$"), ("mc_joint_fb045", r"$T=200$, $\phi=0.45$"),
        ("mc_joint_T400", r"$T=400$, $\phi=0.3$")]
SCH = [("oracle", "Known weights"), ("fixed", "Fixed weights"), ("refit", "Re-estimated, controls fixed"),
       ("joint", "Joint, not recentred"), ("jointrc", "Joint, recentred (Theorem \\ref{thm:joint})")]
H = [0, 1, 3, 6, 12, 24]
lines = []
for folder, lab in RUNS:
    f = os.path.join(a.res, folder, "all_summaries.csv")
    if not os.path.exists(f):
        print("missing", f)
        continue
    d = pd.read_csv(f)
    d = d[d.target == "gamma"]
    reps = None
    lines.append(r"\multicolumn{8}{l}{" + lab + r"} \\")
    for k, nm in SCH:
        z = d[d.scheme == k].set_index("h")
        if z.empty:
            continue
        cells = [nm] + [f"{z.loc[h].coverage:.2f}" for h in H] + [f"{z.loc[12].width:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
lines = lines[:-1]
tab = [r"\begin{table}[htbp]", r"\centering",
       r"\caption{Coverage of the 90 percent bands for $\gamma_h$ with estimated weights: bootstrap schemes\label{tab:mc-joint}}",
       r"\small", r"\begin{tabular}{lccccccc}", r"\toprule",
       r" & \multicolumn{6}{c}{Coverage, months after the surprise} & Width \\",
       r"\cmidrule(lr){2-7}\cmidrule(lr){8-8}",
       r" & " + " & ".join(str(h) for h in H) + r" & 12 \\", r"\midrule"] + lines + [
       r"\bottomrule", r"\end{tabular}",
       r"\par\smallskip\parbox{0.95\textwidth}{\footnotesize Six controls with one cointegrating relation among them, "
       r"feedback $\phi$ from the lagged cycle to the policy, 12 lags in the VAR of the CVAR and in the joint VAR, "
       r"500 replications and 499 bootstrap draws. Known weights: pseudo-true weights. Fixed weights: estimated "
       r"weights held fixed. Re-estimated, controls fixed: weights re-estimated in every draw with the controls at "
       r"their observed values. Joint: the controls are regenerated with the policies and the cycle and the weights re-estimated, without recentring, or with the draws recentred at the response of the bootstrap world as in Theorem \ref{thm:joint}.}",
       r"\end{table}"]
os.makedirs(os.path.join(a.res, "tex"), exist_ok=True)
open(os.path.join(a.res, "tex", "tab_mc_joint.tex"), "w").write("\n".join(tab) + "\n")
print("written", os.path.join(a.res, "tex", "tab_mc_joint.tex"))
