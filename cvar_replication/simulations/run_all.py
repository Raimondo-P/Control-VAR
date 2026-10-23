#!/usr/bin/env python3
"""
Reproduce every simulation table and figure of
"Control VAR: a counterfactual based approach to inference in macroeconomics".

Usage (from this folder):
    py run_all.py                      # everything, in order
    py run_all.py --steps sim1 test    # selected steps only
    py run_all.py --list               # show the steps
    py run_all.py --workers 8          # number of processes (default: all cores)
    py run_all.py --reps 5000 --steps test diff   # more replications for selected steps
Each step writes to results/<step folder>. Steps are independent and can be run in any order,
except "tex", which reads the outputs of "test", "lags_boot" and "supp", and "figs", which copies the figures of "test" and "supp".
Run times are for 8 cores and are indicative.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
RES = os.path.join(HERE, "results")
COPY_FIGS = ("import os, shutil; R = r'" + RES + "'; os.makedirs(os.path.join(R, 'figures'), exist_ok=True); "
             "M = {'mc_final/power_curves_T200.png': 'Figure_MC_test_T200.png', "
             "'mc_final/power_curves_T400.png': 'Figure_MC_test_T400.png', "
             "'mc_cpok/power_curves.png': 'Figure_MC_cpo.png'}; "
             "[shutil.copy(os.path.join(R, s), os.path.join(R, 'figures', d)) for s, d in M.items() "
             "if os.path.exists(os.path.join(R, s))]")

# replication flag of each simulation script (run_all.py --reps N passes N to all of them)
REPS_FLAG = {"mc_bootstrap_cvar.py": "--reps", "mc_extensions.py": "--reps", "mc_final_test.py": "--reps",
             "make_tables_11_14.py": "--reps", "mc_boot_fix.py": "--reps", "mc_boot_fix2.py": "--reps",
             "mc_cpok.py": "--reps", "mc_diff_branch.py": "--R"}


def steps(w):
    W = ["--workers", str(w)]
    return [
        ("sim1", "Simulation 1, VAR without controls vs CVAR (tab:mc-var); 1-2 h",
         [["mc_bootstrap_cvar.py", "--T", "200", "--K", "6", "--r", "5", "--feedback", "0.3", "--lagrule", "12", "--out", f"{RES}/mc_sim1"] + W]),
        ("lags", "Simulation 2, lag length (tab:mc-lags), AIC, 8 and 12 lags; 2-4 h",
         [["mc_bootstrap_cvar.py", "--T", "200", "--K", "6", "--r", "5", "--feedback", "0.3", "--lagrule", lr, "--out", f"{RES}/mc_lag_{lr}"] + W
          for lr in ["aic", "8", "12"]]),
        ("grid", "Simulation 3, bootstrap grid of 17 designs (tab:mc-boot, tab:mc-grid); 10-20 h",
         [["mc_bootstrap_cvar.py", "--grid", "--lagrule", "12", "--out", f"{RES}/mc_grid_p12"] + W]),
        ("ext", "Spillovers, non-linear effects, stationarity tables (tab:mc-spill, tab:mc-nonlin, tab:mc-stat); 0.5-1 h",
         [["mc_extensions.py", "--exp", "all", "--out", f"{RES}/mc_ext"] + W]),
        ("test", "Test of the design, power table and figures (tab:mc-test-power, fig:mc-test-T200, fig:mc-test-T400); 1-3 h",
         [["mc_final_test.py", "--out", f"{RES}/mc_final"] + W]),
        ("tables11_14", "Size table and feedback table with the bootstrap lag rule (tab:mc-test-size, tab:mc-test-feedback); 2-4 h",
         [["make_tables_11_14.py", "--out", f"{RES}/tables_11_14"] + W]),
        ("lags_boot", "Order of the sieve autoregression (tab:mc-test-lags); 2-3 h",
         [["mc_boot_fix.py", "--out", f"{RES}/mc_bootfix"] + W, ["mc_boot_fix2.py", "--out", f"{RES}/mc_bootfix2"] + W]),
        ("supp", "Conditional and adaptive tests, supplementary appendix (tab:supp-cpo, fig:supp-cpo); 0.5-1 h",
         [["mc_cpok.py", "--out", f"{RES}/mc_cpok"] + W]),
        ("tex", "LaTeX for tab:mc-test-power, tab:mc-test-lags, tab:supp-cpo (reads test, lags_boot, supp); seconds",
         [["build_tex_tables.py", "--results", RES, "--out", f"{RES}/tex"]]),
        ("joint", "Joint bootstrap with the lag rule of the paper, design of sim1 and a stronger feedback (tab:mc-boot update); feedback 0.3 and 0.45 at T = 200, and T = 400; 3-6 h",
         [["mc_bootstrap_cvar.py", "--T", "200", "--K", "6", "--r", "5", "--feedback", "0.3", "--lagrule", "12", "--out", f"{RES}/mc_joint_fb03"] + W,
          ["mc_bootstrap_cvar.py", "--T", "200", "--K", "6", "--r", "5", "--feedback", "0.45", "--lagrule", "12", "--out", f"{RES}/mc_joint_fb045"] + W,
          ["mc_bootstrap_cvar.py", "--T", "400", "--K", "6", "--r", "5", "--feedback", "0.3", "--lagrule", "12", "--out", f"{RES}/mc_joint_T400"] + W]),
        ("tex_joint", "LaTeX table of the bootstrap schemes (tab:mc-joint), from the outputs of joint; seconds",
         [["build_joint_table.py", "--res", RES]]),
        ("diff", "CVAR in changes with imposed weights (tab:mc-diff); 1-3 h",
         [["mc_diff_branch.py", "--out", f"{RES}/mc_diff"] + W]),
        ("figs", "Copy the simulation figures under the names used in the paper (results/figures); seconds",
         [["-c", COPY_FIGS]]),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reps", type=int, default=None,
                    help="replications for every simulation script (default: each script's own default)")
    a = ap.parse_args()
    S = steps(a.workers)
    if a.list:
        for name, desc, _ in S: print(f"{name:12s} {desc}")
        return
    chosen = [s for s in S if not a.steps or s[0] in a.steps]
    os.makedirs(RES, exist_ok=True)
    env = dict(os.environ, CVAR_OUT=os.path.join(RES, "application"))
    for name, desc, cmds in chosen:
        print(f"\n=== {name}: {desc}", flush=True); t0 = time.time()
        for c in cmds:
            if a.reps and c[0] in REPS_FLAG:
                c = c + [REPS_FLAG[c[0]], str(a.reps)]
            subprocess.run([sys.executable] + c, cwd=CODE, env=env, check=True)
        print(f"=== {name} done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
