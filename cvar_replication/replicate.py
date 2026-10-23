r"""replicate.py: reproduce every table and figure of the paper and its supplement.

Steps, in order (each can be run on its own with --only):
  alaska     applications/ak_0 ... ak_6: screen, estimates, checks, exhibits, diagnostic
  eb         applications/eb_1 ... eb_7: extended unemployment benefits
  ca         applications/ca_1 ... ca_4: the Low Carbon Fuel Standard
  figures    applications/desc_figures.py: the two data figures
  sims       simulations/run_all.py: every simulation of the supplement (the longest step)
  screens    screens/: the test of the design on US states and its power
  tables     simulations/make_tables.py and screens/make_screen_tables.py: the tables built from the outputs above
Outputs: results/ (applications; results/paper holds the exhibits the paper reads),
simulations/results/ (simulations), screens/results/ (screens). Console output of each script is
saved in results/logs/.
Usage (from this folder):
  python replicate.py                         # everything, with the paper's numbers of draws
  python replicate.py --only alaska eb ca     # some steps
  python replicate.py --B 99 --reps 50        # a quick check with few draws (not the paper's results)
Requires Python 3.10 or later with numpy, scipy, pandas, statsmodels, matplotlib, requests.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "applications")
STEPS = {
    "alaska": [(APP, s) for s in ["ak_0_screen.py", "ak_1_application.py", "ak_2_checks.py", "ak_3_exhibits.py",
                                  "ak_6_diagnostic.py"]],
    "eb": [(APP, s) for s in ["eb_1_unit_roots.py", "eb_2_design_tests.py", "eb_3_lags.py", "eb_4_states.py",
                              "eb_5_average.py", "eb_6_exhibits.py", "eb_7_lpdid.py"]],
    "ca": [(APP, s) for s in ["ca_1_cec.py", "ca_2_robustness.py", "ca_3_a4.py", "ca_4_exhibits.py"]],
    "figures": [(APP, "desc_figures.py")],
    "sims": [(os.path.join(HERE, "simulations"), "run_all.py")],
    "screens": [(os.path.join(HERE, "screens"), s) for s in ["st_screen.py", "st_screen_vars.py", "mc_screen_power.py"]],
    "tables": [(os.path.join(HERE, "simulations"), "make_tables.py"), (os.path.join(HERE, "screens"), "make_screen_tables.py")],
}
# options each script accepts
TAKES_B = {"ak_0_screen.py", "ak_1_application.py", "ak_2_checks.py", "ak_3_exhibits.py", "ak_6_diagnostic.py",
           "eb_1_unit_roots.py", "eb_2_design_tests.py", "eb_4_states.py", "eb_5_average.py", "eb_6_exhibits.py",
           "ca_2_robustness.py", "ca_4_exhibits.py", "st_screen.py", "st_screen_vars.py", "mc_screen_power.py"}
TAKES_W = {"ak_0_screen.py", "ak_1_application.py", "ak_2_checks.py", "ak_3_exhibits.py", "ak_6_diagnostic.py",
           "eb_1_unit_roots.py", "eb_2_design_tests.py", "eb_4_states.py", "run_all.py", "mc_screen_power.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(STEPS), choices=list(STEPS))
    ap.add_argument("--B", type=int, default=None, help="bootstrap draws (default: each script's own, as in the paper)")
    ap.add_argument("--reps", type=int, default=None, help="simulation replications (default: as in the paper)")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    a = ap.parse_args()
    logs = os.path.join(HERE, "results", "logs")
    os.makedirs(logs, exist_ok=True)
    t0 = time.time()
    for step in a.only:
        for folder, script in STEPS[step]:
            path = os.path.join(folder, script)
            if not os.path.exists(path):
                print(f"==== {step}: {script} not found, skipped", flush=True)
                continue
            cmd = [sys.executable, path]
            if a.B is not None and script in TAKES_B:
                cmd += ["--B", str(a.B)]
            if script in TAKES_W:
                cmd += ["--workers", str(a.workers)]
            if script == "run_all.py" and a.reps is not None:
                cmd += ["--reps", str(a.reps)]
            log = os.path.join(logs, f"{step}_{script.replace('.py', '')}.log")
            t = time.time()
            print(f"==== {step}: {script}", flush=True)
            with open(log, "w") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=folder)
            status = "done" if r.returncode == 0 else f"FAILED (see {os.path.relpath(log, HERE)})"
            print(f"     {status} in {(time.time() - t) / 60:.1f} min", flush=True)
            if r.returncode:
                sys.exit(1)
    print(f"all steps done in {(time.time() - t0) / 3600:.1f} h; exhibits in results/paper and simulations/results")


if __name__ == "__main__":
    main()
