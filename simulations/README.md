# Simulations: Control VAR

Python 3.10 or later with numpy, scipy, pandas, statsmodels and matplotlib. All scripts use all cores by
default (`--workers` to change it). `mc_diff_branch.py` imports the `cvar` package from `../applications`.

## How to run

From this folder:

```
py run_all.py --list               # the steps and the tables or figures each produces
py run_all.py                      # everything
py run_all.py --steps diff figs    # selected steps
```

Every step writes to `results/<folder>`. The whole set takes roughly one to two days on 8 cores, most of it
in the bootstrap grid (`grid`).

## Map from the paper to the code

| Paper (LaTeX label) | Step | Script | Output |
|---|---|---|---|
| tab:mc-var (VAR without controls vs CVAR) | sim1 | mc_bootstrap_cvar.py | results/mc_sim1 |
| tab:mc-lags (lag length) | lags | mc_bootstrap_cvar.py, lag rules aic, 8, 12 | results/mc_lag_* |
| tab:mc-boot, tab:mc-grid (bootstrap grid) | grid | mc_bootstrap_cvar.py --grid | results/mc_grid_p12 |
| tab:mc-spill, tab:mc-nonlin, tab:mc-stat | ext | mc_extensions.py | results/mc_ext |
| tab:mc-test-size, tab:mc-test-feedback | tables11_14 | make_tables_11_14.py | results/tables_11_14 (CSV and LaTeX) |
| tab:mc-test-power, fig:mc-test-T200, fig:mc-test-T400 | test, tex, figs | mc_final_test.py, build_tex_tables.py | results/mc_final, results/tex, results/figures |
| tab:mc-test-lags | lags_boot, tex | mc_boot_fix.py, mc_boot_fix2.py, build_tex_tables.py | results/mc_bootfix*, results/tex |
| tab:supp-cpo, fig:supp-cpo | supp, tex, figs | mc_cpok.py, build_tex_tables.py | results/mc_cpok, results/tex, results/figures |
| tab:mc-diff (CVAR in changes, Theorem thm:diff) | diff | mc_diff_branch.py | results/mc_diff (CSV and LaTeX) |
| tab:mc-joint (bootstrap schemes with estimated weights, Theorem thm:joint) | joint, tex_joint | mc_bootstrap_cvar.py, build_joint_table.py | results/mc_joint_*, results/tex/tab_mc_joint.tex |

The `figs` step copies the simulation figures to `results/figures/` under the file names used in the paper.
Every simulation table is written as LaTeX: tab:mc-test-size, tab:mc-test-feedback and tab:mc-diff by their scripts,
tab:mc-test-power, tab:mc-test-lags and tab:supp-cpo by build_tex_tables.py, tab:mc-joint by build_joint_table.py,
and the others by make_tables.py (run by the `tables` step of ../replicate.py).

## Code
Entry scripts (run by `run_all.py`): mc_bootstrap_cvar.py, mc_extensions.py, mc_final_test.py, make_tables_11_14.py, mc_boot_fix.py, mc_boot_fix2.py, mc_cpok.py, build_tex_tables.py, mc_diff_branch.py.

Library modules (imported by the entry scripts; not run on their own for the paper):

| Module | Content |
|---|---|
| mc_bootstrap_cvar.py | data generating process of the Monte Carlo, the weights regression and the rank step (also an entry script) |
| po_critical_values.py | the point-optimal statistic and quasi-differencing helpers |
| robust_po.py | the statistic with the Ng-Perron long-run variance (the test of Section 5) |
| mc_competitors.py | Perron-Rodriguez, Engle-Granger, Johansen, error-correction and Harvey-Thiele tests |
| mc_adaptive.py | triangular and CVAR test designs, the known-weights bound |
| mc_cpo2.py, mc_cpok.py | conditional tests of the supplementary appendix |
| mc_envelope.py, mc_power.py | helpers imported by robust_po.py |

## Data
`../applications/data/eb/eb_euc_panel.csv` (read by the applications, not by the simulations): monthly state panel, October 2002 to December 2019. Unemployment rates are the seasonally adjusted LAUS series from FRED (`{ST}UR`, for example `TXUR`). Weeks of extended benefits and of emergency unemployment compensation come from the replication files of Boone, Dube, Goodman and Kaplan (2021, AEJ: Economic Policy; openICPSR 118525, `duration_weekly.dta`), aggregated to months.

## Test of the design in applications
The recommended test is the statistic of Section 5 (`robust_po.po_maic` with cbar = -7) with the sieve-bootstrap p-value and an autoregression of order p_T = ceil(16 (T/200)^(1/3)), holding the controls and policy terms fixed (`mc_final_test.boot_p` with `P_SIEVE` set by the rule, as in `make_tables_11_14.py`). The asymptotic critical values are those of Elliott, Rothenberg and Stock (1996) with a constant: 1.99, 2.55, 3.26 and 4.48 at 1, 2.5, 5 and 10 percent.
