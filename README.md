# Replication package: Control VAR

Code and data for "Control VAR: a counterfactual based approach to inference in macroeconomics" (Raimondo Pala).
Everything is in Python; the CVAR itself is the package in `applications/cvar` (also released, with a MATLAB
version, as the library `controlvar`). A full replication README, with the map from every exhibit to the script
that produces it and the run times, accompanies the release.

## Running

Python 3.10 or later with numpy, scipy, pandas, statsmodels, matplotlib and requests. From this folder:

    python replicate.py                  # every table and figure, with the paper's numbers of draws
    python replicate.py --only alaska    # one step: alaska, eb, ca, figures, sims, screens, tables
    python replicate.py --B 99 --reps 50 # a quick check with few draws (not the paper's results)

`replicate.py` runs the steps in order and saves the console output of each script in `results/logs/`. The
simulations (`sims`) are by far the longest step. The exhibits of the paper and the supplement are written to
`results/paper`, `simulations/results` and `screens/results`.

## Layout

    replicate.py            driver of the whole package
    applications/
      cvar/                 the CVAR package (estimation, bootstrap, test of the design)
      ak_*.py               Alaska Permanent Fund Dividend: screen, estimates, checks, exhibits, diagnostic
      eb_*.py               extended unemployment benefits
      ca_*.py               Low Carbon Fuel Standard
      desc_figures.py       the two data figures; common.py: shared loaders
      data/                 data snapshot: alaska/, eb/, california/, states/
    simulations/
      run_all.py, code/     every Monte Carlo simulation of the supplement
      make_tables.py        the simulation tables from their outputs
    screens/                the test of the design on US states, regions and metropolitan areas, its power, its tables
    results/                outputs of the applications; results/paper holds the exhibits the paper reads

## Data

The data are public and included as a snapshot, which the scripts read by default so that the results
reproduce the paper. `applications/ak_0_screen.py --refresh` and `applications/data/california/download_california.py`
download them again; the Bureau of Labor Statistics revises its series every year, so refreshed data change the
results slightly.
