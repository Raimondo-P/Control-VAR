# Replication Folder: "Control VAR: a counterfactual based approach to inference in macroeconomics"

This folder contains all code and data to reproduce the figures and results in Pala (2025). The companion folder `CVAR example/` is a simpler self-contained application and is the recommended starting point if you are new to the toolbox.

---

## Directory structure

```
REPLICATION FOLDER/
├── scripts/                        # Run these to reproduce all paper figures
│   ├── a_preliminary_figure.m      # Descriptive figures (disasters, IP, uncertainty)
│   ├── b_original_LMN.m            # Benchmark VAR: filtered and unfiltered (LMN replication)
│   ├── c_CVAR_model.m              # Main result: Control VAR with US/Germany
│   ├── d_rob1_justdummy.m          # Robustness 1: disaster as binary dummy
│   └── e_rob2_manycontrols.m       # Robustness 2: multiple control countries (DE, IT, FR, ES)
│
├── CVAR_toolbox/                   # Estimation functions
│   ├── core/                       # VAR estimation, IRF, lag selection, VECM
│   ├── identification/             # SVARCholDiD.m, SVARivDiD.m
│   ├── diagnostics/                # ADF, cointegration, serial correlation tests
│   └── plotting/                   # PlotSwathe, VARirplot, figure formatting
│
├── data/
│   ├── raw/                        # Excel input files
│   │   ├── replication_data.xlsx   # US uncertainty, IP, disaster cost and deaths (1980–2020)
│   │   ├── fredgraph.xlsx          # Industrial production: US, DE, IT, FR, ES (FRED)
│   │   ├── Input_Matlab_VAR_Data.xlsx     # EU industrial production by country
│   │   └── Uncertainty_indicators_MU1_MU2.xlsx  # EU macroeconomic uncertainty
│   └── processed/                  # Pre-computed .mat files
│       ├── data_for_VAR.mat        # Processed uncertainty series (mac_unc_US, mac_unc_GER, etc.)
│       └── GERMANY_DISASTERS.mat   # German disaster cost series
│
└── output/                         # Generated figures and diagnostics (empty on first run)
    ├── figures/
    └── diagnostics/
```

---

## How to reproduce the results

Scripts are named alphabetically in the order they appear in the paper. Each script adds its own paths and is fully independent — run any one individually.

### Step 1 — Open MATLAB and navigate to the scripts folder

```matlab
cd 'REPLICATION FOLDER/scripts'
```

### Step 2 — Run the scripts

| Script | Figure | Content |
|--------|--------|---------|
| `a_preliminary_figure.m` | `Figure_preliminary_1/2/3.pdf` | Descriptive: disaster costs, IP and uncertainty for US and Germany (1996–2015) |
| `b_original_LMN.m` | `Figure_IRF_detrended.pdf`, `Figure_IRF_trend.pdf` | Benchmark VAR-in-differences: filtered (Mann-Whitney detrended) and unfiltered IRFs, replicating LMN-style results |
| `c_CVAR_model.m` | `Figure_IRF_CVAR.pdf` | Main result: Control VAR with US treated, Germany as control; treatment = disaster cost dummy (top 5%); GFC dummy included |
| `d_rob1_justdummy.m` | `Figure_IRF_rob1_dummy.pdf` | Robustness: disaster as binary dummy (cost ≥ 95th percentile) without cointegration correction |
| `e_rob2_manycontrols.m` | `Figure_IRF_rob2_manycontrols.pdf` | Robustness: Control VAR with four control countries (DE, IT, FR, ES) for both IP and uncertainty |

All figures are saved to `output/figures/`. Diagnostic tables (lag selection, cointegration rank) are saved to `output/diagnostics/`.

---

## Data sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| Disaster cost and deaths | NOAA/NCEI Billion-Dollar Weather and Climate Disasters | US, 1980–2020 |
| US macroeconomic uncertainty | Baker, Bloom & Davis (2016) | US, 1980–2020 |
| Industrial production | FRED (Federal Reserve Bank of St. Louis) | US, DE, IT, FR, ES |
| EU macroeconomic uncertainty | Rossi, Sekhposyan & Soupre | DE, IT, FR, ES |
| German disaster costs | `GERMANY_DISASTERS.mat` | DE |

---

## Key modelling choices in the main application (`c_CVAR_model.m`)

- **Sample:** 1996:08 – 2015:12 (234 monthly observations)
- **Treated unit:** United States (`y_t` = log IP, macroeconomic uncertainty)
- **Control unit:** Germany (`y_c` = log IP, macroeconomic uncertainty)
- **Treatment variable:** disaster cost dummy (`D_t = 1` if cost ≥ 95th percentile)
- **Lag order:** selected by `SVARCholDiD` up to `r = 6`
- **Additional regressor:** GFC dummy (`other`, 2007:08 – 2008:12)
- **Identification:** Cholesky, `D_t` ordered first (DiD exclusion restriction)
- **Horizons:** 20 months

---

## Citation

```bibtex
@article{pala2025controlvar,
  title   = {Control {VAR}: a counterfactual based approach to inference in macroeconomics},
  author  = {Raimondo Pala},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.23762}
}
```

---

**Last updated:** June 2026
