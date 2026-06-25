# Control VAR (CVAR) Toolbox

Companion code for:
> Pala, R. (2025). *Control VAR: a counterfactual based approach to inference in macroeconomics*. arXiv:2510.23762.

---

## Requirements

- MATLAB (R2019b or later recommended)
- The `CVAR_toolbox` folder (included) must be on the path

---

## Folder structure

```
CVAR_toolbox/
    core/           VAR estimation utilities (Cesa-Bianchi toolbox)
    identification/ Cholesky and IV identification
    diagnostics/    Lag selection, cointegration tests
    plotting/       IRF plotting helpers
SVARCholDiD.m           Full pipeline wrapper (lag selection → estimation → IRF)
SVARCholDiD_lags.m      Step 1: lag order selection via information criteria
SVARCholDiD_estimate.m  Step 2: VECM/VAR estimation, Cholesky identification
SVARCholDiD_irf.m       Step 3: IRF computation and bootstrap confidence bands
SVARivDiD.m             IV identification variant
CVARparalleltrend.m     Pre-treatment parallel trend diagnostic
example_single.m        Illustrative single-dataset example
example_mc.m            Monte Carlo bias comparison
```

---

## Quick start

### Single-dataset example

```matlab
run example_single.m
```

Generates one dataset from the DGP below, estimates both a VAR-in-differences
and a CVAR, and plots their impulse response functions side by side.

### Monte Carlo simulation

```matlab
run example_mc.m
```

Runs 2000 simulated datasets and compares the two estimators in terms of
bias, standard deviation, and IRF shape.

---

## The Monte Carlo simulation

### What it does

The simulation compares two estimators of the average treatment effect on
the treated (ATT) under the following data generating process:

```
tau_t  = tau_{t-1} + eta_t          shared I(1) stochastic trend
y_t    = tau_t + gamma_0 * D_t + u_t    treated series
y_c    = tau_t + w_t                     control series
D_t  ~ Bernoulli(0.10)              binary treatment, i.i.d.
```

Because `y_t` and `y_c` share the same trend, `y_t - y_c ~ I(0)`:
the two series are cointegrated with known vector `beta = [1, -1]`.
The true impact effect is `gamma_0 = -1`.

### Estimator 1 — VAR-in-differences (benchmark)

Regresses `diff(y_t)` on its own lags and `D_t`, with no control series.
This is the standard approach in the macro literature when data are
non-stationary. The level IRF is recovered as the cumulative sum of
the first-differenced IRF.

**Failure mode:** without the EC term there is no mechanism that forces
the level IRF back to zero. The mean IRF across 2000 draws remains
persistently negative at long horizons, identifying a filtered ATT
(F-ATT) rather than the true ATT.

### Estimator 2 — CVAR / VECM

Estimates `[D_t, y_t]` jointly in levels with the error-correction term
`ec_{t-1} = y_{t-1} - y_{c,t-1}` entered as an exogenous regressor.
`D_t` is ordered first in the Cholesky decomposition, which imposes that
contemporaneous shocks to `y_t` cannot move `D_t` (the DiD exclusion
restriction). The ATT is identified as `gamma_0 = P(2,1) / P(1,1)`.

**Why it works:** the EC term absorbs the shared I(1) trend without
filtering, so the system is stationary in levels and the IRF converges
to zero at finite horizons by Granger's Representation Theorem
(Johansen 1995, Thm 4.2; Pala 2025, Theorem 8).

### Results

| Estimator | Bias | Std | IRF at h=10 |
|-----------|------|-----|-------------|
| VAR-diff  | ~0.10 | ~0.16 | ≠ 0 (does not converge) |
| CVAR      | ~0.00 | ~0.16 | ≈ 0 (converges) |

The CVAR is approximately unbiased and its IRF converges to zero.
The VAR-diff is slightly biased on impact and its level IRF accumulates
a spurious long-run effect driven by the unmodelled shared trend.

---

## Using the toolbox on your own data

The simplest entry point is `SVARCholDiD`, which runs the full pipeline:

```matlab
% Inputs
%   D_t      : (T x 1) binary or continuous treatment variable
%   y_t      : (T x n) treated outcome series
%   y_c      : (T x n) control series (same dimension as y_t)
%   r        : maximum lag order to consider (e.g. 4)
%   horizons : number of IRF horizons (e.g. 20)
%   colnames : variable names, e.g. {'D_t','IP_Italy','IP_Germany'}
%   foldername : output folder for tables (e.g. pwd)

[IRbar, VARopt, IRinf, IRsup] = SVARCholDiD( ...
    D_t, y_t, y_c, 4, 20, colnames, [], foldername);
```

To impose a known cointegrating vector (e.g. `[1,-1]`) and skip
Johansen estimation:

```matlab
[IRbar, VARopt, IRinf, IRsup] = SVARCholDiD( ...
    D_t, y_t, y_c, 4, 20, colnames, [], foldername, 1, [1;-1]);
```

To run the steps individually:

```matlab
p   = SVARCholDiD_lags(D_t, y_t, y_c, 4, foldername);
VAR = SVARCholDiD_estimate(D_t, y_t, y_c, p, colnames, foldername, ...
          [], [], [1;-1], 0, 0, 0);
[IRbar, IRinf, IRsup] = SVARCholDiD_irf(VAR, VARoption, 20);
```

To run the parallel trend diagnostic after estimation:

```matlab
CVARparalleltrend(VAR, D_t, colnames);
```

---

## Key parameters in `SVARCholDiD_estimate`

| Parameter | Values | Description |
|-----------|--------|-------------|
| `D_exog`  | `1` (default) | D_t exogenous; gamma = OLS coeff |
|           | `0` | D_t endogenous; Cholesky identification |
| `beta_fix` | e.g. `[1;-1]` | Impose cointegrating vector; skip Johansen |
| `ncoint`  | integer | Number of cointegrating relations (Johansen path) |
| `include_D_lag` | `0/1` | Include D_{t-1} as regressor (D_exog=1 only) |
| `const`   | `0/1` | Include intercept |

