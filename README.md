Control-VAR: MATLAB Toolbox for Counterfactual Inference in Macroeconomics
A self-contained MATLAB package for estimating Control VARs (CVARs) — a cointegration-based approach to difference-in-differences identification in VAR models. The CVAR exploits a shared stochastic trend between a treated and a control series to recover the average treatment effect on the treated (ATT) without filtering, using a Cholesky or IV identification scheme on a VECM representation. To the best of the author's knowledge, this is one of the few publicly available MATLAB implementations of this approach.
Author: Raimondo Pala (University of Rome Tor Vergata)
Paper: Control VAR: a counterfactual based approach to inference in macroeconomics
 
What this repository contains
Control-VAR/
├── CVAR example/        # Self-contained worked example: disaster impacts on economic activity
├── REPLICATION FOLDER/  # Full replication code and data for all results in Pala (2025)
└── Control VAR.pdf      # Working paper
•	CVAR example/ is the quickest entry point. It contains a stripped-down application estimating the impact of natural disasters on industrial production, with a single setpath.m initializer and five ready-to-run scripts. Start here if you want to understand how the toolbox works or adapt it to your own data.
•	REPLICATION FOLDER/ contains the full replication code for all empirical applications and Monte Carlo simulations in Pala (2025), including multi-country comparisons (US, Germany, Italy, France, Spain) and the bias comparison between VAR-in-differences and CVAR. Both folders bundle the same CVAR_toolbox functions.
 
When to use this package
•	You want to estimate a difference-in-differences treatment effect in a VAR/time series setting where the treated and control series share a stochastic trend (are cointegrated).
•	You want to avoid the attenuation bias of VAR-in-differences when data are non-stationary: without an error-correction term, the level IRF does not converge to zero and identifies a filtered ATT rather than the true ATT.
•	You want Cholesky identification imposing that contemporaneous shocks to the outcome cannot move the treatment variable (the DiD exclusion restriction).
•	You want IV identification (SVARivDiD) as an alternative to Cholesky.
•	You want a parallel trend diagnostic (CVARparalleltrend) after estimation.
•	You are replicating or building on Pala (2025).
 
Quick start
% 1. Navigate to CVAR example/ and initialise paths
cd 'CVAR example'
setpath

% 2. Run the worked example (~30 seconds)
cd scripts
only_dummy

% Output: output/figures/Figure_IRF_dummy.pdf
To use the toolbox on your own data:
% Full pipeline: lag selection → VECM estimation → Cholesky IRF
[IRbar, VARopt, IRinf, IRsup] = SVARCholDiD( ...
    D_t, y_t, y_c, 4, 20, colnames, [], foldername);

% With a known cointegrating vector (e.g. [1, -1]), skipping Johansen
[IRbar, VARopt, IRinf, IRsup] = SVARCholDiD( ...
    D_t, y_t, y_c, 4, 20, colnames, [], foldername, 1, [1;-1]);

 
Why CVAR instead of VAR-in-differences
When y_t (treated) and y_c (control) share a common I(1) trend, first-differencing removes the trend but also discards the error-correction mechanism. The level IRF recovered by cumulating a differenced VAR does not mean-revert, producing a spurious long-run effect. The CVAR enters ec_{t-1} = y_{t-1} - y_{c,t-1} as an exogenous regressor, which absorbs the shared trend while keeping the system in levels. By Granger's Representation Theorem the IRF then converges to zero at finite horizons (Johansen 1995, Thm 4.2; Pala 2025, Theorem 8).
Monte Carlo evidence (2000 draws):
Estimator	Bias	Std	IRF at h=10
VAR-in-diff	~0.10	~0.16	≠ 0 (does not converge)
CVAR	~0.00	~0.16	≈ 0 (converges)
 
Requirements
•	MATLAB R2019b or later
•	No additional toolboxes required for core estimation; Statistics Toolbox used in some diagnostics
 
Citation
If you use this package or replicate results from the paper, please cite:
@article{pala2025controlvar,
  title   = {Control {VAR}: a counterfactual based approach to inference in macroeconomics},
  author  = {Raimondo Pala},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.23762}
}
 
Keywords
control VAR · CVAR · difference-in-differences · DiD · VAR · VECM · cointegration · error correction · MATLAB · counterfactual · treatment effects · ATT · Cholesky identification · SVAR · IV identification · impulse response functions · IRF · non-stationary time series · stochastic trend · macroeconometrics · natural disasters · fiscal policy

