"""desc_figures.py: descriptive figures for the two applications, two panels stacked vertically.
  fig_eb_data.pdf/.png  top: weeks of EB and of EUC available, averaged over the treated states;
                        bottom: unemployment rate, average of the treated states and average of the
                        never-triggered controls (equal weights, as in the CVAR).
  fig_ca_data.pdf/.png  top: LCFS cost per gallon (CARB formula until April 2024, refiners' reported
                        cost from May 2024; the formula over the whole sample dashed);
                        bottom: retail regular gasoline, California and the average of the East Coast,
                        Midwest, Gulf Coast and Rocky Mountain regions (monthly averages of weekly prices,
                        not seasonally adjusted).
Requires ca_1_cec.py for the California figure. Output: results/paper/
Usage: py desc_figures.py
"""
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = outdir("paper")

# ------------------------------------------------------------------ extended benefits
UR, EB, EUC, elig = load_eb()
eb_w = 10 * EB[elig].mean(1)            # weeks (load_eb divides by ten)
euc_w = 10 * EUC[elig].mean(1)
ur_t = UR[elig].mean(1)
ur_c = UR[CTRL].mean(1)
fig, ax = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
ax[0].plot(eb_w.index, eb_w.values, color="C1", lw=1.5, label="EB")
ax[0].plot(euc_w.index, euc_w.values, color="C7", lw=1.2, ls="--", label="EUC")
ax[0].set_ylabel("weeks of benefits")
ax[0].set_title(f"Weeks of extended benefits, average of the {len(elig)} treated states", fontsize=10)
ax[0].legend(fontsize=8, frameon=False)
ax[1].plot(ur_t.index, ur_t.values, color="C3", lw=1.5, label=f"treated states ({len(elig)})")
ax[1].plot(ur_c.index, ur_c.values, color="C0", lw=1.5, label=f"controls ({len(CTRL)})")
ax[1].set_ylabel("percent")
ax[1].set_title("Unemployment rate, averages", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)
for x in ax:
    x.grid(alpha=0.25)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_eb_data.{ext}"), dpi=200)
plt.close(fig)
print("fig_eb_data written")

# ------------------------------------------------------------------ California
D, first = load_california()
D = D[D.cost_spliced.notna()]
ctrl = ["gas_r10", "gas_r20", "gas_gulf", "gas_r40"]
fig, ax = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
ax[0].plot(D.index, D.cost_spliced, color="C1", lw=1.5, label="cost per gallon (formula, then refiners' reports)")
ax[0].plot(D.index, D.cost_tv, color="C1", lw=1.0, ls=":", label="formula over the whole sample")
ax[0].axvline(first, color="grey", lw=0.8, ls="-.")
ax[0].set_ylabel("cents per gallon")
ax[0].set_title("Cost of the Low Carbon Fuel Standard per gallon of gasoline", fontsize=10)
ax[0].legend(fontsize=8, frameon=False)
ax[1].plot(D.index, 100 * D.gas_ca, color="C3", lw=1.5, label="California")
ax[1].plot(D.index, 100 * D[ctrl].mean(1), color="C0", lw=1.5, label="average of PADDs 1 to 4")
ax[1].set_ylabel("cents per gallon")
ax[1].set_title("Retail price of regular gasoline", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)
for x in ax:
    x.grid(alpha=0.25)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_ca_data.{ext}"), dpi=200)
plt.close(fig)
print("fig_ca_data written")
