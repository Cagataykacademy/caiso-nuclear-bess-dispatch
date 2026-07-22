"""
Extra reviewer-grade figures (no retraining; all from shipped predictions/CSVs):
  1. fig_conformal_diagnostics.png
     (a) QR (uncalibrated) vs CQR bands on a sample week, with misses marked
     (b) conditional coverage by hour-of-day vs the 90% target
  2. fig_import_sensitivity.png
     S1/S4 cost vs import price; nuclear premium annotated (appendix)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(P, "outputs", "figures")
TBL = os.path.join(P, "outputs", "tables")
Q_HAT = 1044.0

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'serif','font.size':11,'axes.grid':True,'grid.alpha':0.3,
    'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9,
    'axes.spines.top':False,'axes.spines.right':False})

pred = pd.read_csv(os.path.join(P, "data", "ml_predictions_for_milp.csv"),
                   parse_dates=['timestamp'])
y   = pred['net_load_actual'].values
lo_c, hi_c = pred['net_load_Q10'].values, pred['net_load_Q90'].values
lo_u, hi_u = lo_c + Q_HAT, hi_c - Q_HAT
inside = (y >= lo_c) & (y <= hi_c)

# =====================================================================
# 1) Conformal diagnostics
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                         gridspec_kw={'width_ratios': [1.5, 1]})

# (a) sample week: QR vs CQR bands
ax = axes[0]
sl = slice(0, 168)
t  = pred['timestamp'].iloc[sl]
ax.fill_between(t, lo_c[sl], hi_c[sl], alpha=0.25, color='#6A1B9A',
                label='CQR 90% PI (calibrated)')
ax.fill_between(t, lo_u[sl], hi_u[sl], alpha=0.35, color='#FF9800',
                label='QR interval (uncalibrated)')
ax.plot(t, y[sl], 'k-', lw=1.4, label='Actual net load')
miss = ~inside[sl.start:sl.stop]
ax.plot(t[miss], y[sl][miss], 'rx', ms=7, mew=2, label='Outside CQR PI')
ax.set_ylabel('Net Load (MW)')
ax.set_title('(a) Uncalibrated QR vs. CQR Intervals (first test week)',
             fontweight='bold')
ax.legend(loc='upper left', ncol=2)
ax.tick_params(axis='x', rotation=30)

# (b) conditional coverage by hour of day
ax = axes[1]
hr = pred['timestamp'].dt.hour
cov_hr = pd.Series(inside).groupby(hr).mean() * 100
colors = ['#B71C1C' if c < 85 else '#6A1B9A' for c in cov_hr]
ax.bar(cov_hr.index, cov_hr.values, color=colors, alpha=0.85,
       edgecolor='black', lw=0.4)
ax.axhline(90, color='black', ls='--', lw=1.2, label='90% target')
ax.axhline(inside.mean()*100, color='#2E7D32', ls=':', lw=1.5,
           label=f'Marginal ({inside.mean()*100:.1f}%)')
ax.axvspan(15.5, 21.5, alpha=0.10, color='red')
ax.text(18.5, 99, 'evening\nramp', ha='center', fontsize=9, color='#B71C1C')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Empirical Coverage (%)')
ax.set_ylim(70, 103)
ax.set_title('(b) Conditional Coverage by Hour', fontweight='bold')
ax.legend(loc='lower left')

plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig_conformal_diagnostics.png"))
plt.close()
print("-> fig_conformal_diagnostics.png")
print(f"   coverage by hour: min {cov_hr.min():.1f}% (h={cov_hr.idxmin()}), "
      f"max {cov_hr.max():.1f}%")

# =====================================================================
# 2) Import price sensitivity
# =====================================================================
imp = pd.read_csv(os.path.join(TBL, "sensitivity_import_price.csv"))
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(imp['Import_Cost_MWh'], imp['S1_Cost_MWh'], 'o-', color='#4CAF50',
        lw=2, ms=8, label='S1 (with nuclear)')
ax.plot(imp['Import_Cost_MWh'], imp['S4_Cost_MWh'], 's-', color='#E53935',
        lw=2, ms=8, label='S4 (no nuclear)')
ax.fill_between(imp['Import_Cost_MWh'], imp['S1_Cost_MWh'],
                imp['S4_Cost_MWh'], alpha=0.12, color='#E53935')
for _, r in imp.iterrows():
    ax.annotate(f"+{r['Premium_pct']:.1f}%",
                ((r['Import_Cost_MWh']), (r['S1_Cost_MWh']+r['S4_Cost_MWh'])/2),
                ha='center', fontsize=9, color='#B71C1C')
ax.axvline(55, color='gray', ls=':', alpha=0.7, label='Baseline ($55/MWh)')
ax.set_xlabel('Import Price ($/MWh)')
ax.set_ylabel('System Cost ($/MWh)')
ax.set_title('Nuclear Premium vs. Import Price (November week)',
             fontweight='bold')
ax.legend(loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig_import_sensitivity.png"))
plt.close()
print("-> fig_import_sensitivity.png")
