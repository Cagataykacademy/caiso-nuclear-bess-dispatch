"""
=============================================================================
 PHASE 7: FULL-YEAR (50-WEEK) DISPATCH ANALYSIS
 Addresses the "single test week" concern: the improved two-tier UC MILP is
 solved for every consecutive 168-hour window of realized 2023 net load,
 with and without nuclear -> distribution of costs and nuclear premium.

 Weekly independence: each window starts at SoC = 50% (no inter-week
 coupling); end-of-horizon SoC >= 35% prevents end-game depletion, so
 windows are conservative, self-contained dispatch problems.

 Outputs:
   - outputs/tables/full_year_dispatch.csv
   - outputs/tables/full_year_summary.csv
   - outputs/figures/fig_fullyear_dispatch.png
=============================================================================
"""
import os, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P    = os.path.dirname(HERE)
TBL  = os.path.join(P, "outputs", "tables")
FIG  = os.path.join(P, "outputs", "figures")

spec = importlib.util.spec_from_file_location(
    "improved_milp", os.path.join(HERE, "16_improved_milp.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

# Annual sweep: lift the load-shedding cap so extreme heat weeks price
# scarcity through VOLL instead of going infeasible. The stylized 37.3 GW
# fleet omits CAISO hydro (~6-8 GW) and emergency mechanisms, so weeks with
# shed > 0 under nuclear are flagged as "scarcity weeks" and analysed
# separately from normal weeks.
M.SHED_CAP = 20000

import matplotlib.pyplot as plt

EF_CCGT, EF_PEAK, EF_IMP = 0.37, 0.55, 0.428

def co2_of(d):
    return (np.sum(d['P_ccgt'])*EF_CCGT + np.sum(d['P_peak'])*EF_PEAK
            + np.sum(d['P_imp'])*EF_IMP)

df = pd.read_csv(os.path.join(P, "data", "caiso_preprocessed_v2_2023.csv"))
df['period'] = pd.to_datetime(df['period'])
nl_all = df['net_load_MW'].values
ts_all = df['period'].values
H = 168
n_weeks = len(nl_all) // H
print(f"Full-year dispatch: {n_weeks} consecutive weeks x 2 scenarios "
      f"({2*n_weeks} MILP solves)\n")

rows = []
for w in range(n_weeks):
    nl = nl_all[w*H:(w+1)*H]
    start = pd.Timestamp(ts_all[w*H])
    _, d1 = M.build_and_solve(nl, f"W{w}_nuc", use_nuc=True)
    _, d0 = M.build_and_solve(nl, f"W{w}_nonuc", use_nuc=False)
    if d1 is None or d0 is None:
        print(f"  week {w:2d} {start.date()}  INFEASIBLE"); continue
    prem = (d0['cost_per_mwh'] - d1['cost_per_mwh']) / d1['cost_per_mwh'] * 100
    rows.append({
        'Week': w+1, 'Start': start.date(),
        'NL_Mean_MW': round(nl.mean(), 0), 'NL_Min_MW': round(nl.min(), 0),
        'NL_Max_MW': round(nl.max(), 0),
        'Cost_Nuclear_MWh': round(d1['cost_per_mwh'], 2),
        'Cost_NoNuclear_MWh': round(d0['cost_per_mwh'], 2),
        'Premium_pct': round(prem, 1),
        'Shed_Nuclear_MWh': round(d1['shed_total'], 0),
        'Shed_NoNuclear_MWh': round(d0['shed_total'], 0),
        'Peaker_Nuclear_MW': round(d1['avg_peak'], 0),
        'Peaker_NoNuclear_MW': round(d0['avg_peak'], 0),
        'CO2_Nuclear_t': round(co2_of(d1), 0),
        'CO2_NoNuclear_t': round(co2_of(d0), 0),
        'BESS_Cycles': round(d1['bess_cycles'], 2)})
    print(f"  week {w+1:2d} {start.date()}  nuc=${d1['cost_per_mwh']:6.2f}  "
          f"no-nuc=${d0['cost_per_mwh']:6.2f}  prem=+{prem:4.1f}%  "
          f"shed={d0['shed_total']:5.0f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TBL, "full_year_dispatch.csv"), index=False)

# ---- annual summary (normal vs scarcity weeks) ----------------------------
res['Scarcity'] = res['Shed_NoNuclear_MWh'] > 0
norm  = res[~res['Scarcity']]
scarc = res[res['Scarcity']]
co2_saving_annual = (res['CO2_NoNuclear_t'] - res['CO2_Nuclear_t']).sum() \
                    * (52 / len(res)) / 1e6
summ = {
    'Weeks_Solved': len(res),
    'Weeks_Normal': len(norm),
    'Weeks_Scarcity': len(scarc),
    'Cost_Nuclear_Mean': res['Cost_Nuclear_MWh'].mean(),
    'Cost_Nuclear_Min': res['Cost_Nuclear_MWh'].min(),
    'Cost_Nuclear_Max': res['Cost_Nuclear_MWh'].max(),
    'Premium_Median_pct': res['Premium_pct'].median(),
    'Premium_Normal_Mean_pct': norm['Premium_pct'].mean(),
    'Premium_Normal_Min_pct': norm['Premium_pct'].min(),
    'Premium_Normal_Max_pct': norm['Premium_pct'].max(),
    'Premium_Scarcity_Mean_pct': scarc['Premium_pct'].mean() if len(scarc) else 0,
    'Premium_Scarcity_Max_pct': scarc['Premium_pct'].max() if len(scarc) else 0,
    'Weeks_Shed_WithNuclear': int((res['Shed_Nuclear_MWh'] > 0).sum()),
    'Total_Shed_NoNuclear_MWh': res['Shed_NoNuclear_MWh'].sum(),
    'CO2_Saving_Annualized_Mt': round(co2_saving_annual, 2),
}
pd.DataFrame([summ]).round(2).to_csv(
    os.path.join(TBL, "full_year_summary.csv"), index=False)

print("\nANNUAL SUMMARY")
for k, v in summ.items():
    print(f"  {k:<28} {v:,.2f}" if isinstance(v, float) else f"  {k:<28} {v}")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                         gridspec_kw={'height_ratios': [1.4, 1]})
x = pd.to_datetime(res['Start'])

ax = axes[0]
ax.plot(x, res['Cost_Nuclear_MWh'], 'o-', color='#4CAF50', lw=1.8, ms=5,
        label='With nuclear (2,256 MW)')
ax.plot(x, res['Cost_NoNuclear_MWh'], 's-', color='#E53935', lw=1.8, ms=5,
        label='No nuclear')
ax.fill_between(x, res['Cost_Nuclear_MWh'], res['Cost_NoNuclear_MWh'],
                alpha=0.12, color='#E53935')
ax.set_ylabel('Weekly Dispatch Cost ($/MWh)')
ax.set_title('Full-Year Weekly Dispatch (50 consecutive weeks, realized 2023 net load)',
             fontweight='bold')
ax.legend(loc='upper left')

ax = axes[1]
cols = ['#FF8F00' if s else '#B71C1C' for s in res['Scarcity']]
ax.bar(x, res['Premium_pct'], width=6, color=cols, alpha=0.85,
       edgecolor='black', lw=0.3)
ax.axhline(norm['Premium_pct'].mean(), color='black', ls='--', lw=1.2,
           label=f"Normal-week mean (+{norm['Premium_pct'].mean():.1f}%)")
from matplotlib.patches import Patch
handles = [Patch(fc='#B71C1C', label='Normal week'),
           Patch(fc='#FF8F00', label='Scarcity week (shed > 0 without nuclear)'),
           ax.lines[0]]
ax.set_ylabel('Nuclear Removal Premium (%)')
ax.set_xlabel('Week Starting')
ax.set_title('Nuclear Cost Premium by Week', fontweight='bold')
ax.legend(handles=handles, loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig_fullyear_dispatch.png"))
plt.close()
print("\n-> full_year_dispatch.csv, full_year_summary.csv, fig_fullyear_dispatch.png")
