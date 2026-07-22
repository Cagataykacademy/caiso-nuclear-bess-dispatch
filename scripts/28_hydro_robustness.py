"""
=============================================================================
 PHASE 8: HYDRO ROBUSTNESS CHECK FOR THE 15 SCARCITY WEEKS
 Reviewer concern: the annual-sweep fleet (Section 4.12) omits CAISO's real
 hydroelectric generation, so summer shedding "with nuclear" could be an
 artifact of a missing supply resource rather than a genuine finding.

 Test: re-run the 15 flagged scarcity weeks with realized 2023 hourly CAISO
 hydro generation (EIA fuel-type series, WAT) added as a must-take resource
 (treated like solar/wind: subtracted from net load before dispatch, since
 CAISO hydro is largely run-of-river / environmentally constrained and is
 dispatched ahead of thermal in the actual merit order). This requires NO
 change to the core MILP (16_improved_milp.py) or any of its established
 results elsewhere in the paper -- it only modifies the net load PROFILE fed
 into the unmodified solver for this one targeted check.

 Outputs:
   - outputs/tables/hydro_robustness.csv
   - outputs/figures/fig_hydro_robustness.png
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
M.SHED_CAP = 20000   # match the annual-sweep cap so scarcity is priced, not infeasible

import matplotlib.pyplot as plt

# --- load realized 2023 net load (as used in the annual sweep) ------------
df = pd.read_csv(os.path.join(P, "data", "caiso_preprocessed_v2_2023.csv"))
df['period'] = pd.to_datetime(df['period'])
nl_all = df['net_load_MW'].values
ts_all = df['period'].values
H = 168

# --- load realized 2023 hourly hydro (WAT) generation ----------------------
gen = pd.read_csv(os.path.join(P, "data", "caiso_generation_by_fuel_2023.csv"))
wat = gen[gen['fueltype'] == 'WAT'].copy()
wat['period'] = pd.to_datetime(wat['period'])
wat = wat.drop_duplicates(subset='period').set_index('period')['value']

# --- identify the 15 scarcity weeks from the existing annual sweep --------
full = pd.read_csv(os.path.join(TBL, "full_year_dispatch.csv"))
full['Start'] = pd.to_datetime(full['Start'])
scarcity = full[full['Shed_NoNuclear_MWh'] > 0].copy()
print(f"Re-testing {len(scarcity)} scarcity weeks with realized hydro added "
      f"as a must-take resource (with nuclear, matching the paper's "
      f"preferred/baseline configuration)...\n")

rows = []
for _, row in scarcity.iterrows():
    start = row['Start']
    end_ts = start + pd.Timedelta(hours=H - 1)
    mask = (pd.Series(ts_all) >= start) & (pd.Series(ts_all) <= end_ts)
    idx = np.where(mask.values)[0]
    if len(idx) != H:
        print(f"  {start.date()}  SKIPPED (incomplete window, n={len(idx)})")
        continue
    nl_week = nl_all[idx]
    ts_week = pd.to_datetime(ts_all[idx])

    hydro_week = wat.reindex(ts_week).values
    if np.isnan(hydro_week).any():
        hydro_week = np.nan_to_num(hydro_week, nan=np.nanmean(hydro_week))

    nl_with_hydro = nl_week - hydro_week   # hydro dispatched must-take, ahead of thermal

    _, d_before = M.build_and_solve(nl_week, f"{start.date()}_orig", use_nuc=True)
    _, d_after  = M.build_and_solve(nl_with_hydro, f"{start.date()}_hydro", use_nuc=True)

    rows.append({
        'Week_Start': start.date(),
        'Hydro_Mean_MW': round(hydro_week.mean(), 0),
        'Hydro_Max_MW': round(hydro_week.max(), 0),
        'Cost_Before_MWh': round(d_before['cost_per_mwh'], 2),
        'Cost_After_MWh': round(d_after['cost_per_mwh'], 2),
        'Shed_Before_MWh': round(d_before['shed_total'], 0),
        'Shed_After_MWh': round(d_after['shed_total'], 0),
        'Shed_Eliminated_pct': round(
            100 * (1 - d_after['shed_total'] / d_before['shed_total']), 1)
            if d_before['shed_total'] > 0 else 100.0,
    })
    print(f"  {start.date()}  shed {d_before['shed_total']:7.0f} -> "
          f"{d_after['shed_total']:7.0f} MWh   "
          f"(hydro avail. mean={hydro_week.mean():.0f} MW)")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TBL, "hydro_robustness.csv"), index=False)

total_before = res['Shed_Before_MWh'].sum()
total_after = res['Shed_After_MWh'].sum()
n_fully_resolved = int((res['Shed_After_MWh'] == 0).sum())
print(f"\nSUMMARY")
print(f"  Total shedding (with nuclear), 15 scarcity weeks:")
print(f"    Before realized hydro: {total_before:,.0f} MWh")
print(f"    After realized hydro:  {total_after:,.0f} MWh")
print(f"    Reduction: {100*(1 - total_after/total_before):.1f}%")
print(f"  Weeks fully resolved (zero shedding after hydro): "
      f"{n_fully_resolved} / {len(res)}")

# --- figure -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(res))
w = 0.38
ax.bar(x - w/2, res['Shed_Before_MWh'], width=w, color='#B71C1C', alpha=0.85,
       label='Without hydro (stylized annual-sweep fleet)', edgecolor='black', lw=0.3)
ax.bar(x + w/2, res['Shed_After_MWh'], width=w, color='#2E7D32', alpha=0.85,
       label='With realized 2023 hydro added', edgecolor='black', lw=0.3)
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in res['Week_Start']], rotation=45, ha='right')
ax.set_ylabel('Load Shedding, With Nuclear (MWh)')
ax.set_title('Effect of Realized CAISO Hydro Generation on Scarcity-Week Shedding\n'
             f'(15 weeks; total shedding reduced {100*(1-total_after/total_before):.0f}%, '
             f'{n_fully_resolved}/{len(res)} weeks fully resolved)',
             fontweight='bold')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig_hydro_robustness.png"))
plt.close()
print("\n-> hydro_robustness.csv, fig_hydro_robustness.png")
