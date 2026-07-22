"""
=============================================================================
 PHASE 4: EVALUATION & PUBLICATION FIGURES
 Final analysis, sensitivity, and paper-ready visualizations
=============================================================================
"""

import os
import sys
import io
import warnings
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR  = os.path.join(PROJECT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.figsize': (14, 6),
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print("=" * 78)
print("  PHASE 4: EVALUATION & PUBLICATION FIGURES")
print("=" * 78)

# =====================================================================
#  LOAD ALL DATA
# =====================================================================
print("\n[1/5] Loading all project data...")

demand_df = pd.read_csv(os.path.join(DATA_DIR, "caiso_demand_2023.csv"))
demand_df['timestamp'] = pd.to_datetime(demand_df['period'])

processed_df = pd.read_csv(os.path.join(DATA_DIR, "caiso_preprocessed_2023.csv"))
processed_df['timestamp'] = pd.to_datetime(processed_df['period'])

preds = pd.read_csv(os.path.join(DATA_DIR, "ml_predictions_for_milp.csv"))
preds['timestamp'] = pd.to_datetime(preds['timestamp'])

ml_results = pd.read_csv(os.path.join(TABLE_DIR, "ml_results_summary.csv"))
scenario_comp = pd.read_csv(os.path.join(TABLE_DIR, "scenario_comparison.csv"))

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_2023.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val_2023.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_2023.csv"))

print(f"  Demand data: {len(demand_df)} records")
print(f"  Preprocessed: {len(processed_df)} records")
print(f"  Predictions: {len(preds)} records")
print(f"  Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

# =====================================================================
#  FIGURE 16: DUCK CURVE ANALYSIS
# =====================================================================
print("\n[2/5] Duck Curve analysis figure...")

demand_df['hour'] = pd.to_datetime(demand_df['period']).dt.hour
demand_df['month'] = pd.to_datetime(demand_df['period']).dt.month

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Duck curve by season
ax = axes[0]
seasons = {
    'Winter (Dec-Feb)': [12, 1, 2],
    'Spring (Mar-May)': [3, 4, 5],
    'Summer (Jun-Aug)': [6, 7, 8],
    'Fall (Sep-Nov)': [9, 10, 11],
}
colors_season = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

for (sname, months), color in zip(seasons.items(), colors_season):
    mask = demand_df['month'].isin(months)
    seasonal = demand_df[mask].groupby('hour').agg(
        demand=('total_demand_MW', 'mean'),
        net_gen=('net_demand_MW', 'mean')
    ).reset_index()

    ax.plot(seasonal['hour'], seasonal['demand'], '-', color=color, linewidth=2, label=f'{sname} Demand')
    ax.plot(seasonal['hour'], seasonal['net_gen'], '--', color=color, linewidth=1.5, alpha=0.6, label=f'{sname} Net Gen')

ax.fill_between(range(24), 14000, 18000, alpha=0.1, color='red', label='Duck Belly Zone')
ax.axvline(x=17, color='red', linestyle=':', alpha=0.5, label='Evening Ramp Start')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Power (MW)')
ax.set_title('CAISO Duck Curve by Season (2023)', fontweight='bold')
ax.legend(loc='upper left', fontsize=7, ncol=2)
ax.set_xlim(0, 23)

# Right: Net load shape showing the "duck"
ax = axes[1]
spring = demand_df[demand_df['month'].isin([3, 4, 5])]
spring_hourly = spring.groupby('hour').agg(
    demand=('total_demand_MW', 'mean'),
    net_gen=('net_demand_MW', 'mean'),
).reset_index()

ax.plot(spring_hourly['hour'], spring_hourly['demand'], 'b-', linewidth=2.5, label='Total Demand')
ax.plot(spring_hourly['hour'], spring_hourly['net_gen'], 'g-', linewidth=2.5, label='Net Generation')

duck_belly_hour = spring_hourly['net_gen'].idxmin()
ramp_start = spring_hourly.loc[duck_belly_hour:, 'net_gen'].values
ramp_end_val = spring_hourly['net_gen'].iloc[-6:]

ax.annotate('Duck Belly\n(Solar Displacement)',
            xy=(duck_belly_hour, spring_hourly['net_gen'].iloc[duck_belly_hour]),
            xytext=(duck_belly_hour - 4, spring_hourly['net_gen'].iloc[duck_belly_hour] - 3000),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', fontweight='bold')

ax.annotate('Evening Ramp\n(3-hour window)',
            xy=(18, spring_hourly['net_gen'].iloc[18]),
            xytext=(20, spring_hourly['net_gen'].iloc[18] + 3000),
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
            fontsize=10, color='orange', fontweight='bold')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Power (MW)')
ax.set_title('Spring Duck Curve Profile (Mar-May 2023)', fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(0, 23)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig16_duck_curve_analysis.png"))
plt.close()
print("  -> fig16_duck_curve_analysis.png")

# =====================================================================
#  FIGURE 17: ML PREDICTION INTERVALS (Full Test Period)
# =====================================================================
print("\n[3/5] ML prediction intervals figure...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Net load predictions with PI
ax = axes[0]
t_hours = np.arange(len(preds))

ax.fill_between(t_hours, preds['net_load_Q10'], preds['net_load_Q90'],
                alpha=0.25, color='#2196F3', label='80% Prediction Interval')
ax.plot(t_hours, preds['net_load_actual'], 'k-', linewidth=0.8, alpha=0.7, label='Actual')
ax.plot(t_hours, preds['net_load_predicted'], 'r-', linewidth=0.8, alpha=0.7, label='Predicted (Point)')

ax.set_ylabel('Net Load (MW)')
ax.set_title('Net Load: Conformalized Quantile Regression (Full Test Period)', fontweight='bold')
ax.legend(loc='upper right')

net_load_row = ml_results[ml_results['Target'] == 'net_load_MW'].iloc[0]
textstr = f"MAE: {net_load_row['MAE']:.1f} MW | R²: {net_load_row['R2']:.4f} | 90% Coverage: {net_load_row['90% Coverage']:.1f}%"
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Price predictions with PI
ax = axes[1]
ax.fill_between(t_hours, preds['price_Q10'], preds['price_Q90'],
                alpha=0.25, color='#FF9800', label='80% Prediction Interval')
ax.plot(t_hours, preds['price_actual'], 'k-', linewidth=0.8, alpha=0.7, label='Actual')
ax.plot(t_hours, preds['price_predicted'], 'r-', linewidth=0.8, alpha=0.7, label='Predicted (Point)')

ax.set_ylabel('Price Proxy ($/MWh)')
ax.set_xlabel('Test Period Hour Index')
ax.set_title('Price Proxy: Conformalized Quantile Regression (Full Test Period)', fontweight='bold')
ax.legend(loc='upper right')

price_row = ml_results[ml_results['Target'] == 'price_proxy_USD_MWh'].iloc[0]
textstr = f"MAE: {price_row['MAE']:.1f} $/MWh | R²: {price_row['R2']:.4f} | 90% Coverage: {price_row['90% Coverage']:.1f}%"
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig17_prediction_intervals.png"))
plt.close()
print("  -> fig17_prediction_intervals.png")

# =====================================================================
#  FIGURE 18: COMPREHENSIVE SCENARIO DASHBOARD
# =====================================================================
print("\n[4/5] Scenario dashboard figure...")

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

# Scenario data
sc = scenario_comp.copy()
sc_names = [s.replace('_', '\n') for s in sc['Scenario']]
colors_sc = ['#4CAF50', '#E53935', '#2196F3', '#FF9800', '#9C27B0', '#795548']

# (0,0) Total cost bar
ax = fig.add_subplot(gs[0, 0])
bars = ax.barh(range(len(sc)), sc['Total Cost ($)'] / 1e6, color=colors_sc, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sc)))
ax.set_yticklabels(sc_names, fontsize=8)
ax.set_xlabel('Total Cost (Million $)')
ax.set_title('Total System Cost', fontweight='bold')
for bar, val in zip(bars, sc['Total Cost ($)'] / 1e6):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'${val:.1f}M', va='center', fontsize=8, fontweight='bold')

# (0,1) Cost per MWh
ax = fig.add_subplot(gs[0, 1])
bars = ax.barh(range(len(sc)), sc['Cost/MWh ($/MWh)'], color=colors_sc, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sc)))
ax.set_yticklabels(sc_names, fontsize=8)
ax.set_xlabel('Unit Cost ($/MWh)')
ax.set_title('Cost per MWh', fontweight='bold')
for bar, val in zip(bars, sc['Cost/MWh ($/MWh)']):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'${val:.1f}', va='center', fontsize=8, fontweight='bold')

# (0,2) Generation mix pie (S1 Deterministic)
ax = fig.add_subplot(gs[0, 2])
s1 = sc[sc['Scenario'] == 'S1_Deterministic'].iloc[0]
mix_vals = [s1['Avg Nuclear (MW)'], s1['Avg Gas (MW)'], s1['Avg Import (MW)']]
mix_labels = ['Nuclear', 'Gas', 'Import']
mix_colors = ['#4CAF50', '#FF9800', '#9C27B0']
result = ax.pie(mix_vals, labels=mix_labels, colors=mix_colors,
                autopct='%1.1f%%', startangle=90, pctdistance=0.85)
for t in result[2]:
    t.set_fontsize(9)
    t.set_fontweight('bold')
ax.set_title('Generation Mix (S1)', fontweight='bold')

# (1,0) BESS cycles comparison
ax = fig.add_subplot(gs[1, 0])
bars = ax.barh(range(len(sc)), sc['BESS Cycles'], color=colors_sc, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sc)))
ax.set_yticklabels(sc_names, fontsize=8)
ax.set_xlabel('BESS Cycles (weekly)')
ax.set_title('Battery Utilization', fontweight='bold')

# (1,1) Nuclear vs Gas contribution
ax = fig.add_subplot(gs[1, 1])
x = np.arange(len(sc))
w = 0.35
ax.bar(x - w/2, sc['Avg Nuclear (MW)'], w, color='#4CAF50', alpha=0.8, label='Nuclear')
ax.bar(x + w/2, sc['Avg Gas (MW)'], w, color='#FF9800', alpha=0.8, label='Gas')
ax.set_xticks(x)
ax.set_xticklabels(sc_names, fontsize=7)
ax.set_ylabel('Average Power (MW)')
ax.set_title('Nuclear vs Gas by Scenario', fontweight='bold')
ax.legend()

# (1,2) Cost breakdown delta from baseline
ax = fig.add_subplot(gs[1, 2])
baseline_cost = sc.loc[sc['Scenario'] == 'S1_Deterministic', 'Total Cost ($)'].values[0]
deltas = (sc['Total Cost ($)'] - baseline_cost) / 1e6
bar_colors = ['green' if d <= 0 else 'red' for d in deltas]
bars = ax.barh(range(len(sc)), deltas, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(range(len(sc)))
ax.set_yticklabels(sc_names, fontsize=8)
ax.set_xlabel('Cost Delta from S1 (Million $)')
ax.set_title('Cost Impact vs Deterministic', fontweight='bold')
for bar, val in zip(bars, deltas):
    sign = '+' if val > 0 else ''
    ax.text(bar.get_width() + 0.1 if val >= 0 else bar.get_width() - 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{sign}${val:.1f}M', va='center', fontsize=8, fontweight='bold')

fig.suptitle('MILP Dispatch Optimization: 6-Scenario Comparison Dashboard',
             fontsize=15, fontweight='bold', y=1.01)

fig.savefig(os.path.join(FIG_DIR, "fig18_scenario_dashboard.png"))
plt.close()
print("  -> fig18_scenario_dashboard.png")

# =====================================================================
#  FIGURE 19: END-TO-END PIPELINE SUMMARY
# =====================================================================
print("\n[5/5] Pipeline summary figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (0,0) Data overview - demand time series with train/val/test split
ax = axes[0, 0]
processed_df['month'] = pd.to_datetime(processed_df['timestamp']).dt.month

n_train = len(train_df)
n_val = len(val_df)
n_test = len(test_df)
total = n_train + n_val + n_test

ax.axvspan(0, n_train, alpha=0.1, color='blue', label=f'Train ({n_train}h)')
ax.axvspan(n_train, n_train + n_val, alpha=0.1, color='orange', label=f'Val ({n_val}h)')
ax.axvspan(n_train + n_val, total, alpha=0.1, color='green', label=f'Test ({n_test}h)')

if 'net_load_MW' in processed_df.columns:
    ax.plot(range(total), processed_df['net_load_MW'].iloc[:total], 'k-', linewidth=0.3, alpha=0.5)
elif 'demand' in demand_df.columns:
    ax.plot(range(min(total, len(demand_df))), demand_df['demand'].iloc[:min(total, len(demand_df))], 'k-', linewidth=0.3, alpha=0.5)

ax.set_xlabel('Hour Index')
ax.set_ylabel('Net Load (MW)')
ax.set_title('Data Split: Train / Validation / Test', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)

# (0,1) ML Model performance scatter
ax = axes[0, 1]
ax.scatter(preds['net_load_actual'], preds['net_load_predicted'], s=3, alpha=0.3, c='#2196F3')
lims = [preds['net_load_actual'].min(), preds['net_load_actual'].max()]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Net Load (MW)')
ax.set_ylabel('Predicted Net Load (MW)')
ax.set_title(f'ML Model: Actual vs Predicted (R² = {net_load_row["R2"]:.4f})', fontweight='bold')
ax.legend()

# (1,0) Scenario comparison radar-like grouped bar
ax = axes[1, 0]
metrics = ['Cost/MWh ($/MWh)', 'Avg Gas (MW)', 'Avg Import (MW)']
x = np.arange(len(metrics))
width = 0.12

for i, (_, row) in enumerate(sc.iterrows()):
    vals = [row[m] for m in metrics]
    vals_norm = [v / sc[m].max() * 100 for v, m in zip(vals, metrics)]
    ax.bar(x + i * width, vals_norm, width, color=colors_sc[i], alpha=0.8,
           label=row['Scenario'].replace('_', ' '))

ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(['Unit Cost', 'Gas Usage', 'Import'], fontsize=9)
ax.set_ylabel('Normalized Value (%)')
ax.set_title('Scenario Metric Comparison (Normalized)', fontweight='bold')
ax.legend(fontsize=6, loc='upper right', ncol=2)

# (1,1) Key findings summary table
ax = axes[1, 1]
ax.axis('off')

s1_cost = sc.loc[sc['Scenario'] == 'S1_Deterministic', 'Total Cost ($)'].values[0]
s4_cost = sc.loc[sc['Scenario'] == 'S4_No_Nuclear', 'Total Cost ($)'].values[0]
nuclear_impact = (s4_cost - s1_cost) / s1_cost * 100

findings = [
    ['Metric', 'Value'],
    ['Dataset', 'CAISO 2023 (8,760 hours)'],
    ['ML Model', 'LightGBM + CQR'],
    ['Net Load MAE', f'{net_load_row["MAE"]:.1f} MW'],
    ['Net Load R²', f'{net_load_row["R2"]:.4f}'],
    ['90% PI Coverage', f'{net_load_row["90% Coverage"]:.1f}%'],
    ['Optimization', '168h MILP (6 scenarios)'],
    ['Optimal Cost/MWh', f'${sc["Cost/MWh ($/MWh)"].min():.2f}'],
    ['Nuclear Value', f'+{nuclear_impact:.1f}% cost w/o nuclear'],
    ['Load Shedding', f'{sc["Load Shed (MWh)"].sum():.0f} MWh (all scenarios)'],
    ['BESS Utilization', f'{sc["BESS Cycles"].mean():.1f} cycles/week avg'],
]

table = ax.table(cellText=findings[1:], colLabels=findings[0],
                 loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f0f0f0')
    cell.set_edgecolor('#cccccc')

ax.set_title('Key Results Summary', fontweight='bold', pad=20)

fig.suptitle('End-to-End Pipeline: Data → ML → Optimization',
             fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig19_pipeline_summary.png"))
plt.close()
print("  -> fig19_pipeline_summary.png")

# =====================================================================
#  COMPREHENSIVE RESULTS TABLE
# =====================================================================
print("\n" + "=" * 78)
print("  Generating comprehensive results table...")

results_summary = {
    'Phase': ['Data Acquisition', 'EDA', 'Preprocessing', 'ML Modeling', 'ML Modeling', 'MILP Optimization', 'MILP Optimization'],
    'Metric': ['Total Records', 'Demand Range (MW)', 'Features Generated',
               'Net Load MAE (MW)', 'Net Load R²',
               'Optimal Cost ($/MWh)', 'Nuclear Cost Savings (%)'],
    'Value': [
        f'{len(demand_df):,}',
        f'{demand_df["total_demand_MW"].min():,.0f} - {demand_df["total_demand_MW"].max():,.0f}',
        f'{len(train_df.columns)}',
        f'{net_load_row["MAE"]:.2f}',
        f'{net_load_row["R2"]:.4f}',
        f'{sc["Cost/MWh ($/MWh)"].min():.2f}',
        f'{nuclear_impact:.1f}%',
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(TABLE_DIR, "comprehensive_results.csv"), index=False)
print(f"  -> comprehensive_results.csv")

# =====================================================================
#  FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 78)
print("  PHASE 4 COMPLETE: All Figures & Tables Generated")
print("=" * 78)

print("\n  ALL OUTPUT FILES:")
print(f"\n  Figures ({FIG_DIR}):")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'):
        fpath = os.path.join(FIG_DIR, f)
        print(f"    {f:45s} ({os.path.getsize(fpath)/1024:>6.0f} KB)")

print(f"\n  Tables ({TABLE_DIR}):")
for f in sorted(os.listdir(TABLE_DIR)):
    if f.endswith('.csv'):
        fpath = os.path.join(TABLE_DIR, f)
        print(f"    {f:45s} ({os.path.getsize(fpath)/1024:>6.0f} KB)")

total_figs = len([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
total_tables = len([f for f in os.listdir(TABLE_DIR) if f.endswith('.csv')])

print(f"\n  TOTALS: {total_figs} figures, {total_tables} tables")
print("\n" + "=" * 78)
print("  PROJECT PIPELINE COMPLETE!")
print("  Ready for paper writing.")
print("=" * 78)
