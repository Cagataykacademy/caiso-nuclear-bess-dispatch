"""
Preliminary EDA on CAISO demand data while generation data downloads.
Generates initial Duck Curve analysis and data quality report.
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
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIG_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
TABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Plot style
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
print("  EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 78)

# =====================================================================
#  LOAD DATA
# =====================================================================
unified_path = os.path.join(DATA_DIR, "caiso_unified_2023.csv")
df = pd.read_csv(unified_path, index_col=0, parse_dates=True)

print(f"\n  Dataset loaded: {df.shape}")
print(f"  Date range: {df.index.min()} -> {df.index.max()}")
print(f"  Columns: {list(df.columns)}")

# Rename for clarity
rename_map = {}
for col in df.columns:
    cl = col.lower()
    if col == 'net_demand_MW':
        rename_map[col] = 'net_generation_MW'  # EIA calls it "Net generation"
df = df.rename(columns=rename_map)

# =====================================================================
#  TEMPORAL FEATURES
# =====================================================================
print("\n  Engineering temporal features...")
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['day_of_year'] = df.index.dayofyear
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}
df['season'] = df['month'].map(season_map)

# =====================================================================
#  SECTION 1: DATA QUALITY REPORT
# =====================================================================
print("\n" + "-" * 78)
print("  SECTION 1: DATA QUALITY REPORT")
print("-" * 78)

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n  Total records: {len(df):,}")
print(f"  Total features: {len(df.columns)}")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

print(f"\n  {'Column':<40} {'Type':<10} {'Non-Null':>8} {'Missing':>8} {'%':>6}")
print(f"  {'---'*25}")
for col in df.columns:
    dtype = str(df[col].dtype)[:8]
    non_null = df[col].notna().sum()
    missing = df[col].isna().sum()
    pct = missing / len(df) * 100
    print(f"  {col:<40} {dtype:<10} {non_null:>8,} {missing:>8,} {pct:>5.1f}%")

# Descriptive stats
print(f"\n  DESCRIPTIVE STATISTICS:")
desc = df[numeric_cols].describe().T
desc['skewness'] = df[numeric_cols].skew()
desc['kurtosis'] = df[numeric_cols].kurtosis()
print(desc.to_string())
desc.to_csv(os.path.join(TABLE_DIR, "descriptive_statistics.csv"))

# =====================================================================
#  SECTION 2: DEMAND PATTERN ANALYSIS
# =====================================================================
print(f"\n" + "-" * 78)
print("  SECTION 2: DEMAND PATTERN ANALYSIS")
print("-" * 78)

demand_col = 'total_demand_MW'
if demand_col in df.columns:
    d = df[demand_col]
    print(f"\n  Total Demand (MW):")
    print(f"    Mean      : {d.mean():>10,.0f} MW")
    print(f"    Std Dev   : {d.std():>10,.0f} MW")
    print(f"    Min       : {d.min():>10,.0f} MW ({d.idxmin()})")
    print(f"    Max       : {d.max():>10,.0f} MW ({d.idxmax()})")
    print(f"    CV        : {d.std()/d.mean()*100:.1f}%")
    
    # Hourly profile
    hourly = df.groupby('hour')[demand_col].agg(['mean', 'std', 'min', 'max'])
    peak_hour = hourly['mean'].idxmax()
    valley_hour = hourly['mean'].idxmin()
    ramp = hourly['mean'].max() - hourly['mean'].min()
    print(f"    Peak hour : {peak_hour}:00 ({hourly['mean'].max():,.0f} MW)")
    print(f"    Valley hr : {valley_hour}:00 ({hourly['mean'].min():,.0f} MW)")
    print(f"    Avg ramp  : {ramp:,.0f} MW (valley -> peak)")

# =====================================================================
#  SECTION 3: VISUALIZATIONS
# =====================================================================
print(f"\n" + "-" * 78)
print("  SECTION 3: GENERATING VISUALIZATIONS")
print("-" * 78)

# --- Fig 1: Demand Time Series Overview ---
print("  Generating Fig 1: Time series overview...")
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

daily = df.resample('D').mean(numeric_only=True)

colors = {'demand': '#5E35B1', 'net_gen': '#E53935', 'interchange': '#00ACC1'}

if 'total_demand_MW' in daily.columns:
    ax = axes[0]
    ax.plot(daily.index, daily['total_demand_MW'], color=colors['demand'], linewidth=0.8)
    ax.fill_between(daily.index, 0, daily['total_demand_MW'], alpha=0.15, color=colors['demand'])
    ax.set_ylabel('Total Demand\n(MW)')
    ax.set_title('CAISO Total Demand - Daily Average (2023)', fontweight='bold', loc='left')

if 'net_generation_MW' in daily.columns:
    ax = axes[1]
    ax.plot(daily.index, daily['net_generation_MW'], color=colors['net_gen'], linewidth=0.8)
    ax.fill_between(daily.index, 0, daily['net_generation_MW'], alpha=0.15, color=colors['net_gen'])
    ax.set_ylabel('Net Generation\n(MW)')
    ax.set_title('CAISO Net Generation - Daily Average (2023)', fontweight='bold', loc='left')

if 'Total interchange' in daily.columns:
    ax = axes[2]
    ax.plot(daily.index, daily['Total interchange'], color=colors['interchange'], linewidth=0.8)
    ax.fill_between(daily.index, 0, daily['Total interchange'], alpha=0.15, color=colors['interchange'])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Interchange\n(MW)')
    ax.set_title('CAISO Total Interchange - Daily Average (2023)', fontweight='bold', loc='left')

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].set_xlabel('Date (2023)')

fig.suptitle('CAISO Grid Overview - 2023', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_timeseries_overview.png"))
plt.close()
print(f"    -> Saved: fig1_timeseries_overview.png")

# --- Fig 2: Hourly Demand Profile by Season ---
print("  Generating Fig 2: Seasonal demand profiles...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
season_colors = ['#1565C0', '#2E7D32', '#E65100', '#BF360C']

for idx, (season, color) in enumerate(zip(seasons, season_colors)):
    ax = axes[idx // 2, idx % 2]
    season_data = df[df['season'] == season]
    
    if season_data.empty:
        ax.set_title(f'{season} (No Data)')
        continue
    
    hourly = season_data.groupby('hour')[demand_col]
    mean = hourly.mean()
    p10 = hourly.quantile(0.10)
    p90 = hourly.quantile(0.90)
    
    # Individual days
    for date, grp in season_data.groupby(season_data.index.date):
        if len(grp) >= 20:
            ax.plot(grp['hour'].values, grp[demand_col].values, 
                    color=color, alpha=0.02, linewidth=0.5)
    
    ax.fill_between(mean.index, p10, p90, alpha=0.25, color=color, label='P10-P90')
    ax.plot(mean.index, mean, color=color, linewidth=2.5, label='Mean')
    
    ax.set_title(f'{season}', fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Total Demand (MW)')
    ax.set_xlim(0, 23)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(3))

fig.suptitle('CAISO Hourly Demand Profile by Season (2023)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_demand_seasonal.png"))
plt.close()
print(f"    -> Saved: fig2_demand_seasonal.png")

# --- Fig 3: Weekday vs Weekend Demand ---
print("  Generating Fig 3: Weekday vs Weekend...")
fig, ax = plt.subplots(figsize=(12, 6))

weekday_hourly = df[df['is_weekend'] == 0].groupby('hour')[demand_col]
weekend_hourly = df[df['is_weekend'] == 1].groupby('hour')[demand_col]

wd_mean = weekday_hourly.mean()
we_mean = weekend_hourly.mean()

ax.plot(wd_mean.index, wd_mean, color='#1565C0', linewidth=2.5, label='Weekday', marker='o', markersize=4)
ax.fill_between(wd_mean.index, weekday_hourly.quantile(0.25), weekday_hourly.quantile(0.75),
                alpha=0.15, color='#1565C0')
ax.plot(we_mean.index, we_mean, color='#E65100', linewidth=2.5, label='Weekend', marker='s', markersize=4)
ax.fill_between(we_mean.index, weekend_hourly.quantile(0.25), weekend_hourly.quantile(0.75),
                alpha=0.15, color='#E65100')

ax.set_title('CAISO Demand: Weekday vs Weekend (2023)', fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Total Demand (MW)')
ax.set_xlim(0, 23)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.legend(fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_weekday_weekend.png"))
plt.close()
print(f"    -> Saved: fig3_weekday_weekend.png")

# --- Fig 4: Monthly Distribution Boxplot ---
print("  Generating Fig 4: Monthly demand distribution...")
fig, ax = plt.subplots(figsize=(14, 6))
month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

data_by_month = [df[df['month'] == m][demand_col].values for m in range(1, 13)]
bp = ax.boxplot(data_by_month, tick_labels=[month_names[m] for m in range(1, 13)],
                patch_artist=True, showfliers=False, widths=0.7)

cmap = plt.cm.RdYlBu_r
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(cmap(i / 11))
    patch.set_alpha(0.7)

ax.set_title('CAISO Monthly Demand Distribution (2023)', fontweight='bold')
ax.set_ylabel('Total Demand (MW)')
ax.set_xlabel('Month')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_monthly_boxplot.png"))
plt.close()
print(f"    -> Saved: fig4_monthly_boxplot.png")

# --- Fig 5: Demand vs Net Generation Scatter ---
print("  Generating Fig 5: Demand vs Net Generation...")
if 'net_generation_MW' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['total_demand_MW'], df['net_generation_MW'],
                        c=df['hour'], cmap='viridis', alpha=0.3, s=5)
    plt.colorbar(scatter, label='Hour of Day', ax=ax)
    
    # 1:1 line
    lims = [min(df['total_demand_MW'].min(), df['net_generation_MW'].min()),
            max(df['total_demand_MW'].max(), df['net_generation_MW'].max())]
    ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.5, label='1:1 line')
    
    ax.set_xlabel('Total Demand (MW)')
    ax.set_ylabel('Net Generation (MW)')
    ax.set_title('CAISO: Demand vs Net Generation (colored by hour)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_demand_vs_netgen.png"))
    plt.close()
    print(f"    -> Saved: fig5_demand_vs_netgen.png")

# --- Fig 6: Autocorrelation Analysis ---
print("  Generating Fig 6: Autocorrelation...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hourly ACF
from scipy import stats as scipy_stats

lags = range(1, 169)  # Up to 7 days
acf_vals = []
for lag in lags:
    acf_vals.append(df[demand_col].autocorr(lag=lag))

ax = axes[0]
ax.bar(lags, acf_vals, width=1.0, color='#1565C0', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(1.96/np.sqrt(len(df)), color='red', linestyle='--', linewidth=0.8, label='95% CI')
ax.axhline(-1.96/np.sqrt(len(df)), color='red', linestyle='--', linewidth=0.8)
ax.set_title('Autocorrelation Function (Demand)', fontweight='bold')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('ACF')
ax.legend()

# Highlight 24h periodicity
for h in [24, 48, 72, 96, 120, 144, 168]:
    if h <= max(lags):
        ax.axvline(h, color='orange', linestyle=':', alpha=0.5)

# Hourly diff ACF  
ax = axes[1]
demand_diff = df[demand_col].diff().dropna()
acf_diff = []
for lag in range(1, 49):
    acf_diff.append(demand_diff.autocorr(lag=lag))

ax.bar(range(1, 49), acf_diff, width=0.8, color='#E53935', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('ACF of Hourly Demand Change', fontweight='bold')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('ACF')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_autocorrelation.png"))
plt.close()
print(f"    -> Saved: fig6_autocorrelation.png")

# --- Fig 7: Heatmap of hourly demand by month ---
print("  Generating Fig 7: Demand heatmap...")
fig, ax = plt.subplots(figsize=(14, 6))
heatmap_data = df.pivot_table(index='hour', columns='month', values=demand_col, aggfunc='mean')
heatmap_data.columns = [month_names[m] for m in heatmap_data.columns]

sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, annot=True, fmt='.0f',
            cbar_kws={'label': 'Mean Demand (MW)'}, linewidths=0.5)
ax.set_title('CAISO Mean Hourly Demand by Month (MW) - 2023', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Hour of Day')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig7_demand_heatmap.png"))
plt.close()
print(f"    -> Saved: fig7_demand_heatmap.png")

# =====================================================================
#  SECTION 4: KEY STATISTICS SUMMARY
# =====================================================================
print(f"\n" + "-" * 78)
print("  SECTION 4: KEY STATISTICS SUMMARY")
print("-" * 78)

if 'total_demand_MW' in df.columns and 'net_generation_MW' in df.columns:
    surplus = df['net_generation_MW'] - df['total_demand_MW']
    print(f"\n  GENERATION-DEMAND BALANCE:")
    print(f"    Mean surplus/deficit : {surplus.mean():>+10,.0f} MW")
    print(f"    Net importer hours  : {(surplus < 0).sum():>10,} ({(surplus < 0).mean()*100:.1f}%)")
    print(f"    Net exporter hours  : {(surplus > 0).sum():>10,} ({(surplus > 0).mean()*100:.1f}%)")

if 'Total interchange' in df.columns:
    ic = df['Total interchange']
    print(f"\n  INTERCHANGE:")
    print(f"    Mean                : {ic.mean():>+10,.0f} MW (negative = importing)")
    print(f"    Max import          : {ic.min():>10,.0f} MW")
    print(f"    Max export          : {ic.max():>10,.0f} MW")

# Ramping statistics
if demand_col in df.columns:
    ramp = df[demand_col].diff()
    print(f"\n  DEMAND RAMPING:")
    print(f"    Max ramp up         : {ramp.max():>+10,.0f} MW/h")
    print(f"    Max ramp down       : {ramp.min():>+10,.0f} MW/h")
    print(f"    Mean abs ramp       : {ramp.abs().mean():>10,.0f} MW/h")
    print(f"    Std ramp            : {ramp.std():>10,.0f} MW/h")
    
    # Evening ramp (16:00-20:00)
    evening = df[df['hour'].between(16, 20)]
    evening_ramp = evening[demand_col].diff()
    print(f"    Evening ramp mean   : {evening_ramp.mean():>+10,.0f} MW/h (16:00-20:00)")

# Save processed dataset
output_path = os.path.join(DATA_DIR, "caiso_eda_processed_2023.csv")
df.to_csv(output_path)
print(f"\n  Processed dataset saved: {output_path}")
print(f"  Shape: {df.shape}")

print(f"\n  Generated figures:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'):
        fpath = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    * {f} ({size_kb:.0f} KB)")

print("\n" + "=" * 78)
print("  PRELIMINARY EDA COMPLETE")
print("=" * 78)
