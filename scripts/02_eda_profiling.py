"""
=============================================================================
 STEP 2: DATA PROFILING & EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
 Project : Data-Driven Nuclear Baseload & BESS Optimization
           under "Duck Curve" Uncertainty
 Author  : Research Team
 Target  : Q1 Journal (Applied Energy / EJOR)
 
 Description:
   Comprehensive EDA of CAISO grid data. This script:
     1. Loads and merges all acquired datasets
     2. Profiles data quality (missing values, outliers, types)
     3. Engineers temporal features (hour, month, season, weekday)
     4. Constructs the "Duck Curve" signature
     5. Identifies target variables & exogenous features for ML
     6. Generates publication-quality visualizations
=============================================================================
"""

import os
import sys
import io
import warnings
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
FIG_DIR     = os.path.join(PROJECT_DIR, "outputs", "figures")
TABLE_DIR   = os.path.join(PROJECT_DIR, "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Plot Style Configuration ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (12, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'solar':    '#FFB300',
    'wind':     '#00ACC1',
    'nuclear':  '#E53935',
    'net_load': '#1E88E5',
    'demand':   '#5E35B1',
    'price':    '#43A047',
    'battery':  '#8E24AA',
    'duck':     '#FF6F00',
}


def print_header(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# =============================================================================
#  SECTION 1: DATA LOADING & MERGING
# =============================================================================
def load_and_merge_data():
    """Load all CAISO CSV files and merge into a unified hourly DataFrame."""
    print_header("SECTION 1: DATA LOADING & MERGING")
    
    # --- Load Demand Data ---
    demand_path = os.path.join(DATA_DIR, "caiso_demand_2023.csv")
    if os.path.exists(demand_path):
        demand = pd.read_csv(demand_path, parse_dates=True)
        print(f"  ✓ Demand data loaded: {demand.shape}")
        print(f"    Columns: {list(demand.columns)}")
    else:
        print(f"  ✗ Demand file not found: {demand_path}")
        demand = None
    
    # --- Load Supply Data ---
    supply_path = os.path.join(DATA_DIR, "caiso_supply_2023.csv")
    if os.path.exists(supply_path):
        supply = pd.read_csv(supply_path, parse_dates=True)
        print(f"  ✓ Supply data loaded: {supply.shape}")
        print(f"    Columns: {list(supply.columns)}")
    else:
        print(f"  ✗ Supply file not found: {supply_path}")
        supply = None
    
    # --- Load Price Data ---
    price_path = os.path.join(DATA_DIR, "caiso_lmp_2023.csv")
    if os.path.exists(price_path):
        prices = pd.read_csv(price_path, parse_dates=True)
        print(f"  ✓ Price data loaded: {prices.shape}")
        print(f"    Columns: {list(prices.columns)}")
    else:
        print(f"  ✗ Price file not found: {price_path}")
        prices = None
    
    return demand, supply, prices


def identify_datetime_column(df, name="DataFrame"):
    """Auto-detect the datetime column in a DataFrame."""
    datetime_candidates = ['Time', 'Interval Start', 'Interval End', 
                           'datetime', 'timestamp', 'Date', 'date', 'time']
    for col in datetime_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
            print(f"    → {name}: Using '{col}' as datetime index")
            return col
    # Try to find any datetime-like column
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(100))
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
                print(f"    → {name}: Auto-detected '{col}' as datetime index")
                return col
            except:
                continue
    return None


def build_unified_dataset(demand, supply, prices):
    """Merge demand, supply, and price data into a single hourly DataFrame."""
    print("\n  Building unified hourly dataset...")
    
    datasets = {}
    
    # Process Demand
    if demand is not None and not demand.empty:
        dt_col = identify_datetime_column(demand, "Demand")
        if dt_col:
            demand = demand.set_index(dt_col)
            # Resample to hourly if sub-hourly
            if len(demand) > 8800:  # More than ~365 days * 24 hours
                demand = demand.select_dtypes(include=[np.number]).resample('h').mean()
            datasets['demand'] = demand

    # Process Supply
    if supply is not None and not supply.empty:
        dt_col = identify_datetime_column(supply, "Supply")
        if dt_col:
            supply = supply.set_index(dt_col)
            if len(supply) > 8800:
                supply = supply.select_dtypes(include=[np.number]).resample('h').mean()
            datasets['supply'] = supply

    # Process Prices
    if prices is not None and not prices.empty:
        dt_col = identify_datetime_column(prices, "Prices")
        if dt_col:
            prices = prices.set_index(dt_col)
            # Keep only numeric columns for prices, take mean per hour
            price_numeric = prices.select_dtypes(include=[np.number])
            if len(price_numeric) > 8800:
                price_numeric = price_numeric.resample('h').mean()
            datasets['prices'] = price_numeric

    if len(datasets) == 0:
        print("  ✗ No datasets available to merge!")
        return None
    
    # Merge all datasets
    if len(datasets) == 1:
        merged = list(datasets.values())[0]
    else:
        merged = pd.concat(datasets.values(), axis=1, join='outer')
    
    # Remove duplicate columns
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    print(f"\n  ✓ Unified dataset shape: {merged.shape}")
    print(f"    Date range: {merged.index.min()} → {merged.index.max()}")
    print(f"    Columns: {list(merged.columns)}")
    
    return merged


# =============================================================================
#  SECTION 2: DATA PROFILING & QUALITY ASSESSMENT
# =============================================================================
def profile_data(df):
    """Comprehensive data profiling: types, missing values, statistics."""
    print_header("SECTION 2: DATA PROFILING & QUALITY ASSESSMENT")
    
    # --- 2.1: Basic Info ---
    print("\n  2.1 BASIC INFORMATION")
    print(f"  {'─' * 60}")
    print(f"  Total records     : {len(df):,}")
    print(f"  Total features    : {len(df.columns)}")
    print(f"  Memory usage      : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"  Date range        : {df.index.min()} → {df.index.max()}")
    
    # Expected hours in range
    if hasattr(df.index, 'freq') or len(df) > 0:
        date_range = (df.index.max() - df.index.min())
        expected_hours = date_range.total_seconds() / 3600 + 1
        coverage = len(df) / expected_hours * 100 if expected_hours > 0 else 0
        print(f"  Expected hourly records: {expected_hours:,.0f}")
        print(f"  Data coverage     : {coverage:.1f}%")
    
    # --- 2.2: Column Types & Missing Values ---
    print(f"\n  2.2 COLUMN TYPES & MISSING VALUES")
    print(f"  {'─' * 60}")
    print(f"  {'Column':<35} {'Type':<12} {'Non-Null':>10} {'Missing':>8} {'%':>6}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 6}")
    
    quality_report = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        missing = df[col].isna().sum()
        pct = missing / len(df) * 100
        print(f"  {col:<35} {dtype:<12} {non_null:>10,} {missing:>8,} {pct:>5.1f}%")
        quality_report.append({
            'Column': col, 'Type': dtype, 'Non-Null': non_null,
            'Missing': missing, 'Missing_Pct': round(pct, 2)
        })
    
    quality_df = pd.DataFrame(quality_report)
    quality_df.to_csv(os.path.join(TABLE_DIR, "data_quality_report.csv"), index=False)
    print(f"\n  → Quality report saved to: outputs/tables/data_quality_report.csv")
    
    # --- 2.3: Descriptive Statistics ---
    print(f"\n  2.3 DESCRIPTIVE STATISTICS (Numeric Columns)")
    print(f"  {'─' * 60}")
    
    numeric_df = df.select_dtypes(include=[np.number])
    desc = numeric_df.describe().T
    desc['skewness'] = numeric_df.skew()
    desc['kurtosis'] = numeric_df.kurtosis()
    desc['iqr'] = desc['75%'] - desc['25%']
    
    print(desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness']].to_string())
    
    desc.to_csv(os.path.join(TABLE_DIR, "descriptive_statistics.csv"))
    print(f"\n  → Descriptive stats saved to: outputs/tables/descriptive_statistics.csv")
    
    return numeric_df, quality_df


# =============================================================================
#  SECTION 3: MISSING VALUE HANDLING
# =============================================================================
def handle_missing_values(df):
    """Handle missing values using forward-fill then backward-fill strategy."""
    print_header("SECTION 3: MISSING VALUE HANDLING")
    
    total_missing_before = df.isna().sum().sum()
    print(f"  Total missing values before: {total_missing_before:,}")
    
    if total_missing_before == 0:
        print("  ✓ No missing values detected — dataset is complete!")
        return df
    
    # Strategy: Forward fill (carry last valid observation), then backward fill
    # This is physically meaningful for time-series energy data
    df_clean = df.ffill().bfill()
    
    total_missing_after = df_clean.isna().sum().sum()
    print(f"  Total missing values after : {total_missing_after:,}")
    print(f"  Values imputed             : {total_missing_before - total_missing_after:,}")
    print(f"  Strategy: Forward-fill → Backward-fill (temporally coherent)")
    
    # Drop any remaining columns that are entirely NaN
    cols_before = len(df_clean.columns)
    df_clean = df_clean.dropna(axis=1, how='all')
    cols_after = len(df_clean.columns)
    if cols_before > cols_after:
        print(f"  Dropped {cols_before - cols_after} entirely-NaN columns")
    
    return df_clean


# =============================================================================
#  SECTION 4: TEMPORAL FEATURE ENGINEERING
# =============================================================================
def engineer_temporal_features(df):
    """Create temporal features from the datetime index."""
    print_header("SECTION 4: TEMPORAL FEATURE ENGINEERING")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        print("  ⚠ Index is not DatetimeIndex, attempting conversion...")
        df.index = pd.to_datetime(df.index, utc=True)
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek          # 0=Mon, 6=Sun
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Season mapping (Northern Hemisphere)
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall',  10: 'Fall',  11: 'Fall'}
    df['season'] = df['month'].map(season_map)
    
    # Cyclical encoding for hour and month (important for ML models)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    new_features = ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
                    'season', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    print(f"  ✓ Created {len(new_features)} temporal features:")
    for f in new_features:
        print(f"    • {f}")
    
    return df


# =============================================================================
#  SECTION 5: DUCK CURVE ANALYSIS & KEY VISUALIZATIONS
# =============================================================================
def identify_key_columns(df):
    """Identify which columns correspond to our target variables."""
    col_lower = {c: c.lower() for c in df.columns}
    
    result = {}
    
    # Solar generation
    for c, cl in col_lower.items():
        if 'solar' in cl and ('gen' in cl or 'mw' in cl or cl == 'solar'):
            result['solar'] = c
            break
    if 'solar' not in result:
        for c, cl in col_lower.items():
            if 'solar' in cl:
                result['solar'] = c
                break
    
    # Wind generation
    for c, cl in col_lower.items():
        if 'wind' in cl and ('gen' in cl or 'mw' in cl or cl == 'wind'):
            result['wind'] = c
            break
    if 'wind' not in result:
        for c, cl in col_lower.items():
            if 'wind' in cl:
                result['wind'] = c
                break
    
    # Net load / Net demand
    for c, cl in col_lower.items():
        if 'net' in cl and ('load' in cl or 'demand' in cl):
            result['net_load'] = c
            break
    
    # Total demand / load
    for c, cl in col_lower.items():
        if ('demand' in cl or 'load' in cl) and 'net' not in cl:
            result['demand'] = c
            break
    
    # Nuclear
    for c, cl in col_lower.items():
        if 'nuclear' in cl:
            result['nuclear'] = c
            break
    
    # Price / LMP
    for c, cl in col_lower.items():
        if 'lmp' in cl or 'price' in cl or 'energy' in cl.replace('renewable', ''):
            if 'congestion' not in cl and 'loss' not in cl:
                result['price'] = c
                break
    
    # Battery / Storage
    for c, cl in col_lower.items():
        if 'batter' in cl or 'storage' in cl:
            result['battery'] = c
            break
    
    print("\n  KEY COLUMN MAPPING:")
    for role, col in result.items():
        print(f"    {role:<12} → '{col}'")
    
    return result


def plot_duck_curve(df, col_map):
    """Generate the iconic Duck Curve visualization by season."""
    print("\n  Generating Duck Curve visualization...")
    
    if 'net_load' not in col_map and 'demand' in col_map:
        # Compute net load if solar is available
        if 'solar' in col_map:
            net_load_col = f"{col_map['demand']}_minus_solar"
            df[net_load_col] = df[col_map['demand']] - df[col_map['solar']]
            col_map['net_load'] = net_load_col
            if 'wind' in col_map:
                df[net_load_col] = df[net_load_col] - df[col_map['wind']]
            print(f"    → Computed Net Load = Demand - Solar (- Wind)")
        else:
            print("    ⚠ Cannot compute Duck Curve: need net_load or (demand + solar)")
            return
    
    if 'net_load' not in col_map:
        print("    ⚠ No net load data available for Duck Curve")
        return
    
    net_col = col_map['net_load']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CAISO "Duck Curve" — Hourly Net Load Profile by Season (2023)',
                 fontsize=15, fontweight='bold', y=1.02)
    
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    season_colors = ['#1565C0', '#2E7D32', '#E65100', '#BF360C']
    
    for idx, (season, color) in enumerate(zip(seasons, season_colors)):
        ax = axes[idx // 2, idx % 2]
        season_data = df[df['season'] == season]
        
        if season_data.empty:
            ax.set_title(f'{season} (No Data)')
            continue
        
        # Group by hour
        hourly = season_data.groupby('hour')[net_col]
        
        # Plot individual days as thin transparent lines
        for date, grp in season_data.groupby(season_data.index.date):
            if len(grp) >= 20:  # Need most hours of the day
                ax.plot(grp['hour'].values, grp[net_col].values, 
                        color=color, alpha=0.03, linewidth=0.5)
        
        # Plot mean ± std
        mean = hourly.mean()
        std = hourly.std()
        p10 = hourly.quantile(0.10)
        p90 = hourly.quantile(0.90)
        
        ax.fill_between(mean.index, p10, p90, alpha=0.2, color=color, label='P10–P90')
        ax.plot(mean.index, mean, color=color, linewidth=2.5, label='Mean')
        
        # Mark the "belly" of the duck
        min_hour = mean.idxmin()
        min_val = mean.min()
        ax.annotate(f'Duck belly\n{min_val:,.0f} MW',
                    xy=(min_hour, min_val), fontsize=8,
                    xytext=(min_hour + 2, min_val + mean.std() * 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontweight='bold', color='red')
        
        ax.set_title(f'{season}', fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Net Load (MW)')
        ax.set_xlim(0, 23)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_duck_curve_seasonal.png"))
    plt.close()
    print(f"    → Saved: outputs/figures/fig1_duck_curve_seasonal.png")


def plot_generation_stack(df, col_map):
    """Plot a stacked area chart of generation by fuel type for a sample week."""
    print("  Generating generation stack chart...")
    
    gen_cols = {}
    for key in ['solar', 'wind', 'nuclear', 'battery']:
        if key in col_map:
            gen_cols[key] = col_map[key]
    
    if len(gen_cols) < 2:
        print("    ⚠ Insufficient generation columns for stack chart")
        return
    
    # Pick a representative spring week (peak duck curve season)
    spring = df[df['month'].isin([3, 4, 5])]
    if len(spring) > 168:  # 7 days * 24 hours
        sample = spring.iloc[:168]
    elif len(df) > 168:
        sample = df.iloc[:168]
    else:
        sample = df
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors_list = [COLORS.get(k, '#999999') for k in gen_cols.keys()]
    labels = [k.title() for k in gen_cols.keys()]
    
    data_arrays = []
    for col in gen_cols.values():
        vals = sample[col].fillna(0).clip(lower=0).values
        data_arrays.append(vals)
    
    ax.stackplot(range(len(sample)), *data_arrays, labels=labels,
                 colors=colors_list, alpha=0.85)
    
    if 'demand' in col_map:
        ax.plot(range(len(sample)), sample[col_map['demand']].values,
                color=COLORS['demand'], linewidth=2, label='Total Demand',
                linestyle='--')
    
    ax.set_title('CAISO Generation Mix — Sample Week (Spring 2023)',
                 fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Generation (MW)')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_generation_stack.png"))
    plt.close()
    print(f"    → Saved: outputs/figures/fig2_generation_stack.png")


def plot_price_analysis(df, col_map):
    """Analyze and plot electricity price patterns."""
    if 'price' not in col_map:
        print("    ⚠ No price data available for price analysis")
        return
    
    print("  Generating price analysis plots...")
    
    price_col = col_map['price']
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Price distribution
    ax1 = fig.add_subplot(gs[0, 0])
    price_data = df[price_col].dropna()
    # Clip extreme outliers for visualization
    p1, p99 = price_data.quantile(0.01), price_data.quantile(0.99)
    price_clipped = price_data.clip(p1, p99)
    ax1.hist(price_clipped, bins=60, color=COLORS['price'], alpha=0.7, 
             edgecolor='white', linewidth=0.5)
    ax1.axvline(price_data.median(), color='red', linestyle='--', label=f'Median: ${price_data.median():.1f}')
    ax1.set_title('LMP Distribution', fontweight='bold')
    ax1.set_xlabel('Price ($/MWh)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 2. Hourly price profile
    ax2 = fig.add_subplot(gs[0, 1])
    hourly_price = df.groupby('hour')[price_col].agg(['mean', 'median', 'std'])
    ax2.plot(hourly_price.index, hourly_price['mean'], color=COLORS['price'],
             linewidth=2, label='Mean')
    ax2.fill_between(hourly_price.index, 
                     hourly_price['mean'] - hourly_price['std'],
                     hourly_price['mean'] + hourly_price['std'],
                     alpha=0.2, color=COLORS['price'])
    ax2.set_title('Hourly Price Profile', fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Price ($/MWh)')
    ax2.legend()
    
    # 3. Price vs Net Load scatter
    ax3 = fig.add_subplot(gs[0, 2])
    if 'net_load' in col_map:
        nl_col = col_map['net_load']
        # Sample for scatter plot performance
        sample_idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
        ax3.scatter(df[nl_col].iloc[sample_idx], df[price_col].iloc[sample_idx],
                    alpha=0.15, s=5, color=COLORS['net_load'])
        ax3.set_xlabel('Net Load (MW)')
        ax3.set_ylabel('Price ($/MWh)')
        ax3.set_title('Price vs. Net Load', fontweight='bold')
    
    # 4. Monthly price boxplot
    ax4 = fig.add_subplot(gs[1, 0:2])
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df_plot = df[[price_col, 'month']].dropna()
    df_plot['month_name'] = df_plot['month'].map(month_names)
    bp = ax4.boxplot([df_plot[df_plot['month'] == m][price_col].values 
                       for m in range(1, 13)],
                      labels=[month_names[m] for m in range(1, 13)],
                      patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['price'])
        patch.set_alpha(0.6)
    ax4.set_title('Monthly Price Distribution', fontweight='bold')
    ax4.set_ylabel('Price ($/MWh)')
    
    # 5. Negative price frequency by hour
    ax5 = fig.add_subplot(gs[1, 2])
    neg_pct = df.groupby('hour')[price_col].apply(lambda x: (x < 0).mean() * 100)
    ax5.bar(neg_pct.index, neg_pct.values, color=COLORS['duck'], alpha=0.8)
    ax5.set_title('Negative Price Frequency', fontweight='bold')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('% Hours with LMP < $0')
    
    fig.suptitle('CAISO Electricity Price Analysis (2023)', fontsize=14, 
                 fontweight='bold', y=1.02)
    fig.savefig(os.path.join(FIG_DIR, "fig3_price_analysis.png"))
    plt.close()
    print(f"    → Saved: outputs/figures/fig3_price_analysis.png")


def plot_correlation_matrix(df, col_map):
    """Plot correlation heatmap for key features."""
    print("  Generating correlation heatmap...")
    
    # Select key numeric columns
    key_cols = []
    for role in ['solar', 'wind', 'nuclear', 'demand', 'net_load', 'price', 'battery']:
        if role in col_map and col_map[role] in df.columns:
            key_cols.append(col_map[role])
    
    # Add temporal features
    for tc in ['hour', 'month', 'is_weekend']:
        if tc in df.columns:
            key_cols.append(tc)
    
    if len(key_cols) < 3:
        print("    ⚠ Too few columns for correlation analysis")
        return
    
    corr = df[key_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center='light', as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True,
                linewidths=0.5, annot=True, fmt='.2f', ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix — Key Variables',
                 fontweight='bold', fontsize=13, pad=15)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_correlation_heatmap.png"))
    plt.close()
    print(f"    → Saved: outputs/figures/fig4_correlation_heatmap.png")


def plot_time_series_overview(df, col_map):
    """Plot a multi-panel time-series overview of the full year."""
    print("  Generating time-series overview...")
    
    panels = []
    for role, color_key in [('demand', 'demand'), ('net_load', 'net_load'),
                             ('solar', 'solar'), ('price', 'price')]:
        if role in col_map and col_map[role] in df.columns:
            panels.append((role, col_map[role], COLORS[color_key]))
    
    if len(panels) == 0:
        print("    ⚠ No data for time series overview")
        return
    
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    
    # Resample to daily for cleaner overview
    daily = df.resample('D').mean()
    
    for ax, (role, col, color) in zip(axes, panels):
        if col in daily.columns:
            ax.plot(daily.index, daily[col], color=color, linewidth=0.8, alpha=0.9)
            ax.fill_between(daily.index, 0, daily[col], color=color, alpha=0.15)
            ax.set_ylabel(f'{role.replace("_", " ").title()}\n(MW)')
            if role == 'price':
                ax.set_ylabel('Price\n($/MWh)')
            ax.set_title(f'{role.replace("_", " ").title()} — Daily Average', 
                        fontweight='bold', fontsize=11, loc='left')
    
    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    
    fig.suptitle('CAISO Grid Data Overview — Full Year 2023', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_timeseries_overview.png"))
    plt.close()
    print(f"    → Saved: outputs/figures/fig5_timeseries_overview.png")


# =============================================================================
#  SECTION 6: TARGET & FEATURE IDENTIFICATION
# =============================================================================
def identify_targets_and_features(df, col_map):
    """Formally identify target variables and exogenous features for ML."""
    print_header("SECTION 6: TARGET & FEATURE IDENTIFICATION FOR ML")
    
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │              ML MODEL VARIABLE SPECIFICATION                │")
    print("  └─────────────────────────────────────────────────────────────┘")
    
    # Target Variables
    print("\n  TARGET VARIABLES (Y):")
    print("  " + "─" * 60)
    targets = {}
    
    if 'net_load' in col_map:
        targets['Net Load'] = col_map['net_load']
        print(f"    Y₁ : Net Grid Load (MW)       → '{col_map['net_load']}'")
    if 'price' in col_map:
        targets['Price'] = col_map['price']
        print(f"    Y₂ : Electricity Price ($/MWh) → '{col_map['price']}'")
    
    # Exogenous Features
    print("\n  EXOGENOUS FEATURES (X):")
    print("  " + "─" * 60)
    
    feature_groups = {
        'Temporal': ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
                     'hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
        'Generation': [],
        'Demand': [],
    }
    
    for role in ['solar', 'wind', 'nuclear', 'battery']:
        if role in col_map and col_map[role] in df.columns:
            feature_groups['Generation'].append(col_map[role])
    
    if 'demand' in col_map and col_map['demand'] in df.columns:
        feature_groups['Demand'].append(col_map['demand'])
    
    for group, features in feature_groups.items():
        avail = [f for f in features if f in df.columns]
        print(f"\n    {group} Features ({len(avail)}):")
        for f in avail:
            print(f"      • {f}")
    
    # Also list any remaining numeric columns
    used = set(sum(feature_groups.values(), []))
    used.update(targets.values())
    remaining = [c for c in df.select_dtypes(include=[np.number]).columns 
                 if c not in used]
    if remaining:
        print(f"\n    Other Available Numeric Features ({len(remaining)}):")
        for f in remaining[:15]:
            print(f"      • {f}")
        if len(remaining) > 15:
            print(f"      ... and {len(remaining) - 15} more")
    
    return targets, feature_groups


# =============================================================================
#  SECTION 7: STATISTICAL SUMMARY
# =============================================================================
def statistical_summary(df, col_map):
    """Print key statistical findings relevant to the Duck Curve thesis."""
    print_header("SECTION 7: KEY STATISTICAL FINDINGS")
    
    findings = []
    
    if 'solar' in col_map and col_map['solar'] in df.columns:
        solar = df[col_map['solar']]
        peak_solar = solar.max()
        mean_solar = solar.mean()
        zero_hours = (solar <= 0).sum()
        total = len(solar)
        print(f"\n  SOLAR GENERATION:")
        print(f"    Peak     : {peak_solar:>10,.0f} MW")
        print(f"    Mean     : {mean_solar:>10,.0f} MW")
        print(f"    Zero-gen : {zero_hours:>10,} hours ({zero_hours/total*100:.1f}%)")
        findings.append(f"Solar peak: {peak_solar:,.0f} MW, zero-gen {zero_hours/total*100:.1f}% of hours")
    
    if 'net_load' in col_map and col_map['net_load'] in df.columns:
        nl = df[col_map['net_load']]
        ramp = nl.diff()
        max_ramp_up = ramp.max()
        max_ramp_down = ramp.min()
        # Duck curve depth: min net load typically at solar peak
        duck_depth = nl.groupby(df['hour']).mean()
        belly = duck_depth.min()
        belly_hour = duck_depth.idxmin()
        evening_peak = duck_depth.loc[17:21].max() if len(duck_depth) > 20 else duck_depth.max()
        ramp_range = evening_peak - belly
        
        print(f"\n  NET LOAD (Duck Curve Metrics):")
        print(f"    Duck belly      : {belly:>10,.0f} MW (hour {belly_hour}:00)")
        print(f"    Evening peak    : {evening_peak:>10,.0f} MW")
        print(f"    Ramp range      : {ramp_range:>10,.0f} MW (belly → peak)")
        print(f"    Max hourly ramp ↑: {max_ramp_up:>10,.0f} MW/h")
        print(f"    Max hourly ramp ↓: {max_ramp_down:>10,.0f} MW/h")
    
    if 'price' in col_map and col_map['price'] in df.columns:
        p = df[col_map['price']]
        neg_hours = (p < 0).sum()
        total = len(p)
        print(f"\n  ELECTRICITY PRICES:")
        print(f"    Mean     : ${p.mean():>8.2f}/MWh")
        print(f"    Median   : ${p.median():>8.2f}/MWh")
        print(f"    Std dev  : ${p.std():>8.2f}/MWh")
        print(f"    Min      : ${p.min():>8.2f}/MWh")
        print(f"    Max      : ${p.max():>8.2f}/MWh")
        print(f"    Negative : {neg_hours:>8,} hours ({neg_hours/total*100:.1f}%)")
    
    if 'nuclear' in col_map and col_map['nuclear'] in df.columns:
        nuc = df[col_map['nuclear']]
        print(f"\n  NUCLEAR GENERATION:")
        print(f"    Mean     : {nuc.mean():>10,.0f} MW")
        print(f"    Std dev  : {nuc.std():>10,.0f} MW  (CV = {nuc.std()/nuc.mean()*100:.1f}%)")
        print(f"    Min      : {nuc.min():>10,.0f} MW")
        print(f"    Max      : {nuc.max():>10,.0f} MW")
        print(f"    → Nuclear is {'highly stable' if nuc.std()/nuc.mean() < 0.1 else 'moderately variable'} (low CV confirms baseload role)")


# =============================================================================
#  MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    # Step 1: Load data
    demand, supply, prices = load_and_merge_data()
    
    # Step 2: Build unified dataset
    df = build_unified_dataset(demand, supply, prices)
    
    if df is None or df.empty:
        print("\n  ✗ FATAL: No data available for EDA. Exiting.")
        sys.exit(1)
    
    # Step 3: Profile data
    numeric_df, quality_df = profile_data(df)
    
    # Step 4: Handle missing values
    df = handle_missing_values(df)
    
    # Step 5: Engineer temporal features
    df = engineer_temporal_features(df)
    
    # Step 6: Identify key columns
    col_map = identify_key_columns(df)
    
    # Step 7: Visualizations
    print_header("SECTION 5: VISUALIZATIONS")
    plot_duck_curve(df, col_map)
    plot_generation_stack(df, col_map)
    plot_price_analysis(df, col_map)
    plot_correlation_matrix(df, col_map)
    plot_time_series_overview(df, col_map)
    
    # Step 8: Identify targets & features
    targets, feature_groups = identify_targets_and_features(df, col_map)
    
    # Step 9: Statistical summary
    statistical_summary(df, col_map)
    
    # Step 10: Save final processed dataset
    output_path = os.path.join(DATA_DIR, "caiso_unified_2023.csv")
    df.to_csv(output_path)
    print(f"\n  ✓ Final unified dataset saved: {output_path}")
    print(f"    Shape: {df.shape}")
    
    print("\n" + "=" * 78)
    print("  EDA PIPELINE COMPLETE")
    print("=" * 78)
