"""
=============================================================================
 PHASE 1: DATA PREPROCESSING & FEATURE ENGINEERING
=============================================================================
 Project : Data-Driven Nuclear Baseload & BESS Optimization
           under "Duck Curve" Uncertainty
 
 Pipeline:
   1. Load unified CAISO data
   2. Handle missing values (ffill -> interpolate -> bfill)
   3. Engineer lagged features, rolling statistics, ramp rates
   4. Compute net load (target Y1) and price proxy (target Y2)
   5. Train/Validation/Test split (temporal, no shuffling)
   6. Save preprocessed datasets
=============================================================================
"""

import os
import sys
import io
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 78)
print("  PHASE 1: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 78)

# =====================================================================
#  STEP 1: LOAD DATA
# =====================================================================
print("\n[1/6] Loading data...")

# Try to load unified dataset (with generation breakdown if available)
unified_path = os.path.join(DATA_DIR, "caiso_unified_2023.csv")
df = pd.read_csv(unified_path, index_col=0, parse_dates=True)
print(f"  Loaded: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df.index.min()} -> {df.index.max()}")

# Check if generation data is available
has_solar = any('solar' in c.lower() for c in df.columns)
has_wind = any('wind' in c.lower() for c in df.columns)
has_nuclear = any('nuclear' in c.lower() for c in df.columns)

print(f"\n  Solar data: {'YES' if has_solar else 'NO'}")
print(f"  Wind data:  {'YES' if has_wind else 'NO'}")
print(f"  Nuclear:    {'YES' if has_nuclear else 'NO'}")

# Standardize column names
rename = {}
for c in df.columns:
    cl = c.lower()
    if c == 'net_demand_MW':
        rename[c] = 'net_generation_MW'
df = df.rename(columns=rename)

# =====================================================================
#  STEP 2: HANDLE MISSING VALUES
# =====================================================================
print("\n[2/6] Handling missing values...")
total_before = df.isna().sum().sum()
print(f"  Missing values before: {total_before}")

# Strategy: ffill(limit=3) -> time interpolation -> bfill(limit=3)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].ffill(limit=3)
df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit=6)
df[numeric_cols] = df[numeric_cols].bfill(limit=3)

total_after = df.isna().sum().sum()
print(f"  Missing values after:  {total_after}")
print(f"  Imputed: {total_before - total_after}")

# =====================================================================
#  STEP 3: TEMPORAL FEATURES
# =====================================================================
print("\n[3/6] Engineering temporal features...")

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek        # 0=Mon, 6=Sun
df['month'] = df.index.month
df['day_of_year'] = df.index.dayofyear
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['week_of_year'] = df.index.isocalendar().week.astype(int)

# Cyclical encoding (critical for gradient boosting models)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print(f"  Created 12 temporal features")

# =====================================================================
#  STEP 4: TARGET VARIABLES
# =====================================================================
print("\n[4/6] Computing target variables...")

# Y1: Net Load
# If we have solar and wind, compute: net_load = demand - solar - wind
# Otherwise, approximate using interchange: net_load ~ net_generation (from EIA)
if has_solar and has_wind:
    solar_col = [c for c in df.columns if 'solar' in c.lower()][0]
    wind_col = [c for c in df.columns if 'wind' in c.lower()][0]
    df['net_load_MW'] = df['total_demand_MW'] - df[solar_col] - df[wind_col]
    print(f"  Y1 (net_load_MW) = total_demand - {solar_col} - {wind_col}")
elif 'net_generation_MW' in df.columns:
    # Net generation is total in-state generation. 
    # Net load (demand on conventional plants) can be approximated
    # by total_demand + interchange (imports fill the gap)
    df['net_load_MW'] = df['total_demand_MW']
    print(f"  Y1 (net_load_MW) = total_demand_MW (solar/wind breakdown pending)")
    print(f"       Note: Will be updated once generation-by-fuel data arrives")
else:
    df['net_load_MW'] = df['total_demand_MW']
    print(f"  Y1 (net_load_MW) = total_demand_MW (fallback)")

# Y2: Price Proxy
# Model price as a function of net load using established empirical relationships
# Based on CAISO historical price data: price ~ quadratic function of net load
# Ref: Weron (2014), Electricity price forecasting: A review
print(f"\n  Y2: Constructing price proxy model...")

# Use net generation and demand to create a price proxy
# Higher net load -> higher price (merit order effect)
# Add noise to simulate real price volatility
demand = df['total_demand_MW']
demand_norm = (demand - demand.mean()) / demand.std()

# Base price: quadratic merit order curve
# Typical CAISO prices: $20-80/MWh base, with spikes to $200+ and negatives
base_price = 30 + 25 * demand_norm + 8 * demand_norm**2

# Add time-of-day effect (peak premium)
hour_premium = 5 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)  # Peak around 18:00

# Add stochastic component (realistic price volatility)
np.random.seed(42)
noise = np.random.normal(0, 10, len(df))  # $10/MWh volatility
# Add occasional spikes (5% of hours)
spike_mask = np.random.random(len(df)) < 0.05
noise[spike_mask] = np.random.normal(50, 30, spike_mask.sum())

# Add seasonal component
seasonal = 5 * np.sin(2 * np.pi * (df['day_of_year'] - 80) / 365)  # Peak in summer

df['price_proxy_USD_MWh'] = base_price + hour_premium + seasonal + noise
# Allow negative prices (duck curve effect - midday solar oversupply)
# But cap extreme outliers
df['price_proxy_USD_MWh'] = df['price_proxy_USD_MWh'].clip(-50, 500)

print(f"  Price proxy statistics:")
print(f"    Mean:    ${df['price_proxy_USD_MWh'].mean():.2f}/MWh")
print(f"    Median:  ${df['price_proxy_USD_MWh'].median():.2f}/MWh")
print(f"    Std:     ${df['price_proxy_USD_MWh'].std():.2f}/MWh")
print(f"    Min:     ${df['price_proxy_USD_MWh'].min():.2f}/MWh")
print(f"    Max:     ${df['price_proxy_USD_MWh'].max():.2f}/MWh")
neg_pct = (df['price_proxy_USD_MWh'] < 0).mean() * 100
print(f"    Negative hours: {neg_pct:.1f}%")

# =====================================================================
#  STEP 5: LAGGED & ROLLING FEATURES
# =====================================================================
print("\n[5/6] Engineering lagged and rolling features...")

target_col = 'net_load_MW'

# Lagged features (crucial for time series forecasting)
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    
# Demand lags
for lag in [1, 24]:
    df[f'total_demand_MW_lag{lag}'] = df['total_demand_MW'].shift(lag)

# Price lags
for lag in [1, 24]:
    df[f'price_lag{lag}'] = df['price_proxy_USD_MWh'].shift(lag)

# Rolling statistics
for window in [6, 12, 24]:
    df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(window).mean()
    df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(window).std()
    df[f'{target_col}_rolling_min_{window}h'] = df[target_col].rolling(window).min()
    df[f'{target_col}_rolling_max_{window}h'] = df[target_col].rolling(window).max()

# Ramp rate features
df['demand_ramp'] = df['total_demand_MW'].diff()
df['demand_ramp_24h'] = df['total_demand_MW'].diff(24)
df['net_load_ramp'] = df[target_col].diff()

# Same hour previous day (very predictive for energy data)
df['net_load_same_hour_yesterday'] = df[target_col].shift(24)
df['demand_same_hour_yesterday'] = df['total_demand_MW'].shift(24)

# Exponential weighted moving average
df[f'{target_col}_ewma_24h'] = df[target_col].ewm(span=24).mean()

n_lag_features = sum(1 for c in df.columns if 'lag' in c or 'rolling' in c or 'ramp' in c or 'ewma' in c or 'yesterday' in c)
print(f"  Created {n_lag_features} lagged/rolling features")

# Drop rows with NaN from lagging (first 168 hours = first week)
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"  Dropped {rows_before - rows_after} rows with NaN from lagging (first ~week)")

# =====================================================================
#  STEP 6: TRAIN / VALIDATION / TEST SPLIT
# =====================================================================
print("\n[6/6] Performing temporal train/validation/test split...")

# Temporal split: NO SHUFFLING (this is time series data)
# Train: Jan-Aug 2023 (67%)
# Validation: Sep-Oct 2023 (17%)  
# Test: Nov-Dec 2023 (16%)

train_end = '2023-08-31 23:00:00+00:00'
val_end = '2023-10-31 23:00:00+00:00'

train = df[df.index <= train_end]
val = df[(df.index > train_end) & (df.index <= val_end)]
test = df[df.index > val_end]

print(f"\n  Split Summary:")
print(f"  {'Set':<12} {'Records':>8} {'%':>6} {'Start':>25} {'End':>25}")
print(f"  {'-'*78}")
print(f"  {'Train':<12} {len(train):>8} {len(train)/len(df)*100:>5.1f}% {str(train.index.min()):>25} {str(train.index.max()):>25}")
print(f"  {'Validation':<12} {len(val):>8} {len(val)/len(df)*100:>5.1f}% {str(val.index.min()):>25} {str(val.index.max()):>25}")
print(f"  {'Test':<12} {len(test):>8} {len(test)/len(df)*100:>5.1f}% {str(test.index.min()):>25} {str(test.index.max()):>25}")
print(f"  {'Total':<12} {len(df):>8} {'100.0':>6}%")

# Define feature columns and target columns
target_cols = ['net_load_MW', 'price_proxy_USD_MWh']
exclude_cols = target_cols + ['season']  # Don't include targets as features
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

print(f"\n  Feature columns: {len(feature_cols)}")
print(f"  Target columns: {target_cols}")

# Save splits
train.to_csv(os.path.join(DATA_DIR, "train_2023.csv"))
val.to_csv(os.path.join(DATA_DIR, "val_2023.csv"))
test.to_csv(os.path.join(DATA_DIR, "test_2023.csv"))

# Save feature list
import json
feature_config = {
    'feature_cols': feature_cols,
    'target_cols': target_cols,
    'train_end': train_end,
    'val_end': val_end,
    'n_features': len(feature_cols),
    'n_train': len(train),
    'n_val': len(val),
    'n_test': len(test),
}
with open(os.path.join(DATA_DIR, "feature_config.json"), 'w') as f:
    json.dump(feature_config, f, indent=2)

print(f"\n  Saved files:")
print(f"    * train_2023.csv ({len(train)} rows)")
print(f"    * val_2023.csv ({len(val)} rows)")
print(f"    * test_2023.csv ({len(test)} rows)")
print(f"    * feature_config.json")

# Save full preprocessed dataset too
df.to_csv(os.path.join(DATA_DIR, "caiso_preprocessed_2023.csv"))
print(f"    * caiso_preprocessed_2023.csv ({len(df)} rows, {len(df.columns)} cols)")

print("\n" + "=" * 78)
print("  PHASE 1 COMPLETE: Data ready for ML modeling")
print("=" * 78)
