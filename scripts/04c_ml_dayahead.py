"""
=============================================================================
 PHASE 2 (v3): DAY-AHEAD FORECASTING — PUBLICATION-READY
=============================================================================
 Key methodological fixes:
   1. Forecast horizon = 24h (day-ahead), consistent with MILP dispatch
   2. Only features available at forecast time (lag24+, calendar, ISO forecast)
   3. No lag1-lag12, no rolling features computed at target time
   4. Proper benchmark: LightGBM vs XGBoost vs RandomForest
   5. Overfit diagnostics: train/val/test gaps, learning curves
   6. CQR prediction intervals calibrated on validation set
=============================================================================
"""

import os
import sys
import io
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR  = os.path.join(PROJECT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "outputs", "tables")
MODEL_DIR = os.path.join(PROJECT_DIR, "outputs", "models")
for d in [FIG_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

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
print("  PHASE 2 (v3): DAY-AHEAD FORECASTING — PUBLICATION-READY")
print("  Forecast horizon: 24h (consistent with MILP dispatch)")
print("=" * 78)

# =====================================================================
#  REBUILD FEATURES FROM SCRATCH
# =====================================================================
print("\n[1/7] Rebuilding features for day-ahead forecast horizon...")

unified_path = os.path.join(DATA_DIR, "caiso_unified_2023.csv")
df = pd.read_csv(unified_path, index_col=0, parse_dates=True)
print(f"  Raw data: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# --- Target: total_demand_MW ---
# Note: We forecast total demand. In a full study, net load = demand - solar - wind
# would be ideal, but fuel-by-source breakdown is unavailable.
# The duck curve effects manifest in temporal demand patterns and interchange.
df['target_demand_MW'] = df['total_demand_MW']

# --- Price proxy (synthetic, acknowledged as limitation) ---
demand = df['total_demand_MW']
demand_norm = (demand - demand.mean()) / demand.std()
base_price = 30 + 25 * demand_norm + 8 * demand_norm**2
hour_premium = 5 * np.sin(2 * np.pi * (df.index.hour - 6) / 24)
np.random.seed(42)
noise = np.random.normal(0, 10, len(df))
spike_mask = np.random.random(len(df)) < 0.05
noise[spike_mask] = np.random.normal(50, 30, spike_mask.sum())
seasonal = 5 * np.sin(2 * np.pi * (df.index.dayofyear - 80) / 365)
df['price_proxy_USD_MWh'] = (base_price + hour_premium + seasonal + noise).clip(-50, 500)

# =====================================================================
#  DAY-AHEAD FEATURE ENGINEERING
# =====================================================================
# Rule: At time T, we forecast T+24. Only data from time T and before
# is available. Calendar features at target time are deterministic (known).
# day_ahead_demand_forecast_MW is published before the operating day.

print("\n  Building day-ahead feature set...")
print("  Rule: Only features available 24h+ before the target hour")

# Calendar features (deterministic — always known)
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['day_of_year'] = df.index.dayofyear
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['week_of_year'] = df.index.isocalendar().week.astype(int)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# ISO day-ahead forecast (available before operating day)
# Already in dataset as day_ahead_demand_forecast_MW

# Lagged features — ONLY lag >= 24h (available at forecast time)
for lag in [24, 48, 72, 168]:
    df[f'demand_lag{lag}h'] = df['target_demand_MW'].shift(lag)

# Same-hour-same-weekday last week
df['demand_lag168h_same_dow'] = df['target_demand_MW'].shift(168)

# Yesterday's daily statistics (available at forecast time)
df['yesterday_mean'] = df['target_demand_MW'].shift(24).rolling(24).mean()
df['yesterday_max'] = df['target_demand_MW'].shift(24).rolling(24).max()
df['yesterday_min'] = df['target_demand_MW'].shift(24).rolling(24).min()
df['yesterday_std'] = df['target_demand_MW'].shift(24).rolling(24).std()
df['yesterday_range'] = df['yesterday_max'] - df['yesterday_min']

# Last week's same-day-of-week average for this hour
df['lastweek_mean'] = df['target_demand_MW'].shift(168).rolling(24).mean()

# Demand trend: change between yesterday and day-before-yesterday
df['demand_trend_24h'] = df['demand_lag24h'] - df['demand_lag48h']
df['demand_trend_week'] = df['demand_lag24h'] - df['demand_lag168h_same_dow']

# Price lag (only lag24+)
df['price_lag24h'] = df['price_proxy_USD_MWh'].shift(24)

# Forecast error yesterday (ISO forecast vs actual, known at forecast time)
df['forecast_error_yesterday'] = (
    df['day_ahead_demand_forecast_MW'].shift(24) - df['target_demand_MW'].shift(24)
)

# Drop NaN rows from lagging (first ~168 hours)
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"  Dropped {rows_before - rows_after} rows (lag warmup)")

# Define feature set
FEATURE_COLS = [
    # ISO day-ahead forecast (primary predictor)
    'day_ahead_demand_forecast_MW',
    # Calendar (deterministic)
    'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'week_of_year',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    # Lagged demand (available 24h+ before target)
    'demand_lag24h', 'demand_lag48h', 'demand_lag72h', 'demand_lag168h',
    # Yesterday's statistics
    'yesterday_mean', 'yesterday_max', 'yesterday_min', 'yesterday_std', 'yesterday_range',
    # Last week context
    'lastweek_mean',
    # Trends
    'demand_trend_24h', 'demand_trend_week',
    # Price context
    'price_lag24h',
    # Forecast skill
    'forecast_error_yesterday',
]

TARGET_COLS = ['target_demand_MW', 'price_proxy_USD_MWh']

# Verify all features exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

print(f"\n  Feature set ({len(FEATURE_COLS)} features):")
for f in FEATURE_COLS:
    avail = "24h+ lag" if "lag" in f or "yesterday" in f or "lastweek" in f or "trend" in f or "error" in f else "deterministic"
    if f == 'day_ahead_demand_forecast_MW':
        avail = "ISO published"
    print(f"    {f:<40} ({avail})")

# =====================================================================
#  TEMPORAL SPLIT
# =====================================================================
print("\n[2/7] Temporal train/val/test split...")

train_end = '2023-08-31 23:00:00+00:00'
val_end = '2023-10-31 23:00:00+00:00'

train = df[df.index <= train_end].copy()
val = df[(df.index > train_end) & (df.index <= val_end)].copy()
test = df[df.index > val_end].copy()

X_train = train[FEATURE_COLS].values
X_val = val[FEATURE_COLS].values
X_test = test[FEATURE_COLS].values

print(f"  Train: {X_train.shape} ({train.index.min().date()} to {train.index.max().date()})")
print(f"  Val:   {X_val.shape} ({val.index.min().date()} to {val.index.max().date()})")
print(f"  Test:  {X_test.shape} ({test.index.min().date()} to {test.index.max().date()})")

# =====================================================================
#  HELPER
# =====================================================================
def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# =====================================================================
#  BENCHMARK: 3 MODELS x 2 TARGETS
# =====================================================================
print("\n" + "=" * 78)
print("[3/7] Training benchmarks (LightGBM, XGBoost, RandomForest)...")
print("  Forecast horizon: day-ahead (24h)")

benchmark = {}

for target in TARGET_COLS:
    print(f"\n{'='*60}")
    tname = 'Demand' if 'demand' in target else 'Price'
    print(f"  Target: {tname} ({target})")
    print(f"{'='*60}")

    y_train = train[target].values
    y_val = val[target].values
    y_test = test[target].values

    benchmark[target] = {}

    # --- LightGBM ---
    print(f"\n  [LightGBM]")
    t0 = time.time()
    dtrain = lgb.Dataset(X_train, y_train, feature_name=FEATURE_COLS)
    dval = lgb.Dataset(X_val, y_val, feature_name=FEATURE_COLS, reference=dtrain)

    lgb_model = lgb.train(
        {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
         'num_leaves': 63, 'learning_rate': 0.03, 'feature_fraction': 0.7,
         'bagging_fraction': 0.7, 'bagging_freq': 5, 'min_child_samples': 30,
         'n_estimators': 3000, 'early_stopping_rounds': 100, 'verbose': -1,
         'random_state': 42, 'n_jobs': -1, 'lambda_l1': 0.1, 'lambda_l2': 0.1},
        dtrain, valid_sets=[dval], valid_names=['val'],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    lgb_time = time.time() - t0
    lgb_pred = {'train': lgb_model.predict(X_train), 'val': lgb_model.predict(X_val), 'test': lgb_model.predict(X_test)}
    lgb_m = {s: calc_metrics(train[target].values if s == 'train' else (val[target].values if s == 'val' else y_test), lgb_pred[s]) for s in ['train', 'val', 'test']}

    lgb_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': lgb_model.feature_importance(importance_type='gain')}).sort_values('importance', ascending=False)

    benchmark[target]['LightGBM'] = {'pred': lgb_pred, 'metrics': lgb_m, 'time': lgb_time, 'model': lgb_model, 'importance': lgb_imp}
    print(f"    Train: MAE={lgb_m['train']['MAE']:>8.1f}  R2={lgb_m['train']['R2']:.4f}")
    print(f"    Val:   MAE={lgb_m['val']['MAE']:>8.1f}  R2={lgb_m['val']['R2']:.4f}")
    print(f"    Test:  MAE={lgb_m['test']['MAE']:>8.1f}  R2={lgb_m['test']['R2']:.4f}  ({lgb_time:.1f}s)")

    # --- XGBoost ---
    print(f"\n  [XGBoost]")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000, max_depth=6, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=30,
        early_stopping_rounds=100, eval_metric='mae',
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_time = time.time() - t0
    xgb_pred = {'train': xgb_model.predict(X_train), 'val': xgb_model.predict(X_val), 'test': xgb_model.predict(X_test)}
    xgb_m = {s: calc_metrics(train[target].values if s == 'train' else (val[target].values if s == 'val' else y_test), xgb_pred[s]) for s in ['train', 'val', 'test']}

    xgb_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=False)

    benchmark[target]['XGBoost'] = {'pred': xgb_pred, 'metrics': xgb_m, 'time': xgb_time, 'model': xgb_model, 'importance': xgb_imp}
    print(f"    Train: MAE={xgb_m['train']['MAE']:>8.1f}  R2={xgb_m['train']['R2']:.4f}")
    print(f"    Val:   MAE={xgb_m['val']['MAE']:>8.1f}  R2={xgb_m['val']['R2']:.4f}")
    print(f"    Test:  MAE={xgb_m['test']['MAE']:>8.1f}  R2={xgb_m['test']['R2']:.4f}  ({xgb_time:.1f}s)")

    # --- Random Forest ---
    print(f"\n  [RandomForest]")
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=20,
        max_features=0.7, random_state=42, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_pred = {'train': rf_model.predict(X_train), 'val': rf_model.predict(X_val), 'test': rf_model.predict(X_test)}
    rf_m = {s: calc_metrics(train[target].values if s == 'train' else (val[target].values if s == 'val' else y_test), rf_pred[s]) for s in ['train', 'val', 'test']}

    rf_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)

    benchmark[target]['RandomForest'] = {'pred': rf_pred, 'metrics': rf_m, 'time': rf_time, 'model': rf_model, 'importance': rf_imp}
    print(f"    Train: MAE={rf_m['train']['MAE']:>8.1f}  R2={rf_m['train']['R2']:.4f}")
    print(f"    Val:   MAE={rf_m['val']['MAE']:>8.1f}  R2={rf_m['val']['R2']:.4f}")
    print(f"    Test:  MAE={rf_m['test']['MAE']:>8.1f}  R2={rf_m['test']['R2']:.4f}  ({rf_time:.1f}s)")

# Select best model per target
best_models = {}
for target in TARGET_COLS:
    best_name = min(benchmark[target].keys(), key=lambda k: benchmark[target][k]['metrics']['val']['MAE'])
    best_models[target] = best_name
    print(f"\n  Best for {target}: {best_name}")

# =====================================================================
#  PERSISTENCE BASELINE (naive day-ahead: yesterday same hour)
# =====================================================================
print("\n" + "=" * 78)
print("[4/7] Persistence baseline (yesterday same hour)...")

for target in TARGET_COLS:
    tname = 'Demand' if 'demand' in target else 'Price'
    y_test = test[target].values
    y_persist = test[f'demand_lag24h'].values if 'demand' in target else test['price_lag24h'].values
    persist_m = calc_metrics(y_test, y_persist)
    print(f"  {tname} persistence: MAE={persist_m['MAE']:.1f}  R2={persist_m['R2']:.4f}")
    benchmark[target]['Persistence'] = {'metrics': {'test': persist_m}, 'pred': {'test': y_persist}}

# =====================================================================
#  QUANTILE REGRESSION + CQR
# =====================================================================
print("\n" + "=" * 78)
print("[5/7] Quantile Regression + Conformal Calibration...")

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
conformal = {}

for target in TARGET_COLS:
    tname = 'Demand' if 'demand' in target else 'Price'
    print(f"\n  --- {tname} ---")
    y_val = val[target].values
    y_test = test[target].values

    dtrain = lgb.Dataset(X_train, train[target].values, feature_name=FEATURE_COLS)
    dval = lgb.Dataset(X_val, y_val, feature_name=FEATURE_COLS, reference=dtrain)

    q_preds = {'val': {}, 'test': {}}
    for q in quantiles:
        m_q = lgb.train(
            {'objective': 'quantile', 'alpha': q, 'metric': 'quantile',
             'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.03,
             'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
             'min_child_samples': 30, 'n_estimators': 2000, 'early_stopping_rounds': 100,
             'verbose': -1, 'random_state': 42, 'n_jobs': -1},
            dtrain, valid_sets=[dval], valid_names=['val'],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        m_q.save_model(os.path.join(MODEL_DIR, f"lgb_q{int(q*100)}_{target}_v3.txt"))
        q_preds['val'][q] = m_q.predict(X_val)
        q_preds['test'][q] = m_q.predict(X_test)

    # CQR calibration
    conformity = np.maximum(
        q_preds['val'][0.10] - y_val,
        y_val - q_preds['val'][0.90]
    )
    n = len(conformity)
    q_level = min(np.ceil((n + 1) * 0.90) / n, 1.0)
    Q_hat = np.quantile(conformity, q_level)

    q_lo = q_preds['test'][0.10] - Q_hat
    q_hi = q_preds['test'][0.90] + Q_hat
    q25 = q_preds['test'][0.25] - Q_hat * 0.5
    q75 = q_preds['test'][0.75] + Q_hat * 0.5

    cov90 = np.mean((y_test >= q_lo) & (y_test <= q_hi))
    width90 = np.mean(q_hi - q_lo)

    print(f"    Q_hat = {Q_hat:.1f}")
    print(f"    90% PI coverage: {cov90*100:.1f}% (target: 90%)")
    print(f"    90% PI width:    {width90:.1f}")

    conformal[target] = {
        'Q_hat': Q_hat, 'q_lo': q_lo, 'q_hi': q_hi,
        'q25': q25, 'q75': q75,
        'coverage_90': cov90, 'width_90': width90,
    }

# =====================================================================
#  PUBLICATION FIGURES
# =====================================================================
print("\n" + "=" * 78)
print("[6/7] Generating publication figures...")

MODEL_NAMES = ['LightGBM', 'XGBoost', 'RandomForest', 'Persistence']
COLORS = {'LightGBM': '#4CAF50', 'XGBoost': '#2196F3', 'RandomForest': '#FF9800', 'Persistence': '#9E9E9E'}

# --- Fig 08: Benchmark Comparison ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, target in enumerate(TARGET_COLS):
    tname = 'Demand (MW)' if 'demand' in target else 'Price ($/MWh)'

    # MAE
    ax = axes[0, idx]
    names_all = MODEL_NAMES
    test_maes = [benchmark[target][m]['metrics']['test']['MAE'] for m in names_all if m in benchmark[target]]
    names_plot = [m for m in names_all if m in benchmark[target]]
    colors_bar = [COLORS[m] for m in names_plot]

    if 'demand' in target:
        # Also show train/val for ML models
        x = np.arange(len(names_plot))
        ml_names = [m for m in names_plot if m != 'Persistence']
        w = 0.25
        train_maes = [benchmark[target][m]['metrics']['train']['MAE'] for m in ml_names]
        val_maes = [benchmark[target][m]['metrics']['val']['MAE'] for m in ml_names]
        test_maes_ml = [benchmark[target][m]['metrics']['test']['MAE'] for m in ml_names]

        ax.bar(np.arange(len(ml_names)) - w, train_maes, w, color='#81C784', label='Train', edgecolor='black', linewidth=0.5)
        ax.bar(np.arange(len(ml_names)), val_maes, w, color='#64B5F6', label='Validation', edgecolor='black', linewidth=0.5)
        ax.bar(np.arange(len(ml_names)) + w, test_maes_ml, w, color='#E57373', label='Test', edgecolor='black', linewidth=0.5)

        # Persistence as horizontal line
        persist_mae = benchmark[target]['Persistence']['metrics']['test']['MAE']
        ax.axhline(persist_mae, color='#9E9E9E', linestyle='--', linewidth=2, label=f'Persistence ({persist_mae:.0f})')

        ax.set_xticks(np.arange(len(ml_names)))
        ax.set_xticklabels(ml_names)
        ax.legend(fontsize=8)

        for i, (tr, va, te) in enumerate(zip(train_maes, val_maes, test_maes_ml)):
            ax.text(i + w, te + max(test_maes_ml)*0.02, f'{te:.0f}', ha='center', fontsize=8, fontweight='bold')
    else:
        bars = ax.bar(range(len(names_plot)), test_maes, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names_plot)))
        ax.set_xticklabels(names_plot)

    ax.set_ylabel('MAE')
    ax.set_title(f'{tname} — MAE (Day-Ahead)', fontweight='bold')

    # R2
    ax = axes[1, idx]
    test_r2s = [benchmark[target][m]['metrics']['test']['R2'] for m in names_plot]
    bars = ax.bar(range(len(names_plot)), test_r2s, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names_plot)))
    ax.set_xticklabels(names_plot)
    ax.set_ylabel('R²')
    ax.set_title(f'{tname} — R² Score (Test Set)', fontweight='bold')

    for i, v in enumerate(test_r2s):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

fig.suptitle('Day-Ahead Forecast Benchmark (24h Horizon, No Leakage)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig08_benchmark_dayahead.png"))
plt.close()
print("  -> fig08_benchmark_dayahead.png")

# --- Fig 09: Best Model Predictions with PI ---
for idx, target in enumerate(TARGET_COLS):
    tname = 'Demand' if 'demand' in target else 'Price'
    best = best_models[target]
    y_test = test[target].values
    y_pred = benchmark[target][best]['pred']['test']
    cr = conformal[target]
    m = benchmark[target][best]['metrics']['test']
    unit = 'MW' if 'demand' in target else '$/MWh'

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    test_idx = test.index

    # Full period
    ax = axes[0]
    ax.fill_between(test_idx, cr['q_lo'], cr['q_hi'], alpha=0.15, color='#1565C0', label='90% PI (CQR)')
    ax.fill_between(test_idx, cr['q25'], cr['q75'], alpha=0.3, color='#42A5F5', label='50% PI')
    ax.plot(test_idx, y_test, color='#333333', linewidth=0.8, alpha=0.8, label='Actual')
    ax.plot(test_idx, y_pred, color='#E53935', linewidth=0.8, alpha=0.7, label=f'{best}', linestyle='--')
    ax.set_ylabel(f'{tname} ({unit})')
    ax.set_title(f'{tname}: Day-Ahead Forecast ({best})\n'
                 f'MAE={m["MAE"]:.1f} {unit}  |  RMSE={m["RMSE"]:.1f} {unit}  |  '
                 f'R²={m["R2"]:.4f}  |  MAPE={m["MAPE"]:.2f}%  |  90% Coverage={cr["coverage_90"]*100:.1f}%',
                 fontweight='bold', fontsize=11)
    ax.legend(loc='upper right', ncol=5, fontsize=8)

    # Zoomed 1 week
    ax = axes[1]
    n_week = min(168, len(test_idx) - 1)
    sl = slice(0, n_week + 1)
    ax.fill_between(test_idx[sl], cr['q_lo'][sl], cr['q_hi'][sl], alpha=0.15, color='#1565C0', label='90% PI')
    ax.fill_between(test_idx[sl], cr['q25'][sl], cr['q75'][sl], alpha=0.3, color='#42A5F5', label='50% PI')
    ax.plot(test_idx[sl], y_test[sl], 'k-', linewidth=1.5, marker='o', markersize=2, label='Actual')
    ax.plot(test_idx[sl], y_pred[sl], 'r--', linewidth=1.2, label=f'{best}')

    misses = (y_test[sl] < cr['q_lo'][sl]) | (y_test[sl] > cr['q_hi'][sl])
    if misses.any():
        ax.scatter(test_idx[sl][misses], y_test[sl][misses], color='red', s=30, zorder=5, marker='x', label='PI Miss')

    ax.set_ylabel(f'{tname} ({unit})')
    ax.set_xlabel('Date')
    ax.set_title('Zoomed: First Week of Test Period', fontweight='bold')
    ax.legend(loc='upper right', ncol=5, fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig{9+idx:02d}_{tname.lower()}_dayahead.png"))
    plt.close()
    print(f"  -> fig{9+idx:02d}_{tname.lower()}_dayahead.png")

# --- Fig 11: Overfit Diagnostic ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, target in enumerate(TARGET_COLS):
    ax = axes[idx]
    tname = 'Demand' if 'demand' in target else 'Price'
    ml_models = ['LightGBM', 'XGBoost', 'RandomForest']

    r2_data = {split: [benchmark[target][m]['metrics'][split]['R2'] for m in ml_models]
               for split in ['train', 'val', 'test']}

    x = np.arange(len(ml_models))
    w = 0.2
    colors_split = ['#81C784', '#64B5F6', '#E57373']
    for i, (split, vals) in enumerate(r2_data.items()):
        ax.bar(x + i*w - w, vals, w, label=split.title(), color=colors_split[i],
               alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ml_models)
    ax.set_ylabel('R²')
    ax.set_title(f'{tname}: Overfit Check', fontweight='bold')
    ax.legend()

    for m_idx, m_name in enumerate(ml_models):
        gap = r2_data['train'][m_idx] - r2_data['test'][m_idx]
        y_pos = min(r2_data['test'][m_idx], r2_data['val'][m_idx])
        ax.annotate(f'Gap: {gap:.3f}', xy=(m_idx, y_pos),
                    xytext=(m_idx, y_pos - 0.06), ha='center', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig11_overfit_diagnostic.png"))
plt.close()
print("  -> fig11_overfit_diagnostic.png")

# --- Fig 12: Feature Importance ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for idx, target in enumerate(TARGET_COLS):
    ax = axes[idx]
    tname = 'Demand' if 'demand' in target else 'Price'
    best = best_models[target]
    imp = benchmark[target][best]['importance'].head(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp)))
    ax.barh(range(len(imp)), imp['importance'].values, color=colors)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(f'{tname} — Top Features ({best})', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig12_feature_importance_dayahead.png"))
plt.close()
print("  -> fig12_feature_importance_dayahead.png")

# --- Fig 10: Error Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, target in enumerate(TARGET_COLS):
    tname = 'Demand' if 'demand' in target else 'Price'
    best = best_models[target]
    y_test = test[target].values
    y_pred = benchmark[target][best]['pred']['test']
    errors = y_test - y_pred
    unit = 'MW' if 'demand' in target else '$/MWh'

    # Error histogram
    ax = axes[0, idx]
    ax.hist(errors, bins=50, density=True, alpha=0.7, color='#5E35B1', edgecolor='white')
    ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(np.mean(errors), color='orange', linewidth=1.5, linestyle=':',
               label=f'Mean: {np.mean(errors):.1f} {unit}')
    ax.axvline(np.median(errors), color='green', linewidth=1.5, linestyle=':',
               label=f'Median: {np.median(errors):.1f} {unit}')
    ax.set_title(f'{tname}: Error Distribution', fontweight='bold')
    ax.set_xlabel(f'Error ({unit})')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    # Scatter: actual vs predicted
    ax = axes[1, idx]
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='#1565C0')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel(f'Actual ({unit})')
    ax.set_ylabel(f'Predicted ({unit})')
    r2 = benchmark[target][best]['metrics']['test']['R2']
    ax.set_title(f'{tname}: Actual vs Predicted (R²={r2:.4f})', fontweight='bold')
    ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig10_error_analysis_dayahead.png"))
plt.close()
print("  -> fig10_error_analysis_dayahead.png")

# =====================================================================
#  SAVE ALL RESULTS
# =====================================================================
print("\n" + "=" * 78)
print("[7/7] Saving results...")

# Benchmark table
rows = []
for target in TARGET_COLS:
    tname = 'Demand' if 'demand' in target else 'Price'
    for mname in MODEL_NAMES:
        if mname not in benchmark[target]:
            continue
        m = benchmark[target][mname]['metrics']
        t_time = benchmark[target][mname].get('time', 0)

        row = {'Target': tname, 'Model': mname}
        for split in ['train', 'val', 'test']:
            if split in m:
                for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
                    if metric in m[split]:
                        row[f'{split}_{metric}'] = m[split][metric]
        row['Train_Time_s'] = t_time
        if 'train' in m and 'test' in m and 'R2' in m['train'] and 'R2' in m['test']:
            row['Overfit_Gap_R2'] = m['train']['R2'] - m['test']['R2']
        rows.append(row)

bench_df = pd.DataFrame(rows)
bench_df.to_csv(os.path.join(TABLE_DIR, "benchmark_dayahead.csv"), index=False)
print("  -> benchmark_dayahead.csv")

# ML summary (best models)
summary_rows = []
for target in TARGET_COLS:
    tname = 'Demand' if 'demand' in target else 'Price'
    best = best_models[target]
    m = benchmark[target][best]['metrics']['test']
    cr = conformal[target]
    summary_rows.append({
        'Target': tname, 'Best_Model': best,
        'Forecast_Horizon': '24h (day-ahead)',
        'N_Features': len(FEATURE_COLS),
        'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'], 'MAPE': m['MAPE'],
        '90% Coverage': cr['coverage_90'] * 100,
        '90% PI Width': cr['width_90'],
        'Q_hat': cr['Q_hat'],
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TABLE_DIR, "ml_results_summary.csv"), index=False)
print("  -> ml_results_summary.csv")

# Feature importance
for target in TARGET_COLS:
    tname = 'demand' if 'demand' in target else 'price'
    best = best_models[target]
    benchmark[target][best]['importance'].to_csv(
        os.path.join(TABLE_DIR, f"feature_importance_{tname}_dayahead.csv"), index=False)

# Predictions for MILP
best_demand = best_models['target_demand_MW']
best_price = best_models['price_proxy_USD_MWh']
predictions = pd.DataFrame({
    'timestamp': test.index,
    'net_load_actual': test['target_demand_MW'].values,
    'net_load_predicted': benchmark['target_demand_MW'][best_demand]['pred']['test'],
    'net_load_Q10': conformal['target_demand_MW']['q_lo'],
    'net_load_Q90': conformal['target_demand_MW']['q_hi'],
    'price_actual': test['price_proxy_USD_MWh'].values,
    'price_predicted': benchmark['price_proxy_USD_MWh'][best_price]['pred']['test'],
    'price_Q10': conformal['price_proxy_USD_MWh']['q_lo'],
    'price_Q90': conformal['price_proxy_USD_MWh']['q_hi'],
})
predictions.to_csv(os.path.join(DATA_DIR, "ml_predictions_for_milp.csv"), index=False)
print(f"  -> ml_predictions_for_milp.csv ({len(predictions)} rows)")

# Feature config for reproducibility
feature_config = {
    'feature_cols': FEATURE_COLS,
    'target_cols': TARGET_COLS,
    'forecast_horizon': '24h (day-ahead)',
    'train_end': train_end,
    'val_end': val_end,
    'n_features': len(FEATURE_COLS),
    'n_train': len(train),
    'n_val': len(val),
    'n_test': len(test),
    'best_models': best_models,
    'leakage_prevention': 'No features with lag < 24h. No rolling stats at target time.',
}
with open(os.path.join(DATA_DIR, "feature_config.json"), 'w') as f:
    json.dump(feature_config, f, indent=2)
print("  -> feature_config.json")

# =====================================================================
#  FINAL REPORT
# =====================================================================
print("\n" + "=" * 78)
print("  DAY-AHEAD FORECAST RESULTS — PUBLICATION-READY")
print("=" * 78)

print(f"\n  Forecast Horizon: 24 hours (day-ahead)")
print(f"  Features: {len(FEATURE_COLS)} (all available at forecast time)")
print(f"  Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")

print(f"\n  {'Model':<15} {'Target':<10} {'Test MAE':>10} {'Test R2':>10} {'Train R2':>10} {'Gap':>8} {'vs Persist':>12}")
print(f"  {'-'*75}")

for target in TARGET_COLS:
    tname = 'Demand' if 'demand' in target else 'Price'
    persist_mae = benchmark[target]['Persistence']['metrics']['test']['MAE']

    for mname in ['LightGBM', 'XGBoost', 'RandomForest']:
        m = benchmark[target][mname]['metrics']
        gap = m['train']['R2'] - m['test']['R2']
        skill = (1 - m['test']['MAE'] / persist_mae) * 100
        marker = ' <<' if mname == best_models[target] else ''
        print(f"  {mname:<15} {tname:<10} {m['test']['MAE']:>10.1f} {m['test']['R2']:>10.4f} "
              f"{m['train']['R2']:>10.4f} {gap:>8.4f} {skill:>+10.1f}%{marker}")

    print(f"  {'Persistence':<15} {tname:<10} {persist_mae:>10.1f} "
          f"{benchmark[target]['Persistence']['metrics']['test']['R2']:>10.4f} {'—':>10} {'—':>8} {'baseline':>12}")
    print()

print("  Note: Skill% = (1 - MAE_model/MAE_persistence) × 100")
print("        Positive skill = model beats persistence baseline")
print("=" * 78)
