"""
=============================================================================
 PHASE 2 (FIXED): ML UNCERTAINTY MODELING — NO DATA LEAKAGE
 Conformalized Quantile Regression with LightGBM + XGBoost + RandomForest
=============================================================================
 FIX: Previous version had data leakage:
   - net_load_MW = total_demand_MW (target == feature)
   - Contemporaneous features (net_generation_MW, Total interchange) removed
   - Only features available at forecast time are kept

 Benchmark: LightGBM vs XGBoost vs RandomForest
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
    print(f"  XGBoost version: {xgb.__version__}")
except ImportError:
    print("  Installing XGBoost...")
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
print("  PHASE 2 (FIXED): ML MODELING — LEAKAGE-FREE + BENCHMARK")
print("=" * 78)

# =====================================================================
#  LOAD DATA & FIX LEAKAGE
# =====================================================================
print("\n[1/6] Loading data and removing leaked features...")

train = pd.read_csv(os.path.join(DATA_DIR, "train_2023.csv"), index_col=0, parse_dates=True)
val = pd.read_csv(os.path.join(DATA_DIR, "val_2023.csv"), index_col=0, parse_dates=True)
test = pd.read_csv(os.path.join(DATA_DIR, "test_2023.csv"), index_col=0, parse_dates=True)

with open(os.path.join(DATA_DIR, "feature_config.json"), 'r') as f:
    config = json.load(f)

old_features = config['feature_cols']
target_cols = config['target_cols']

# ---- LEAKAGE DIAGNOSIS ----
print("\n  LEAKAGE DIAGNOSIS:")
print(f"  Target: net_load_MW = total_demand_MW (set in preprocessing line 124)")
print(f"  Problem: total_demand_MW IS the target, used as feature -> R2=0.9995")
print(f"  Problem: net_generation_MW, Total interchange are contemporaneous")

# Features to REMOVE (not available at forecast time or are the target itself)
LEAKED_FEATURES = {
    'total_demand_MW',         # = target itself
    'net_generation_MW',       # contemporaneous, unknown at forecast time
    'Total interchange',       # contemporaneous, unknown at forecast time
    'demand_ramp',             # diff of total_demand_MW = diff of target
    'demand_ramp_24h',         # diff of target
    'demand_same_hour_yesterday',  # total_demand_MW shifted = target shifted (redundant with net_load lag)
}

clean_features = [f for f in old_features if f not in LEAKED_FEATURES]

print(f"\n  Original features: {len(old_features)}")
print(f"  Removed (leaked):  {len(old_features) - len(clean_features)}")
print(f"  Clean features:    {len(clean_features)}")
print(f"\n  Removed features:")
for f in sorted(LEAKED_FEATURES):
    if f in old_features:
        print(f"    X  {f}")

print(f"\n  Retained features:")
for f in clean_features:
    print(f"    +  {f}")

# Filter to existing columns
clean_features = [c for c in clean_features if c in train.columns]

X_train = train[clean_features].values
X_val = val[clean_features].values
X_test = test[clean_features].values

print(f"\n  Final shapes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

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
#  BENCHMARK: LightGBM vs XGBoost vs Random Forest
# =====================================================================
print("\n" + "=" * 78)
print("[2/6] Training 3 model benchmarks (LightGBM, XGBoost, RandomForest)...")

benchmark_results = {}

for target in target_cols:
    print(f"\n{'='*60}")
    print(f"  Target: {target}")
    print(f"{'='*60}")

    y_train = train[target].values
    y_val = val[target].values
    y_test = test[target].values

    benchmark_results[target] = {}

    # --- 1. LightGBM ---
    print(f"\n  [LightGBM] Training...")
    t0 = time.time()

    lgb_params = {
        'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
        'num_leaves': 127, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 20, 'n_estimators': 2000,
        'early_stopping_rounds': 50, 'verbose': -1, 'random_state': 42, 'n_jobs': -1,
    }

    dtrain_lgb = lgb.Dataset(X_train, y_train, feature_name=clean_features)
    dval_lgb = lgb.Dataset(X_val, y_val, feature_name=clean_features, reference=dtrain_lgb)

    lgb_model = lgb.train(
        lgb_params, dtrain_lgb,
        valid_sets=[dtrain_lgb, dval_lgb], valid_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(period=500)],
    )

    lgb_time = time.time() - t0
    lgb_pred_test = lgb_model.predict(X_test)
    lgb_pred_val = lgb_model.predict(X_val)
    lgb_pred_train = lgb_model.predict(X_train)

    m_train_lgb = calc_metrics(y_train, lgb_pred_train)
    m_val_lgb = calc_metrics(y_val, lgb_pred_val)
    m_test_lgb = calc_metrics(y_test, lgb_pred_test)

    print(f"    Train:  MAE={m_train_lgb['MAE']:>8.1f}  R2={m_train_lgb['R2']:.4f}")
    print(f"    Val:    MAE={m_val_lgb['MAE']:>8.1f}  R2={m_val_lgb['R2']:.4f}")
    print(f"    Test:   MAE={m_test_lgb['MAE']:>8.1f}  R2={m_test_lgb['R2']:.4f}")
    print(f"    Time:   {lgb_time:.1f}s")

    # Feature importance
    lgb_importance = pd.DataFrame({
        'feature': clean_features,
        'importance': lgb_model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)

    benchmark_results[target]['LightGBM'] = {
        'model': lgb_model, 'pred_test': lgb_pred_test, 'pred_val': lgb_pred_val,
        'metrics_train': m_train_lgb, 'metrics_val': m_val_lgb, 'metrics_test': m_test_lgb,
        'time': lgb_time, 'importance': lgb_importance,
    }

    # --- 2. XGBoost ---
    print(f"\n  [XGBoost] Training...")
    t0 = time.time()

    xgb_model = xgb.XGBRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        early_stopping_rounds=50, eval_metric='mae',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_time = time.time() - t0
    xgb_pred_test = xgb_model.predict(X_test)
    xgb_pred_val = xgb_model.predict(X_val)
    xgb_pred_train = xgb_model.predict(X_train)

    m_train_xgb = calc_metrics(y_train, xgb_pred_train)
    m_val_xgb = calc_metrics(y_val, xgb_pred_val)
    m_test_xgb = calc_metrics(y_test, xgb_pred_test)

    print(f"    Train:  MAE={m_train_xgb['MAE']:>8.1f}  R2={m_train_xgb['R2']:.4f}")
    print(f"    Val:    MAE={m_val_xgb['MAE']:>8.1f}  R2={m_val_xgb['R2']:.4f}")
    print(f"    Test:   MAE={m_test_xgb['MAE']:>8.1f}  R2={m_test_xgb['R2']:.4f}")
    print(f"    Time:   {xgb_time:.1f}s")

    xgb_importance = pd.DataFrame({
        'feature': clean_features,
        'importance': xgb_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    benchmark_results[target]['XGBoost'] = {
        'model': xgb_model, 'pred_test': xgb_pred_test, 'pred_val': xgb_pred_val,
        'metrics_train': m_train_xgb, 'metrics_val': m_val_xgb, 'metrics_test': m_test_xgb,
        'time': xgb_time, 'importance': xgb_importance,
    }

    # --- 3. Random Forest ---
    print(f"\n  [RandomForest] Training...")
    t0 = time.time()

    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_leaf=10,
        max_features=0.8, random_state=42, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_time = time.time() - t0
    rf_pred_test = rf_model.predict(X_test)
    rf_pred_val = rf_model.predict(X_val)
    rf_pred_train = rf_model.predict(X_train)

    m_train_rf = calc_metrics(y_train, rf_pred_train)
    m_val_rf = calc_metrics(y_val, rf_pred_val)
    m_test_rf = calc_metrics(y_test, rf_pred_test)

    print(f"    Train:  MAE={m_train_rf['MAE']:>8.1f}  R2={m_train_rf['R2']:.4f}")
    print(f"    Val:    MAE={m_val_rf['MAE']:>8.1f}  R2={m_val_rf['R2']:.4f}")
    print(f"    Test:   MAE={m_test_rf['MAE']:>8.1f}  R2={m_test_rf['R2']:.4f}")
    print(f"    Time:   {rf_time:.1f}s")

    rf_importance = pd.DataFrame({
        'feature': clean_features,
        'importance': rf_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    benchmark_results[target]['RandomForest'] = {
        'model': rf_model, 'pred_test': rf_pred_test, 'pred_val': rf_pred_val,
        'metrics_train': m_train_rf, 'metrics_val': m_val_rf, 'metrics_test': m_test_rf,
        'time': rf_time, 'importance': rf_importance,
    }

# =====================================================================
#  SELECT BEST MODEL & TRAIN QUANTILE MODELS
# =====================================================================
print("\n" + "=" * 78)
print("[3/6] Selecting best model per target...")

best_models = {}
for target in target_cols:
    models = benchmark_results[target]
    best_name = min(models.keys(), key=lambda k: models[k]['metrics_val']['MAE'])
    best_models[target] = best_name
    print(f"  {target}: Best = {best_name} (Val MAE = {models[best_name]['metrics_val']['MAE']:.1f})")

print("\n" + "=" * 78)
print("[4/6] Training LightGBM Quantile Regression models (for CQR)...")

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
quantile_preds = {}

for target in target_cols:
    print(f"\n  --- {target} ---")
    y_train = train[target].values
    y_val = val[target].values

    dtrain_lgb = lgb.Dataset(X_train, y_train, feature_name=clean_features)
    dval_lgb = lgb.Dataset(X_val, y_val, feature_name=clean_features, reference=dtrain_lgb)

    quantile_preds[target] = {'val': {}, 'test': {}}

    for q in quantiles:
        params_q = {
            'objective': 'quantile', 'alpha': q, 'metric': 'quantile',
            'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'min_child_samples': 20, 'n_estimators': 1500,
            'early_stopping_rounds': 50, 'verbose': -1, 'random_state': 42, 'n_jobs': -1,
        }

        model_q = lgb.train(
            params_q, dtrain_lgb,
            valid_sets=[dval_lgb], valid_names=['val'],
            callbacks=[lgb.log_evaluation(period=0)],
        )

        model_q.save_model(os.path.join(MODEL_DIR, f"lgb_q{int(q*100)}_{target}_fixed.txt"))
        quantile_preds[target]['val'][q] = model_q.predict(X_val)
        quantile_preds[target]['test'][q] = model_q.predict(X_test)

    print(f"    Trained {len(quantiles)} quantile models")

# =====================================================================
#  CONFORMAL PREDICTION
# =====================================================================
print("\n" + "=" * 78)
print("[5/6] Conformal Prediction calibration (CQR)...")

alpha = 0.10
conformal_results = {}

for target in target_cols:
    print(f"\n  --- {target} ---")

    y_val = val[target].values
    y_test = test[target].values

    q_lo_val = quantile_preds[target]['val'][0.10]
    q_hi_val = quantile_preds[target]['val'][0.90]

    conformity_scores = np.maximum(q_lo_val - y_val, y_val - q_hi_val)

    n = len(conformity_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    Q_hat = np.quantile(conformity_scores, q_level)

    q_lo_test = quantile_preds[target]['test'][0.10] - Q_hat
    q_hi_test = quantile_preds[target]['test'][0.90] + Q_hat
    q25_test = quantile_preds[target]['test'][0.25] - Q_hat * 0.5
    q75_test = quantile_preds[target]['test'][0.75] + Q_hat * 0.5

    coverage_90 = np.mean((y_test >= q_lo_test) & (y_test <= q_hi_test))
    width_90 = np.mean(q_hi_test - q_lo_test)

    print(f"    Q_hat = {Q_hat:.2f}")
    print(f"    90% PI Coverage: {coverage_90*100:.1f}% (target: 90%)")
    print(f"    90% PI Width:    {width_90:.1f}")

    conformal_results[target] = {
        'Q_hat': Q_hat, 'q_lo_test': q_lo_test, 'q_hi_test': q_hi_test,
        'q25_test': q25_test, 'q75_test': q75_test,
        'coverage_90': coverage_90, 'width_90': width_90,
    }

# =====================================================================
#  FIGURES & TABLES
# =====================================================================
print("\n" + "=" * 78)
print("[6/6] Generating figures and tables...")

# --- Figure 8 (fixed): Benchmark Comparison Bar Chart ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, target in enumerate(target_cols):
    models_dict = benchmark_results[target]
    model_names = list(models_dict.keys())
    colors = ['#4CAF50', '#2196F3', '#FF9800']

    # MAE comparison
    ax = axes[0, idx]
    maes_train = [models_dict[m]['metrics_train']['MAE'] for m in model_names]
    maes_val = [models_dict[m]['metrics_val']['MAE'] for m in model_names]
    maes_test = [models_dict[m]['metrics_test']['MAE'] for m in model_names]

    x = np.arange(len(model_names))
    w = 0.25
    ax.bar(x - w, maes_train, w, color='#81C784', label='Train', edgecolor='black', linewidth=0.5)
    ax.bar(x, maes_val, w, color='#64B5F6', label='Validation', edgecolor='black', linewidth=0.5)
    ax.bar(x + w, maes_test, w, color='#E57373', label='Test', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('MAE')
    unit = 'MW' if 'load' in target else '$/MWh'
    ax.set_title(f'{target}\nMAE ({unit})', fontweight='bold')
    ax.legend()

    for i, (tr, va, te) in enumerate(zip(maes_train, maes_val, maes_test)):
        ax.text(i + w, te + max(maes_test)*0.02, f'{te:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # R2 comparison
    ax = axes[1, idx]
    r2_train = [models_dict[m]['metrics_train']['R2'] for m in model_names]
    r2_val = [models_dict[m]['metrics_val']['R2'] for m in model_names]
    r2_test = [models_dict[m]['metrics_test']['R2'] for m in model_names]

    ax.bar(x - w, r2_train, w, color='#81C784', label='Train', edgecolor='black', linewidth=0.5)
    ax.bar(x, r2_val, w, color='#64B5F6', label='Validation', edgecolor='black', linewidth=0.5)
    ax.bar(x + w, r2_test, w, color='#E57373', label='Test', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('R²')
    ax.set_title(f'{target}\nR² Score', fontweight='bold')
    ax.legend()

    for i, te in enumerate(r2_test):
        ax.text(i + w, te + 0.005, f'{te:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

fig.suptitle('Model Benchmark: LightGBM vs XGBoost vs RandomForest (No Data Leakage)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig08_benchmark_comparison.png"))
plt.close()
print("  -> fig08_benchmark_comparison.png")

# --- Figure 9 (fixed): Best Model Predictions ---
for idx, target in enumerate(target_cols):
    best_name = best_models[target]
    y_test = test[target].values
    y_pred = benchmark_results[target][best_name]['pred_test']
    cr = conformal_results[target]
    metrics = benchmark_results[target][best_name]['metrics_test']

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    test_idx = test.index

    ax = axes[0]
    ax.fill_between(test_idx, cr['q_lo_test'], cr['q_hi_test'],
                    alpha=0.15, color='#1565C0', label='90% PI (CQR)')
    ax.fill_between(test_idx, cr['q25_test'], cr['q75_test'],
                    alpha=0.3, color='#42A5F5', label='50% PI')
    ax.plot(test_idx, y_test, color='#333333', linewidth=0.8, alpha=0.8, label='Actual')
    ax.plot(test_idx, y_pred, color='#E53935', linewidth=0.8, alpha=0.7, label=f'Predicted ({best_name})', linestyle='--')

    unit = 'MW' if 'load' in target else '$/MWh'
    ax.set_title(f'{target}: {best_name} — Test Period (LEAKAGE FIXED)\n'
                 f'MAE={metrics["MAE"]:.1f} {unit}  |  RMSE={metrics["RMSE"]:.1f} {unit}  |  '
                 f'R²={metrics["R2"]:.4f}  |  90% Coverage={cr["coverage_90"]*100:.1f}%',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel(unit)
    ax.legend(loc='upper right', ncol=4)

    ax = axes[1]
    week_end_idx = min(168, len(test_idx) - 1)
    mask = np.arange(week_end_idx + 1)

    ax.fill_between(test_idx[mask], cr['q_lo_test'][mask], cr['q_hi_test'][mask],
                    alpha=0.15, color='#1565C0', label='90% PI')
    ax.fill_between(test_idx[mask], cr['q25_test'][mask], cr['q75_test'][mask],
                    alpha=0.3, color='#42A5F5', label='50% PI')
    ax.plot(test_idx[mask], y_test[mask], color='#333333', linewidth=1.5, marker='o', markersize=2, label='Actual')
    ax.plot(test_idx[mask], y_pred[mask], color='#E53935', linewidth=1.2, linestyle='--', label='Predicted')

    misses = (y_test[mask] < cr['q_lo_test'][mask]) | (y_test[mask] > cr['q_hi_test'][mask])
    if misses.any():
        ax.scatter(test_idx[mask][misses], y_test[mask][misses],
                   color='red', s=30, zorder=5, marker='x', label='90% PI Miss')

    ax.set_title('Zoomed: First Week of Test Period', fontweight='bold')
    ax.set_ylabel(unit)
    ax.set_xlabel('Date')
    ax.legend(loc='upper right', ncol=5)

    plt.tight_layout()
    fig_name = f"fig{9+idx}_ml_{target.split('_')[0]}_prediction_fixed.png"
    fig.savefig(os.path.join(FIG_DIR, fig_name))
    plt.close()
    print(f"  -> {fig_name}")

# --- Figure 11 (fixed): Overfit Check (Train vs Val vs Test) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, target in enumerate(target_cols):
    ax = axes[idx]
    model_names = list(benchmark_results[target].keys())

    r2_data = {
        'Train': [benchmark_results[target][m]['metrics_train']['R2'] for m in model_names],
        'Val': [benchmark_results[target][m]['metrics_val']['R2'] for m in model_names],
        'Test': [benchmark_results[target][m]['metrics_test']['R2'] for m in model_names],
    }

    x = np.arange(len(model_names))
    w = 0.2
    for i, (split, vals) in enumerate(r2_data.items()):
        bars = ax.bar(x + i * w - w, vals, w, label=split, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('R²')
    ax.set_title(f'{target}: Overfit Check (Train vs Val vs Test)', fontweight='bold')
    ax.legend()

    for m_idx, m_name in enumerate(model_names):
        gap = r2_data['Train'][m_idx] - r2_data['Test'][m_idx]
        ax.annotate(f'Gap: {gap:.4f}', xy=(m_idx, min(r2_data['Test'][m_idx], r2_data['Val'][m_idx])),
                    xytext=(m_idx, min(r2_data['Test'][m_idx], r2_data['Val'][m_idx]) - 0.05),
                    ha='center', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig11_overfit_check.png"))
plt.close()
print("  -> fig11_overfit_check.png")

# --- Figure 12 (fixed): Feature Importance (top model) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for idx, target in enumerate(target_cols):
    ax = axes[idx]
    best_name = best_models[target]
    imp = benchmark_results[target][best_name]['importance'].head(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp)))
    ax.barh(range(len(imp)), imp['importance'].values, color=colors)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(f'Top 15 Features: {target} ({best_name})', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig12_feature_importance_fixed.png"))
plt.close()
print("  -> fig12_feature_importance_fixed.png")

# =====================================================================
#  SAVE RESULTS
# =====================================================================
print("\n" + "=" * 78)
print("  SAVING RESULTS...")

# Benchmark comparison table
bench_rows = []
for target in target_cols:
    for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
        m = benchmark_results[target][model_name]
        bench_rows.append({
            'Target': target,
            'Model': model_name,
            'Train_MAE': m['metrics_train']['MAE'],
            'Train_R2': m['metrics_train']['R2'],
            'Val_MAE': m['metrics_val']['MAE'],
            'Val_R2': m['metrics_val']['R2'],
            'Test_MAE': m['metrics_test']['MAE'],
            'Test_RMSE': m['metrics_test']['RMSE'],
            'Test_R2': m['metrics_test']['R2'],
            'Test_MAPE': m['metrics_test']['MAPE'],
            'Train_Time_s': m['time'],
            'Overfit_Gap_R2': m['metrics_train']['R2'] - m['metrics_test']['R2'],
        })

bench_df = pd.DataFrame(bench_rows)
bench_df.to_csv(os.path.join(TABLE_DIR, "benchmark_comparison.csv"), index=False)
print("  -> benchmark_comparison.csv")

# Updated ML results summary (best model only)
summary_rows = []
for target in target_cols:
    best_name = best_models[target]
    m = benchmark_results[target][best_name]['metrics_test']
    cr = conformal_results[target]
    summary_rows.append({
        'Target': target,
        'Best_Model': best_name,
        'MAE': m['MAE'],
        'RMSE': m['RMSE'],
        'R2': m['R2'],
        'MAPE': m['MAPE'],
        '90% Coverage': cr['coverage_90'] * 100,
        '90% PI Width': cr['width_90'],
        'Q_hat (Conformal)': cr['Q_hat'],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TABLE_DIR, "ml_results_summary_fixed.csv"), index=False)
print("  -> ml_results_summary_fixed.csv")

# Save predictions for MILP (using best model)
best_net_name = best_models['net_load_MW']
best_price_name = best_models['price_proxy_USD_MWh']

predictions = pd.DataFrame({
    'timestamp': test.index,
    'net_load_actual': test['net_load_MW'].values,
    'net_load_predicted': benchmark_results['net_load_MW'][best_net_name]['pred_test'],
    'net_load_Q10': conformal_results['net_load_MW']['q_lo_test'],
    'net_load_Q90': conformal_results['net_load_MW']['q_hi_test'],
    'price_actual': test['price_proxy_USD_MWh'].values,
    'price_predicted': benchmark_results['price_proxy_USD_MWh'][best_price_name]['pred_test'],
    'price_Q10': conformal_results['price_proxy_USD_MWh']['q_lo_test'],
    'price_Q90': conformal_results['price_proxy_USD_MWh']['q_hi_test'],
})
predictions.to_csv(os.path.join(DATA_DIR, "ml_predictions_for_milp.csv"), index=False)
print(f"  -> ml_predictions_for_milp.csv ({len(predictions)} rows)")

# Feature importance tables
for target in target_cols:
    best_name = best_models[target]
    imp = benchmark_results[target][best_name]['importance']
    imp.to_csv(os.path.join(TABLE_DIR, f"feature_importance_{target}_fixed.csv"), index=False)

# =====================================================================
#  FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 78)
print("  PHASE 2 (FIXED) COMPLETE — LEAKAGE-FREE RESULTS")
print("=" * 78)

print(f"\n  {'Model':<15} {'Target':<30} {'MAE':>8} {'R2':>8} {'Train R2':>10} {'Gap':>8}")
print(f"  {'-'*80}")
for target in target_cols:
    for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
        m = benchmark_results[target][model_name]
        gap = m['metrics_train']['R2'] - m['metrics_test']['R2']
        marker = ' *' if model_name == best_models[target] else ''
        print(f"  {model_name:<15} {target:<30} {m['metrics_test']['MAE']:>8.1f} "
              f"{m['metrics_test']['R2']:>8.4f} {m['metrics_train']['R2']:>10.4f} {gap:>8.4f}{marker}")

print(f"\n  * = best model for target")
print(f"\n  Note: R2 should be REALISTIC now (not 0.9995)")
print("=" * 78)
