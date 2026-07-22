"""
=============================================================================
 PHASE 2: ML UNCERTAINTY MODELING
 Conformalized Quantile Regression (CQR) with LightGBM
=============================================================================
 Architecture:
   1. Train LightGBM for point prediction (net load & price)
   2. Train separate quantile models for Q10, Q50, Q90 (prediction intervals)
   3. Apply Conformal Prediction to calibrate intervals on validation set
   4. Evaluate coverage and interval width on test set
   5. Generate publication-quality figures
=============================================================================
"""

import os
import sys
import io
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Check for LightGBM
try:
    import lightgbm as lgb
    print(f"  LightGBM version: {lgb.__version__}")
except ImportError:
    print("  LightGBM not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR  = os.path.join(PROJECT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "outputs", "tables")
MODEL_DIR = os.path.join(PROJECT_DIR, "outputs", "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
print("  PHASE 2: ML UNCERTAINTY MODELING")
print("  Conformalized Quantile Regression (CQR) with LightGBM")
print("=" * 78)

# =====================================================================
#  LOAD DATA
# =====================================================================
print("\n[1/5] Loading preprocessed data...")

train = pd.read_csv(os.path.join(DATA_DIR, "train_2023.csv"), index_col=0, parse_dates=True)
val = pd.read_csv(os.path.join(DATA_DIR, "val_2023.csv"), index_col=0, parse_dates=True)
test = pd.read_csv(os.path.join(DATA_DIR, "test_2023.csv"), index_col=0, parse_dates=True)

with open(os.path.join(DATA_DIR, "feature_config.json"), 'r') as f:
    config = json.load(f)

feature_cols = config['feature_cols']
target_cols = config['target_cols']

# Filter to only features that exist in the dataframe
feature_cols = [c for c in feature_cols if c in train.columns]

X_train = train[feature_cols].values
X_val = val[feature_cols].values
X_test = test[feature_cols].values

print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"  Features: {len(feature_cols)}")
print(f"  Targets: {target_cols}")

# =====================================================================
#  STEP 2: POINT PREDICTION MODELS
# =====================================================================
print("\n[2/5] Training LightGBM point prediction models...")

results = {}

for target in target_cols:
    print(f"\n  --- Target: {target} ---")
    
    y_train = train[target].values
    y_val = val[target].values
    y_test = test[target].values
    
    # LightGBM parameters (tuned for energy time series)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'max_depth': -1,
        'n_estimators': 2000,
        'early_stopping_rounds': 50,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # Create datasets
    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, y_val, feature_name=feature_cols, reference=dtrain)
    
    # Train
    callbacks = [lgb.log_evaluation(period=500)]
    model = lgb.train(
        params, dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )
    
    # Save model
    model.save_model(os.path.join(MODEL_DIR, f"lgb_point_{target}.txt"))
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    def calc_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        print(f"    {name:>12}: MAE={mae:>8.1f}  RMSE={rmse:>8.1f}  R2={r2:.4f}  MAPE={mape:.2f}%")
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    
    m_train = calc_metrics(y_train, y_pred_train, "Train")
    m_val = calc_metrics(y_val, y_pred_val, "Validation")
    m_test = calc_metrics(y_test, y_pred_test, "Test")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)
    
    print(f"\n    Top 10 features:")
    for i, row in importance.head(10).iterrows():
        print(f"      {row['feature']:<40} {row['importance']:>10.0f}")
    
    results[target] = {
        'model': model,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'y_val': y_val,
        'y_pred_val': y_pred_val,
        'metrics_test': m_test,
        'importance': importance,
    }

# =====================================================================
#  STEP 3: QUANTILE REGRESSION MODELS
# =====================================================================
print("\n" + "=" * 78)
print("[3/5] Training Quantile Regression models...")
print("  Quantiles: 0.10, 0.25, 0.50, 0.75, 0.90")

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
quantile_models = {}

for target in target_cols:
    print(f"\n  --- Target: {target} ---")
    y_train = train[target].values
    y_val = val[target].values
    
    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, y_val, feature_name=feature_cols, reference=dtrain)
    
    quantile_models[target] = {}
    
    for q in quantiles:
        params_q = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'n_estimators': 1500,
            'early_stopping_rounds': 50,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        model_q = lgb.train(
            params_q, dtrain,
            valid_sets=[dval],
            valid_names=['val'],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        
        model_q.save_model(os.path.join(MODEL_DIR, f"lgb_q{int(q*100)}_{target}.txt"))
        quantile_models[target][q] = model_q
    
    # Get predictions for all quantiles
    q_preds_val = {q: m.predict(X_val) for q, m in quantile_models[target].items()}
    q_preds_test = {q: m.predict(X_test) for q, m in quantile_models[target].items()}
    
    results[target]['q_preds_val'] = q_preds_val
    results[target]['q_preds_test'] = q_preds_test
    
    print(f"    Trained {len(quantiles)} quantile models")

# =====================================================================
#  STEP 4: CONFORMAL PREDICTION CALIBRATION
# =====================================================================
print("\n" + "=" * 78)
print("[4/5] Applying Conformal Prediction calibration...")
print("  Method: Split Conformal with CQR (Romano et al., 2019)")

alpha = 0.10  # Target miscoverage rate (90% coverage)

for target in target_cols:
    print(f"\n  --- Target: {target} ---")
    
    y_val = results[target]['y_val']
    q_preds_val = results[target]['q_preds_val']
    q_preds_test = results[target]['q_preds_test']
    y_test = results[target]['y_test']
    
    # CQR: Compute conformity scores on validation set
    # E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))
    q_lo_val = q_preds_val[0.10]
    q_hi_val = q_preds_val[0.90]
    
    conformity_scores = np.maximum(q_lo_val - y_val, y_val - q_hi_val)
    
    # Compute conformal quantile
    n = len(conformity_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    Q_hat = np.quantile(conformity_scores, q_level)
    
    print(f"    Conformal correction Q_hat = {Q_hat:.2f}")
    print(f"    Effective coverage target = {(1-alpha)*100:.0f}%")
    
    # Apply correction to test set
    q_lo_test = q_preds_test[0.10] - Q_hat
    q_hi_test = q_preds_test[0.90] + Q_hat
    
    # Also get 50% PI (Q25-Q75 with conformal correction)
    q25_test = q_preds_test[0.25] - Q_hat * 0.5  # Scaled correction
    q75_test = q_preds_test[0.75] + Q_hat * 0.5
    
    # Evaluate
    coverage_90 = np.mean((y_test >= q_lo_test) & (y_test <= q_hi_test))
    width_90 = np.mean(q_hi_test - q_lo_test)
    
    coverage_50 = np.mean((y_test >= q25_test) & (y_test <= q75_test))
    width_50 = np.mean(q75_test - q25_test)
    
    print(f"\n    90% Prediction Interval:")
    print(f"      Coverage: {coverage_90*100:.1f}% (target: 90%)")
    print(f"      Avg width: {width_90:.1f}")
    print(f"    50% Prediction Interval:")
    print(f"      Coverage: {coverage_50*100:.1f}% (target: 50%)")
    print(f"      Avg width: {width_50:.1f}")
    
    # Check interval validity (coverage should be >= target)
    if coverage_90 >= 0.89:
        print(f"    [OK] 90% PI coverage is valid")
    else:
        print(f"    [WARNING] 90% PI coverage below target!")
    
    # Store calibrated results
    results[target]['q_lo_test'] = q_lo_test
    results[target]['q_hi_test'] = q_hi_test
    results[target]['q25_test'] = q25_test
    results[target]['q75_test'] = q75_test
    results[target]['coverage_90'] = coverage_90
    results[target]['width_90'] = width_90
    results[target]['Q_hat'] = Q_hat

# =====================================================================
#  STEP 5: PUBLICATION FIGURES
# =====================================================================
print("\n" + "=" * 78)
print("[5/5] Generating publication-quality figures...")

# --- Figure 8: Point Prediction vs Actual (Test Set) ---
for idx, target in enumerate(target_cols):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    y_test = results[target]['y_test']
    y_pred = results[target]['y_pred_test']
    q_lo = results[target]['q_lo_test']
    q_hi = results[target]['q_hi_test']
    q25 = results[target]['q25_test']
    q75 = results[target]['q75_test']
    
    test_idx = test.index
    
    # Top: Full test period
    ax = axes[0]
    ax.fill_between(test_idx, q_lo, q_hi, alpha=0.15, color='#1565C0', label='90% PI (CQR)')
    ax.fill_between(test_idx, q25, q75, alpha=0.3, color='#42A5F5', label='50% PI')
    ax.plot(test_idx, y_test, color='#333333', linewidth=0.8, alpha=0.8, label='Actual')
    ax.plot(test_idx, y_pred, color='#E53935', linewidth=0.8, alpha=0.7, label='Predicted', linestyle='--')
    
    unit = 'MW' if 'load' in target else '$/MWh'
    m = results[target]['metrics_test']
    ax.set_title(f'{target}: Test Period (Nov-Dec 2023)\n'
                f'MAE={m["MAE"]:.1f} {unit}  |  RMSE={m["RMSE"]:.1f} {unit}  |  '
                f'R2={m["R2"]:.4f}  |  90% Coverage={results[target]["coverage_90"]*100:.1f}%',
                fontweight='bold', fontsize=11)
    ax.set_ylabel(f'{unit}')
    ax.legend(loc='upper right', ncol=4)
    
    # Bottom: Zoomed 1 week
    ax = axes[1]
    week_start = test_idx[0]
    week_end = test_idx[min(168, len(test_idx)-1)]
    mask = (test_idx >= week_start) & (test_idx <= week_end)
    
    ax.fill_between(test_idx[mask], q_lo[mask], q_hi[mask], alpha=0.15, color='#1565C0', label='90% PI')
    ax.fill_between(test_idx[mask], q25[mask], q75[mask], alpha=0.3, color='#42A5F5', label='50% PI')
    ax.plot(test_idx[mask], y_test[mask], color='#333333', linewidth=1.5, marker='o', markersize=2, label='Actual')
    ax.plot(test_idx[mask], y_pred[mask], color='#E53935', linewidth=1.2, linestyle='--', label='Predicted')
    
    # Highlight misses
    misses = (y_test[mask] < q_lo[mask]) | (y_test[mask] > q_hi[mask])
    if misses.any():
        ax.scatter(test_idx[mask][misses], y_test[mask][misses], 
                  color='red', s=30, zorder=5, marker='x', label='90% PI Miss')
    
    ax.set_title(f'Zoomed: First Week of Test Period', fontweight='bold')
    ax.set_ylabel(f'{unit}')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right', ncol=5)
    
    plt.tight_layout()
    fig_name = f"fig{8+idx}_ml_{target.split('_')[0]}_prediction.png"
    fig.savefig(os.path.join(FIG_DIR, fig_name))
    plt.close()
    print(f"  -> {fig_name}")

# --- Figure 10: Feature Importance ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for idx, target in enumerate(target_cols):
    ax = axes[idx]
    imp = results[target]['importance'].head(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp)))
    ax.barh(range(len(imp)), imp['importance'].values, color=colors)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(f'Top 15 Features: {target}', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig10_feature_importance.png"))
plt.close()
print(f"  -> fig10_feature_importance.png")

# --- Figure 11: Prediction Error Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, target in enumerate(target_cols):
    ax = axes[idx]
    errors = results[target]['y_test'] - results[target]['y_pred_test']
    
    ax.hist(errors, bins=50, density=True, alpha=0.7, color='#5E35B1', edgecolor='white')
    ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(np.mean(errors), color='orange', linewidth=1.5, linestyle=':', label=f'Mean: {np.mean(errors):.1f}')
    ax.set_title(f'Error Distribution: {target}', fontweight='bold')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig11_error_distribution.png"))
plt.close()
print(f"  -> fig11_error_distribution.png")

# --- Figure 12: Scatter Plot Actual vs Predicted ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, target in enumerate(target_cols):
    ax = axes[idx]
    y_t = results[target]['y_test']
    y_p = results[target]['y_pred_test']
    
    ax.scatter(y_t, y_p, alpha=0.3, s=10, color='#1565C0')
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{target}: Actual vs Predicted (Test)', fontweight='bold')
    ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig12_actual_vs_predicted.png"))
plt.close()
print(f"  -> fig12_actual_vs_predicted.png")

# =====================================================================
#  SAVE RESULTS SUMMARY
# =====================================================================
print("\n" + "-" * 78)
print("  ML RESULTS SUMMARY")
print("-" * 78)

summary_rows = []
for target in target_cols:
    m = results[target]['metrics_test']
    row = {
        'Target': target,
        'MAE': m['MAE'],
        'RMSE': m['RMSE'],
        'R2': m['R2'],
        'MAPE': m['MAPE'],
        '90% Coverage': results[target]['coverage_90'] * 100,
        '90% PI Width': results[target]['width_90'],
        'Q_hat (Conformal)': results[target]['Q_hat'],
    }
    summary_rows.append(row)
    
    print(f"\n  {target}:")
    for k, v in row.items():
        if k != 'Target':
            print(f"    {k:<25}: {v:>10.2f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TABLE_DIR, "ml_results_summary.csv"), index=False)

# Save predictions for MILP phase
predictions = pd.DataFrame({
    'timestamp': test.index,
    'net_load_actual': results['net_load_MW']['y_test'],
    'net_load_predicted': results['net_load_MW']['y_pred_test'],
    'net_load_Q10': results['net_load_MW']['q_lo_test'],
    'net_load_Q90': results['net_load_MW']['q_hi_test'],
    'price_actual': results['price_proxy_USD_MWh']['y_test'],
    'price_predicted': results['price_proxy_USD_MWh']['y_pred_test'],
    'price_Q10': results['price_proxy_USD_MWh']['q_lo_test'],
    'price_Q90': results['price_proxy_USD_MWh']['q_hi_test'],
})
predictions.to_csv(os.path.join(DATA_DIR, "ml_predictions_for_milp.csv"), index=False)
print(f"\n  Predictions saved for MILP: ml_predictions_for_milp.csv ({len(predictions)} rows)")

# Feature importance tables
for target in target_cols:
    imp = results[target]['importance']
    imp.to_csv(os.path.join(TABLE_DIR, f"feature_importance_{target}.csv"), index=False)

print(f"\n  Generated figures:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('fig') and f.endswith('.png'):
        fpath = os.path.join(FIG_DIR, f)
        print(f"    * {f} ({os.path.getsize(fpath)/1024:.0f} KB)")

print("\n" + "=" * 78)
print("  PHASE 2 COMPLETE: ML models trained, calibrated, and evaluated")
print("  Uncertainty bounds (CQR) ready for Robust MILP optimization")
print("=" * 78)
