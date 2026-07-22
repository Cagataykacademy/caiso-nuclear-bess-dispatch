"""
=============================================================================
 Statistical Tests & Time Series Cross-Validation
 1. Diebold-Mariano test (pairwise model comparison)
 2. Forecast error normality (Jarque-Bera, Shapiro-Wilk)
 3. Expanding window time series CV (5 folds)
 4. Error autocorrelation (Ljung-Box)
=============================================================================
"""
import os, sys, io, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR  = os.path.join(PROJECT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "outputs", "tables")

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print("=" * 78)
print("  STATISTICAL TESTS & TIME SERIES CROSS-VALIDATION")
print("=" * 78)

# =====================================================================
#  LOAD DATA
# =====================================================================
full = pd.read_csv(os.path.join(DATA_DIR, "caiso_preprocessed_v2_2023.csv"), index_col=0, parse_dates=True)
with open(os.path.join(DATA_DIR, "feature_config.json")) as f:
    config = json.load(f)

feature_cols = [c for c in config['feature_cols'] if c in full.columns]
targets = config['target_cols']

print(f"\n  Data: {full.shape}, Features: {len(feature_cols)}, Targets: {targets}")

# =====================================================================
#  1. DIEBOLD-MARIANO TEST
# =====================================================================
print("\n" + "=" * 78)
print("[1/4] Diebold-Mariano Test (pairwise model comparison)...")

train = pd.read_csv(os.path.join(DATA_DIR, "train_2023.csv"), index_col=0, parse_dates=True)
val = pd.read_csv(os.path.join(DATA_DIR, "val_2023.csv"), index_col=0, parse_dates=True)
test = pd.read_csv(os.path.join(DATA_DIR, "test_2023.csv"), index_col=0, parse_dates=True)

X_tr, X_va, X_te = train[feature_cols].values, val[feature_cols].values, test[feature_cols].values

def dm_test(e1, e2, h=1):
    """Diebold-Mariano test. H0: equal predictive accuracy."""
    d = e1**2 - e2**2
    n = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    # Newey-West HAC variance (h-1 lags)
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        d_var += 2 * (1 - k/h) * gamma_k / n
    se = np.sqrt(d_var / n)
    if se < 1e-10:
        return 0, 1.0
    dm_stat = d_mean / se
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

dm_results = []
model_names = ['LightGBM', 'XGBoost', 'RandomForest']

for target in targets:
    tname = 'Net Load' if 'net_load' in target else 'Price'
    y_tr, y_va, y_te = train[target].values, val[target].values, test[target].values

    # Train all models
    preds = {}

    dt = lgb.Dataset(X_tr, y_tr, feature_name=feature_cols)
    dv = lgb.Dataset(X_va, y_va, feature_name=feature_cols, reference=dt)
    m = lgb.train({'objective':'regression','metric':'mae','num_leaves':63,'learning_rate':0.03,
        'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,'min_child_samples':30,
        'n_estimators':3000,'early_stopping_rounds':100,'verbose':-1,'random_state':42,'n_jobs':-1},
        dt, valid_sets=[dv], valid_names=['val'], callbacks=[lgb.log_evaluation(0)])
    preds['LightGBM'] = m.predict(X_te)

    xm = xgb.XGBRegressor(n_estimators=3000,max_depth=6,learning_rate=0.03,subsample=0.7,
        colsample_bytree=0.7,min_child_weight=30,early_stopping_rounds=100,eval_metric='mae',
        random_state=42,n_jobs=-1,verbosity=0)
    xm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds['XGBoost'] = xm.predict(X_te)

    rm = RandomForestRegressor(n_estimators=500,max_depth=15,min_samples_leaf=20,max_features=0.7,random_state=42,n_jobs=-1)
    rm.fit(X_tr, y_tr)
    preds['RandomForest'] = rm.predict(X_te)

    # Persistence
    persist_col = 'netload_lag24h' if 'net_load' in target else 'price_lag24h'
    if persist_col in test.columns:
        preds['Persistence'] = test[persist_col].values

    errors = {name: y_te - pred for name, pred in preds.items()}

    # Pairwise DM tests
    all_models = list(preds.keys())
    for i in range(len(all_models)):
        for j in range(i+1, len(all_models)):
            m1, m2 = all_models[i], all_models[j]
            dm_stat, p_val = dm_test(errors[m1], errors[m2], h=24)
            sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'ns'
            dm_results.append({
                'Target': tname, 'Model_1': m1, 'Model_2': m2,
                'DM_Statistic': dm_stat, 'p_value': p_val, 'Significance': sig,
                'Better': m1 if dm_stat < 0 else m2,
            })
            print(f"  {tname}: {m1} vs {m2} — DM={dm_stat:.3f}, p={p_val:.4f} {sig}")

dm_df = pd.DataFrame(dm_results)
dm_df.to_csv(os.path.join(TABLE_DIR, "diebold_mariano_tests.csv"), index=False)
print("  -> diebold_mariano_tests.csv")

# =====================================================================
#  2. ERROR DIAGNOSTICS
# =====================================================================
print("\n" + "=" * 78)
print("[2/4] Error diagnostics (normality, autocorrelation)...")

diag_results = []
for target in targets:
    tname = 'Net Load' if 'net_load' in target else 'Price'
    y_te = test[target].values

    for mname, pred in preds.items():
        if mname == 'Persistence':
            continue
        e = y_te - pred

        # Jarque-Bera normality test
        jb_stat, jb_p = stats.jarque_bera(e)

        # Shapiro-Wilk (on subsample if n > 5000)
        e_sub = e[:min(5000, len(e))]
        sw_stat, sw_p = stats.shapiro(e_sub)

        # Ljung-Box autocorrelation test (lag 24)
        n = len(e)
        acf_vals = np.array([np.corrcoef(e[k:], e[:-k])[0,1] for k in range(1, 25)])
        lb_stat = n * (n + 2) * np.sum(acf_vals**2 / (n - np.arange(1, 25)))
        lb_p = 1 - stats.chi2.cdf(lb_stat, df=24)

        diag_results.append({
            'Target': tname, 'Model': mname,
            'Mean_Error': np.mean(e), 'Std_Error': np.std(e),
            'Skewness': stats.skew(e), 'Kurtosis': stats.kurtosis(e),
            'JB_Stat': jb_stat, 'JB_p': jb_p, 'Normal_JB': 'Yes' if jb_p > 0.05 else 'No',
            'SW_Stat': sw_stat, 'SW_p': sw_p, 'Normal_SW': 'Yes' if sw_p > 0.05 else 'No',
            'LB_Stat': lb_stat, 'LB_p': lb_p, 'Autocorr_Free': 'Yes' if lb_p > 0.05 else 'No',
        })
        print(f"  {tname}/{mname}: mean={np.mean(e):.1f}, skew={stats.skew(e):.2f}, "
              f"kurt={stats.kurtosis(e):.2f}, JB_p={jb_p:.4f}, LB_p={lb_p:.4f}")

diag_df = pd.DataFrame(diag_results)
diag_df.to_csv(os.path.join(TABLE_DIR, "error_diagnostics.csv"), index=False)
print("  -> error_diagnostics.csv")

# =====================================================================
#  3. TIME SERIES CV (Expanding Window)
# =====================================================================
print("\n" + "=" * 78)
print("[3/4] Time Series Cross-Validation (expanding window, 5 folds)...")

n_total = len(full)
min_train = int(n_total * 0.4)  # minimum 40% for first fold
fold_size = int((n_total - min_train) / 5)

cv_results = []

for fold in range(5):
    train_end_idx = min_train + fold * fold_size
    test_start_idx = train_end_idx
    test_end_idx = min(test_start_idx + fold_size, n_total)

    if test_end_idx <= test_start_idx:
        break

    cv_train = full.iloc[:train_end_idx]
    cv_test = full.iloc[test_start_idx:test_end_idx]

    X_cv_tr = cv_train[feature_cols].values
    X_cv_te = cv_test[feature_cols].values

    print(f"\n  Fold {fold+1}: train={len(cv_train)}, test={len(cv_test)} "
          f"({cv_train.index.min().date()}->{cv_test.index.max().date()})")

    for target in targets:
        tname = 'Net Load' if 'net_load' in target else 'Price'
        y_cv_tr = cv_train[target].values
        y_cv_te = cv_test[target].values

        # LightGBM only for CV (fastest)
        dt = lgb.Dataset(X_cv_tr, y_cv_tr, feature_name=feature_cols)
        m = lgb.train({'objective':'regression','metric':'mae','num_leaves':63,'learning_rate':0.03,
            'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,'min_child_samples':30,
            'n_estimators':2000,'early_stopping_rounds':50,'verbose':-1,'random_state':42,'n_jobs':-1},
            dt, valid_sets=[lgb.Dataset(X_cv_te, y_cv_te, feature_name=feature_cols)],
            valid_names=['test'], callbacks=[lgb.log_evaluation(0)])

        pred = m.predict(X_cv_te)
        mae = mean_absolute_error(y_cv_te, pred)
        rmse = np.sqrt(mean_squared_error(y_cv_te, pred))
        r2 = r2_score(y_cv_te, pred)

        cv_results.append({
            'Fold': fold+1, 'Target': tname,
            'Train_Size': len(cv_train), 'Test_Size': len(cv_test),
            'Train_End': str(cv_train.index.max().date()),
            'Test_End': str(cv_test.index.max().date()),
            'MAE': mae, 'RMSE': rmse, 'R2': r2,
        })
        print(f"    {tname}: MAE={mae:.1f}, R2={r2:.4f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(os.path.join(TABLE_DIR, "timeseries_cv_results.csv"), index=False)
print("\n  -> timeseries_cv_results.csv")

# CV summary stats
print("\n  CV Summary (mean ± std across folds):")
for target_name in ['Net Load', 'Price']:
    sub = cv_df[cv_df['Target'] == target_name]
    print(f"  {target_name}:")
    print(f"    MAE:  {sub['MAE'].mean():.1f} ± {sub['MAE'].std():.1f}")
    print(f"    RMSE: {sub['RMSE'].mean():.1f} ± {sub['RMSE'].std():.1f}")
    print(f"    R²:   {sub['R2'].mean():.4f} ± {sub['R2'].std():.4f}")

# =====================================================================
#  4. CV STABILITY FIGURE
# =====================================================================
print("\n" + "=" * 78)
print("[4/4] Generating CV stability figure...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, target_name in enumerate(['Net Load', 'Price']):
    ax = axes[idx]
    sub = cv_df[cv_df['Target'] == target_name]

    ax2 = ax.twinx()
    x = sub['Fold'].values

    bars = ax.bar(x - 0.15, sub['MAE'], 0.3, color='#E57373', alpha=0.8, label='MAE', edgecolor='black', linewidth=0.5)
    line = ax2.plot(x, sub['R2'], 's-', color='#1565C0', linewidth=2, markersize=8, label='R²')

    ax.set_xlabel('CV Fold')
    ax.set_ylabel('MAE', color='#E57373')
    ax2.set_ylabel('R²', color='#1565C0')
    ax.set_title(f'{target_name}: Expanding Window CV (5 Folds)', fontweight='bold')
    ax.set_xticks(x)

    # Add mean lines
    ax.axhline(sub['MAE'].mean(), color='#E57373', linestyle='--', alpha=0.5, label=f'Mean MAE: {sub["MAE"].mean():.0f}')
    ax2.axhline(sub['R2'].mean(), color='#1565C0', linestyle='--', alpha=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_cv_stability.png"))
plt.close()
print("  -> fig_cv_stability.png")

# =====================================================================
#  SUMMARY
# =====================================================================
print("\n" + "=" * 78)
print("  STATISTICAL ANALYSIS COMPLETE")
print("=" * 78)

print(f"\n  Diebold-Mariano tests: {len(dm_results)} pairwise comparisons")
sig_count = sum(1 for r in dm_results if r['Significance'] != 'ns')
print(f"  Significant differences: {sig_count}/{len(dm_results)}")

print(f"\n  Error diagnostics: {len(diag_results)} model-target combinations")
print(f"  CV folds: {len(cv_df)} results across {cv_df['Fold'].nunique()} folds")

print(f"\n  Output tables:")
for f in ['diebold_mariano_tests.csv', 'error_diagnostics.csv', 'timeseries_cv_results.csv']:
    print(f"    {f}")
print(f"\n  Output figure: fig_cv_stability.png")
print("=" * 78)
