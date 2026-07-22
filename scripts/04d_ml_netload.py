"""
=============================================================================
 ML Day-Ahead Forecast for REAL Net Load (demand - solar - wind)
 Uses preprocessed data from 03b with generation by fuel type.
 Benchmark: LightGBM vs XGBoost vs RandomForest + Persistence baseline
=============================================================================
"""
import os, sys, io, json, time, warnings
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
import lightgbm as lgb
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
    'legend.fontsize': 9, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print("=" * 78)
print("  ML DAY-AHEAD FORECAST: REAL NET LOAD (demand - solar - wind)")
print("=" * 78)

# =====================================================================
#  LOAD PREPROCESSED DATA
# =====================================================================
print("\n[1/6] Loading preprocessed data...")

train = pd.read_csv(os.path.join(DATA_DIR, "train_2023.csv"), index_col=0, parse_dates=True)
val = pd.read_csv(os.path.join(DATA_DIR, "val_2023.csv"), index_col=0, parse_dates=True)
test = pd.read_csv(os.path.join(DATA_DIR, "test_2023.csv"), index_col=0, parse_dates=True)

with open(os.path.join(DATA_DIR, "feature_config.json")) as f:
    config = json.load(f)

feature_cols = [c for c in config['feature_cols'] if c in train.columns]
target_cols = config['target_cols']

X_train, X_val, X_test = train[feature_cols].values, val[feature_cols].values, test[feature_cols].values

print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"  Features: {len(feature_cols)}")
print(f"  Targets: {target_cols}")
print(f"  Net load def: {config.get('net_load_definition', 'unknown')}")

def calc_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100,
    }

# =====================================================================
#  BENCHMARK
# =====================================================================
print("\n" + "=" * 78)
print("[2/6] Training benchmarks...")

MODEL_NAMES = ['LightGBM', 'XGBoost', 'RandomForest']
COLORS = {'LightGBM': '#4CAF50', 'XGBoost': '#2196F3', 'RandomForest': '#FF9800', 'Persistence': '#9E9E9E'}
benchmark = {}

for target in target_cols:
    tname = 'Net Load' if 'net_load' in target else 'Price'
    print(f"\n{'='*60}\n  {tname} ({target})\n{'='*60}")

    y_tr, y_va, y_te = train[target].values, val[target].values, test[target].values
    benchmark[target] = {}

    # LightGBM
    t0 = time.time()
    dt = lgb.Dataset(X_train, y_tr, feature_name=feature_cols)
    dv = lgb.Dataset(X_val, y_va, feature_name=feature_cols, reference=dt)
    lgb_m = lgb.train({'objective':'regression','metric':'mae','num_leaves':63,'learning_rate':0.03,
        'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,'min_child_samples':30,
        'n_estimators':3000,'early_stopping_rounds':100,'verbose':-1,'random_state':42,'n_jobs':-1,
        'lambda_l1':0.1,'lambda_l2':0.1}, dt, valid_sets=[dv], valid_names=['val'],
        callbacks=[lgb.log_evaluation(0)])
    lt = time.time() - t0
    lp = {s: lgb_m.predict(X) for s, X in [('train',X_train),('val',X_val),('test',X_test)]}
    lm = {s: calc_metrics(y, lp[s]) for s, y in [('train',y_tr),('val',y_va),('test',y_te)]}
    li = pd.DataFrame({'feature':feature_cols,'importance':lgb_m.feature_importance(importance_type='gain')}).sort_values('importance',ascending=False)
    benchmark[target]['LightGBM'] = {'pred':lp,'metrics':lm,'time':lt,'importance':li,'model':lgb_m}
    print(f"  LightGBM  — Test MAE={lm['test']['MAE']:.1f}, R2={lm['test']['R2']:.4f} ({lt:.1f}s)")

    # XGBoost
    t0 = time.time()
    xm = xgb.XGBRegressor(n_estimators=3000,max_depth=6,learning_rate=0.03,subsample=0.7,
        colsample_bytree=0.7,min_child_weight=30,early_stopping_rounds=100,eval_metric='mae',
        reg_alpha=0.1,reg_lambda=0.1,random_state=42,n_jobs=-1,verbosity=0)
    xm.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)
    xt = time.time() - t0
    xp = {s: xm.predict(X) for s, X in [('train',X_train),('val',X_val),('test',X_test)]}
    xmet = {s: calc_metrics(y, xp[s]) for s, y in [('train',y_tr),('val',y_va),('test',y_te)]}
    xi = pd.DataFrame({'feature':feature_cols,'importance':xm.feature_importances_}).sort_values('importance',ascending=False)
    benchmark[target]['XGBoost'] = {'pred':xp,'metrics':xmet,'time':xt,'importance':xi}
    print(f"  XGBoost   — Test MAE={xmet['test']['MAE']:.1f}, R2={xmet['test']['R2']:.4f} ({xt:.1f}s)")

    # RandomForest
    t0 = time.time()
    rm = RandomForestRegressor(n_estimators=500,max_depth=15,min_samples_leaf=20,max_features=0.7,random_state=42,n_jobs=-1)
    rm.fit(X_train, y_tr)
    rt = time.time() - t0
    rp = {s: rm.predict(X) for s, X in [('train',X_train),('val',X_val),('test',X_test)]}
    rmet = {s: calc_metrics(y, rp[s]) for s, y in [('train',y_tr),('val',y_va),('test',y_te)]}
    ri = pd.DataFrame({'feature':feature_cols,'importance':rm.feature_importances_}).sort_values('importance',ascending=False)
    benchmark[target]['RandomForest'] = {'pred':rp,'metrics':rmet,'time':rt,'importance':ri}
    print(f"  RF        — Test MAE={rmet['test']['MAE']:.1f}, R2={rmet['test']['R2']:.4f} ({rt:.1f}s)")

    # Persistence
    persist_col = 'netload_lag24h' if 'net_load' in target else 'price_lag24h'
    if persist_col in test.columns:
        y_persist = test[persist_col].values
        pm = calc_metrics(y_te, y_persist)
        benchmark[target]['Persistence'] = {'pred':{'test':y_persist},'metrics':{'test':pm}}
        print(f"  Persist   — Test MAE={pm['MAE']:.1f}, R2={pm['R2']:.4f}")

best_models = {}
for target in target_cols:
    ml_only = {k:v for k,v in benchmark[target].items() if k != 'Persistence'}
    best = min(ml_only, key=lambda k: ml_only[k]['metrics']['val']['MAE'])
    best_models[target] = best
    print(f"\n  Best for {target}: {best}")

# =====================================================================
#  QUANTILE REGRESSION + CQR
# =====================================================================
print("\n" + "=" * 78)
print("[3/6] Quantile Regression + CQR...")

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
conformal = {}

for target in target_cols:
    tname = 'Net Load' if 'net_load' in target else 'Price'
    y_va, y_te = val[target].values, test[target].values
    dt = lgb.Dataset(X_train, train[target].values, feature_name=feature_cols)
    dv = lgb.Dataset(X_val, y_va, feature_name=feature_cols, reference=dt)

    q_preds = {'val': {}, 'test': {}}
    for q in quantiles:
        mq = lgb.train({'objective':'quantile','alpha':q,'metric':'quantile','num_leaves':63,
            'learning_rate':0.03,'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,
            'min_child_samples':30,'n_estimators':2000,'early_stopping_rounds':100,
            'verbose':-1,'random_state':42,'n_jobs':-1}, dt, valid_sets=[dv], valid_names=['val'],
            callbacks=[lgb.log_evaluation(0)])
        q_preds['val'][q] = mq.predict(X_val)
        q_preds['test'][q] = mq.predict(X_test)

    conf_scores = np.maximum(q_preds['val'][0.10] - y_va, y_va - q_preds['val'][0.90])
    n = len(conf_scores)
    Q_hat = np.quantile(conf_scores, min(np.ceil((n+1)*0.90)/n, 1.0))

    q_lo = q_preds['test'][0.10] - Q_hat
    q_hi = q_preds['test'][0.90] + Q_hat
    q25 = q_preds['test'][0.25] - Q_hat * 0.5
    q75 = q_preds['test'][0.75] + Q_hat * 0.5
    cov = np.mean((y_te >= q_lo) & (y_te <= q_hi))
    width = np.mean(q_hi - q_lo)

    conformal[target] = {'Q_hat':Q_hat,'q_lo':q_lo,'q_hi':q_hi,'q25':q25,'q75':q75,'coverage_90':cov,'width_90':width}
    print(f"  {tname}: Q_hat={Q_hat:.1f}, 90% coverage={cov*100:.1f}%, width={width:.1f}")

# =====================================================================
#  FIGURES
# =====================================================================
print("\n" + "=" * 78)
print("[4/6] Generating figures...")

# Fig 08: Benchmark
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, target in enumerate(target_cols):
    tname = 'Net Load (MW)' if 'net_load' in target else 'Price ($/MWh)'
    ml_names = MODEL_NAMES

    ax = axes[0, idx]
    x = np.arange(len(ml_names))
    w = 0.25
    for i, split in enumerate(['train','val','test']):
        vals = [benchmark[target][m]['metrics'][split]['MAE'] for m in ml_names]
        ax.bar(x + i*w - w, vals, w, label=split.title(), alpha=0.8, edgecolor='black', linewidth=0.5)
    if 'Persistence' in benchmark[target]:
        pm = benchmark[target]['Persistence']['metrics']['test']['MAE']
        ax.axhline(pm, color='grey', linestyle='--', linewidth=2, label=f'Persistence ({pm:.0f})')
    ax.set_xticks(x); ax.set_xticklabels(ml_names)
    ax.set_ylabel('MAE'); ax.set_title(f'{tname} — MAE', fontweight='bold'); ax.legend(fontsize=8)

    ax = axes[1, idx]
    all_names = ml_names + (['Persistence'] if 'Persistence' in benchmark[target] else [])
    r2s = [benchmark[target][m]['metrics']['test']['R2'] for m in all_names]
    cols = [COLORS[m] for m in all_names]
    bars = ax.bar(range(len(all_names)), r2s, color=cols, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(all_names))); ax.set_xticklabels(all_names)
    ax.set_ylabel('R²'); ax.set_title(f'{tname} — R²', fontweight='bold')
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

fig.suptitle('Day-Ahead Forecast: Real Net Load (demand - solar - wind)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig08_benchmark_dayahead.png"))
plt.close()
print("  -> fig08_benchmark_dayahead.png")

# Fig 09: Predictions with PI
for idx, target in enumerate(target_cols):
    tname = 'Net Load' if 'net_load' in target else 'Price'
    best = best_models[target]
    y_te = test[target].values
    y_pred = benchmark[target][best]['pred']['test']
    cr = conformal[target]; m = benchmark[target][best]['metrics']['test']
    unit = 'MW' if 'net_load' in target else '$/MWh'

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    ti = test.index

    ax = axes[0]
    ax.fill_between(ti, cr['q_lo'], cr['q_hi'], alpha=0.15, color='#1565C0', label='90% PI')
    ax.fill_between(ti, cr['q25'], cr['q75'], alpha=0.3, color='#42A5F5', label='50% PI')
    ax.plot(ti, y_te, 'k-', linewidth=0.8, alpha=0.8, label='Actual')
    ax.plot(ti, y_pred, 'r--', linewidth=0.8, alpha=0.7, label=best)
    ax.set_ylabel(f'{tname} ({unit})')
    ax.set_title(f'{tname} Day-Ahead ({best}): MAE={m["MAE"]:.0f} {unit} | R²={m["R2"]:.4f} | 90% Cov={cr["coverage_90"]*100:.1f}%',
                 fontweight='bold')
    ax.legend(ncol=5, fontsize=8)

    ax = axes[1]
    n_w = min(168, len(ti)-1); sl = slice(0, n_w+1)
    ax.fill_between(ti[sl], cr['q_lo'][sl], cr['q_hi'][sl], alpha=0.15, color='#1565C0')
    ax.fill_between(ti[sl], cr['q25'][sl], cr['q75'][sl], alpha=0.3, color='#42A5F5')
    ax.plot(ti[sl], y_te[sl], 'k-', linewidth=1.5, marker='o', markersize=2, label='Actual')
    ax.plot(ti[sl], y_pred[sl], 'r--', linewidth=1.2, label=best)
    ax.set_ylabel(f'{tname} ({unit})'); ax.set_xlabel('Date')
    ax.set_title('Zoomed: First Week', fontweight='bold'); ax.legend(ncol=4, fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig{9+idx:02d}_{tname.lower().replace(' ','_')}_dayahead.png"))
    plt.close()
    print(f"  -> fig{9+idx:02d}_{tname.lower().replace(' ','_')}_dayahead.png")

# Fig 11: Overfit + Fig 12: Feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, target in enumerate(target_cols):
    tname = 'Net Load' if 'net_load' in target else 'Price'
    ax = axes[idx]
    x = np.arange(len(MODEL_NAMES)); w = 0.2
    for i, split in enumerate(['train','val','test']):
        vals = [benchmark[target][m]['metrics'][split]['R2'] for m in MODEL_NAMES]
        ax.bar(x+i*w-w, vals, w, label=split.title(), alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel('R²'); ax.set_title(f'{tname}: Overfit Check', fontweight='bold'); ax.legend()
    for mi, mn in enumerate(MODEL_NAMES):
        gap = benchmark[target][mn]['metrics']['train']['R2'] - benchmark[target][mn]['metrics']['test']['R2']
        ax.annotate(f'Gap:{gap:.3f}', xy=(mi, benchmark[target][mn]['metrics']['test']['R2']),
                    xytext=(mi, benchmark[target][mn]['metrics']['test']['R2']-0.06),
                    ha='center', fontsize=8, color='red', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig11_overfit_diagnostic.png")); plt.close()
print("  -> fig11_overfit_diagnostic.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for idx, target in enumerate(target_cols):
    tname = 'Net Load' if 'net_load' in target else 'Price'
    best = best_models[target]
    imp = benchmark[target][best]['importance'].head(15)
    ax = axes[idx]
    ax.barh(range(len(imp)), imp['importance'].values, color=plt.cm.viridis(np.linspace(0.3,0.9,len(imp))))
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp['feature'].values, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel('Importance (Gain)')
    ax.set_title(f'{tname} — Top Features ({best})', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig12_feature_importance.png")); plt.close()
print("  -> fig12_feature_importance.png")

# =====================================================================
#  SAVE
# =====================================================================
print("\n" + "=" * 78)
print("[5/6] Saving results...")

# Benchmark table
rows = []
for target in target_cols:
    tname = 'Net Load' if 'net_load' in target else 'Price'
    for mn in MODEL_NAMES + ['Persistence']:
        if mn not in benchmark[target]: continue
        m = benchmark[target][mn]['metrics']
        row = {'Target': tname, 'Model': mn, 'Test_MAE': m['test']['MAE'], 'Test_R2': m['test']['R2']}
        if 'train' in m:
            row['Train_R2'] = m['train']['R2']
            row['Overfit_Gap'] = m['train']['R2'] - m['test']['R2']
        if mn in ['LightGBM','XGBoost','RandomForest']:
            row['Train_Time_s'] = benchmark[target][mn].get('time', 0)
        rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, "benchmark_dayahead.csv"), index=False)

# ML summary
summary = []
for target in target_cols:
    tname = 'Net Load' if 'net_load' in target else 'Price'
    best = best_models[target]
    m = benchmark[target][best]['metrics']['test']; cr = conformal[target]
    summary.append({'Target': tname, 'Best_Model': best, 'Horizon': '24h',
        'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'], 'MAPE': m['MAPE'],
        '90% Coverage': cr['coverage_90']*100, '90% PI Width': cr['width_90'], 'Q_hat': cr['Q_hat']})
pd.DataFrame(summary).to_csv(os.path.join(TABLE_DIR, "ml_results_summary.csv"), index=False)

# Predictions for MILP
best_nl = best_models['net_load_MW']
best_pr = best_models['price_USD_MWh']
preds = pd.DataFrame({
    'timestamp': test.index,
    'net_load_actual': test['net_load_MW'].values,
    'net_load_predicted': benchmark['net_load_MW'][best_nl]['pred']['test'],
    'net_load_Q10': conformal['net_load_MW']['q_lo'],
    'net_load_Q90': conformal['net_load_MW']['q_hi'],
    'price_actual': test['price_USD_MWh'].values,
    'price_predicted': benchmark['price_USD_MWh'][best_pr]['pred']['test'],
    'price_Q10': conformal['price_USD_MWh']['q_lo'],
    'price_Q90': conformal['price_USD_MWh']['q_hi'],
})
preds.to_csv(os.path.join(DATA_DIR, "ml_predictions_for_milp.csv"), index=False)
print(f"  -> ml_predictions_for_milp.csv ({len(preds)} rows)")

# =====================================================================
#  FINAL REPORT
# =====================================================================
print("\n" + "=" * 78)
print("  RESULTS — REAL NET LOAD (demand - solar - wind)")
print("=" * 78)

print(f"\n  {'Model':<15} {'Target':<12} {'Test MAE':>10} {'Test R2':>10} {'Gap':>8} {'Skill%':>8}")
print(f"  {'-'*65}")
for target in target_cols:
    tname = 'Net Load' if 'net_load' in target else 'Price'
    pm = benchmark[target].get('Persistence',{}).get('metrics',{}).get('test',{}).get('MAE', 1)
    for mn in MODEL_NAMES:
        m = benchmark[target][mn]['metrics']
        gap = m['train']['R2'] - m['test']['R2']
        skill = (1 - m['test']['MAE']/pm)*100 if pm > 0 else 0
        tag = ' <<' if mn == best_models[target] else ''
        print(f"  {mn:<15} {tname:<12} {m['test']['MAE']:>10.1f} {m['test']['R2']:>10.4f} {gap:>8.4f} {skill:>+7.1f}%{tag}")
    if 'Persistence' in benchmark[target]:
        print(f"  {'Persistence':<15} {tname:<12} {pm:>10.1f} {benchmark[target]['Persistence']['metrics']['test']['R2']:>10.4f}")
    print()
print("=" * 78)
