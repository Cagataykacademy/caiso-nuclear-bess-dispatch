"""
=============================================================================
 FIX: fig09_feature_importance.png was rendering feature_importance_demand_dayahead.csv
 / feature_importance_price_dayahead.csv — outputs from an EARLIER, SUPERSEDED
 pipeline iteration (script 04c) whose feature set (demand_lag24h, no VRE lags)
 does not match the canonical 35-feature net load model actually reported in
 Table 3 (script 11_fix_all_weaknesses.py: feature_cols loaded from
 feature_config.json, XGBoost R^2=0.854).

 This script retrains XGBoost with the EXACT hyperparameters and feature set
 used in script 11 (verified against script 11 source) purely to extract a
 correct, matching feature-importance ranking, and regenerates fig09 as a
 single clean panel for the actual net load model (dropping the confusing
 "excluded preliminary price experiments" panel).
=============================================================================
"""
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(P, "data")
FIG  = os.path.join(P, "outputs", "figures")
TBL  = os.path.join(P, "outputs", "tables")

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'serif','font.size':11,'axes.grid':True,'grid.alpha':0.3,
    'axes.spines.top':False,'axes.spines.right':False})

proc = pd.read_csv(os.path.join(DATA, "caiso_preprocessed_v2_2023.csv"),
                   index_col=0, parse_dates=True)
with open(os.path.join(DATA, "feature_config.json")) as f:
    cfg = json.load(f)
feature_cols = [c for c in cfg['feature_cols'] if c in proc.columns]
assert len(feature_cols) == 35, f"expected 35 features, got {len(feature_cols)}"

train_end, val_end = '2023-08-31 23:00:00', '2023-10-31 23:00:00'
train = proc[proc.index <= train_end]
val   = proc[(proc.index > train_end) & (proc.index <= val_end)]

X_tr, y_tr = train[feature_cols].values, train['net_load_MW'].values
X_va, y_va = val[feature_cols].values, val['net_load_MW'].values

print(f"Retraining XGBoost on the canonical {len(feature_cols)}-feature set "
      f"(matching script 11 / Table 3)...")
xgb_m = xgb.XGBRegressor(n_estimators=3000, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=30,
    early_stopping_rounds=100, eval_metric='mae', reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbosity=0)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

imp = pd.DataFrame({'feature': feature_cols,
                    'importance': xgb_m.feature_importances_}
                   ).sort_values('importance', ascending=False).reset_index(drop=True)
imp['importance_pct'] = imp['importance'] / imp['importance'].sum() * 100
imp.to_csv(os.path.join(TBL, "feature_importance_netload_final.csv"), index=False)
print(imp.head(15).to_string(index=False))

# ---- figure: single clean panel, top 15 ------------------------------
top15 = imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top15['feature'], top15['importance_pct'], color='#1565C0', alpha=0.85,
               edgecolor='black', lw=0.4)
for b, v in zip(bars, top15['importance_pct']):
    ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2, f'{v:.1f}%',
            va='center', fontsize=9)
ax.set_xlabel('Relative Importance (%, XGBoost gain)')
ax.set_title('Top-15 Feature Importance: Day-Ahead Net Load Model\n'
             '(XGBoost, 35-feature canonical set)', fontweight='bold')
ax.set_xlim(0, top15['importance_pct'].max() * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig09_feature_importance.png"))
plt.close()
print("\n-> fig09_feature_importance.png (regenerated, single-panel, canonical model)")
print("-> feature_importance_netload_final.csv")
