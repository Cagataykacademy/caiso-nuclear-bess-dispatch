"""
Final 2022 out-of-sample validation WITH temperature features.
"""
import os, sys, io, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(P, "data")
FIG = os.path.join(P, "outputs", "figures")
TBL = os.path.join(P, "outputs", "tables")

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'serif','font.size':11,'axes.grid':True,'grid.alpha':0.3,
    'axes.spines.top':False,'axes.spines.right':False})

def metrics(y, p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p)}

print("="*78)
print("  2022 VALIDATION WITH TEMPERATURE")
print("="*78)

# Load 2023 preprocessed + train model
import json
proc23 = pd.read_csv(os.path.join(DATA,"caiso_preprocessed_v2_2023.csv"),index_col=0,parse_dates=True)
with open(os.path.join(DATA,"feature_config.json")) as f:
    cfg = json.load(f)
feature_cols = [c for c in cfg['feature_cols'] if c in proc23.columns]

train = proc23[proc23.index <= '2023-08-31']
val = proc23[(proc23.index > '2023-08-31') & (proc23.index <= '2023-10-31')]
test = proc23[proc23.index > '2023-10-31']

X_tr = train[feature_cols].values
X_va = val[feature_cols].values
X_te = test[feature_cols].values
y_tr = train['net_load_MW'].values
y_va = val['net_load_MW'].values
y_te = test['net_load_MW'].values

# Train LightGBM
dt = lgb.Dataset(X_tr, y_tr, feature_name=feature_cols)
dv = lgb.Dataset(X_va, y_va, feature_name=feature_cols, reference=dt)
lgb_m = lgb.train({'objective':'regression','metric':'mae','num_leaves':63,'learning_rate':0.03,
    'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,'min_child_samples':30,
    'n_estimators':3000,'early_stopping_rounds':100,'verbose':-1,'random_state':42,'n_jobs':-1,
    'lambda_l1':0.1,'lambda_l2':0.1}, dt, valid_sets=[dv], valid_names=['val'],
    callbacks=[lgb.log_evaluation(0)])

# Build 2022 dataset with temperature
gen22 = pd.read_csv(os.path.join(DATA,"caiso_generation_by_fuel_2022.csv"))
gen22['period'] = pd.to_datetime(gen22['period'])
gen22['value'] = pd.to_numeric(gen22['value'], errors='coerce')
gen22_piv = gen22.pivot_table(index='period', columns='fueltype', values='value', aggfunc='sum').sort_index()
gen22_piv.columns = [f'gen_{c.lower()}_MW' for c in gen22_piv.columns]

reg22 = pd.read_csv(os.path.join(DATA,"caiso_region_data_2022.csv"))
reg22['period'] = pd.to_datetime(reg22['period'])
reg22['value'] = pd.to_numeric(reg22['value'], errors='coerce')
reg22_piv = reg22.pivot_table(index='period', columns='type-name', values='value', aggfunc='sum').sort_index()
reg22_piv.columns = reg22_piv.columns.str.lower().str.replace(' ','_').str.replace('-','_')

df22 = reg22_piv.join(gen22_piv, how='inner').sort_index()
df22 = df22.rename(columns={'demand':'total_demand_MW','day_ahead_demand_forecast':'day_ahead_forecast_MW',
                              'net_generation':'net_generation_MW','total_interchange':'total_interchange_MW'})

df22['net_load_MW'] = df22['total_demand_MW'] - df22.get('gen_sun_MW',0) - df22.get('gen_wnd_MW',0)

# Calendar
df22['hour'] = df22.index.hour
df22['day_of_week'] = df22.index.dayofweek
df22['month'] = df22.index.month
df22['day_of_year'] = df22.index.dayofyear
df22['is_weekend'] = (df22.index.dayofweek >= 5).astype(int)
df22['week_of_year'] = df22.index.isocalendar().week.astype(int)
df22['hour_sin'] = np.sin(2*np.pi*df22['hour']/24)
df22['hour_cos'] = np.cos(2*np.pi*df22['hour']/24)
df22['month_sin'] = np.sin(2*np.pi*df22['month']/12)
df22['month_cos'] = np.cos(2*np.pi*df22['month']/12)
df22['dow_sin'] = np.sin(2*np.pi*df22['day_of_week']/7)
df22['dow_cos'] = np.cos(2*np.pi*df22['day_of_week']/7)

# Lags
for lag in [24,48,72,168]:
    df22[f'netload_lag{lag}h'] = df22['net_load_MW'].shift(lag)
    df22[f'demand_lag{lag}h'] = df22['total_demand_MW'].shift(lag)

for col, src in [('solar_lag24h','gen_sun_MW'),('solar_lag168h','gen_sun_MW'),
                  ('wind_lag24h','gen_wnd_MW'),('wind_lag168h','gen_wnd_MW')]:
    lag = 24 if '24' in col else 168
    df22[col] = df22.get(src, pd.Series(0,index=df22.index)).shift(lag)

df22['yesterday_demand_mean'] = df22['total_demand_MW'].shift(24).rolling(24).mean()
df22['yesterday_demand_max'] = df22['total_demand_MW'].shift(24).rolling(24).max()
df22['yesterday_demand_min'] = df22['total_demand_MW'].shift(24).rolling(24).min()
df22['yesterday_demand_std'] = df22['total_demand_MW'].shift(24).rolling(24).std()
df22['yesterday_netload_mean'] = df22['net_load_MW'].shift(24).rolling(24).mean()
df22['yesterday_solar_mean'] = df22.get('gen_sun_MW', pd.Series(0,index=df22.index)).shift(24).rolling(24).mean()
df22['demand_trend_24h'] = df22.get('demand_lag24h',0) - df22.get('demand_lag48h',0)
df22['netload_trend_24h'] = df22.get('netload_lag24h',0) - df22.get('netload_lag48h',0)
df22['forecast_error_lag24h'] = df22.get('day_ahead_forecast_MW',pd.Series(0,index=df22.index)).shift(24) - df22['total_demand_MW'].shift(24)
df22['price_lag24h'] = 0

# Temperature — NOW WITH REAL DATA
weather22 = pd.read_csv(os.path.join(DATA,"caiso_weather_2022.csv"), index_col=0, parse_dates=True)
df22['temp_lag24h'] = df22.index.map(weather22['temp_avg_C'].to_dict())
df22['temp_lag24h'] = df22['temp_lag24h'].shift(24).interpolate().ffill().bfill()
df22['CDH_lag24h'] = (df22['temp_lag24h'] - 18.3).clip(lower=0)
df22['HDH_lag24h'] = (18.3 - df22['temp_lag24h']).clip(lower=0)
temp_max_series = df22.index.map(weather22.get('temp_avg_C', pd.Series(dtype=float)).to_dict())
df22['temp_yesterday_max'] = pd.Series(temp_max_series, index=df22.index).shift(24).rolling(24).max()
df22['temp_yesterday_mean'] = pd.Series(temp_max_series, index=df22.index).shift(24).rolling(24).mean()
df22['CDH_hour_interaction'] = df22['CDH_lag24h'] * np.abs(df22['hour'] - 15)

df22 = df22.dropna()

# Fill any missing features with 0
for f in feature_cols:
    if f not in df22.columns:
        df22[f] = 0

X_22 = df22[feature_cols].values
y_22 = df22['net_load_MW'].values

# Predict
pred_22 = lgb_m.predict(X_22)
m_22 = metrics(y_22, pred_22)

# Persistence
persist_22 = df22['netload_lag24h'].values
m_p22 = metrics(y_22, persist_22)

skill = (1 - m_22['MAE']/m_p22['MAE'])*100

print(f"\n  2022 Results (WITH temperature):")
print(f"    LightGBM:    MAE={m_22['MAE']:.0f}  R2={m_22['R2']:.4f}")
print(f"    Persistence: MAE={m_p22['MAE']:.0f}  R2={m_p22['R2']:.4f}")
print(f"    Skill:       {skill:+.1f}% vs persistence")
print(f"    N:           {len(y_22)}")

# Also get 2023 test metrics for comparison
pred_23 = lgb_m.predict(X_te)
m_23 = metrics(y_te, pred_23)
m_p23 = metrics(y_te, test['netload_lag24h'].values)

# Save updated OOS table
oos = pd.DataFrame([
    {'Year':2023,'Dataset':'Test (in-sample year)','Model':'LightGBM','MAE':m_23['MAE'],'R2':m_23['R2'],'N':len(y_te)},
    {'Year':2023,'Dataset':'Test (in-sample year)','Model':'Persistence','MAE':m_p23['MAE'],'R2':m_p23['R2'],'N':len(y_te)},
    {'Year':2022,'Dataset':'Out-of-sample (with temp)','Model':'LightGBM','MAE':m_22['MAE'],'R2':m_22['R2'],'N':len(y_22)},
    {'Year':2022,'Dataset':'Out-of-sample (with temp)','Model':'Persistence','MAE':m_p22['MAE'],'R2':m_p22['R2'],'N':len(y_22)},
])
oos.to_csv(os.path.join(TBL,"out_of_sample_2022.csv"), index=False)
print("  -> out_of_sample_2022.csv")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
n = min(168*4, len(y_22))
ax = axes[0]
ax.plot(range(n), y_22[:n], 'k-', linewidth=0.5, alpha=0.7, label='Actual 2022')
ax.plot(range(n), pred_22[:n], 'r--', linewidth=0.5, alpha=0.7, label='Predicted (2023 model)')
ax.set_xlabel('Hour'); ax.set_ylabel('Net Load (MW)')
ax.set_title(f'2022 Out-of-Sample (with temp): R²={m_22["R2"]:.3f}', fontweight='bold')
ax.legend()

ax = axes[1]
ax.scatter(y_22, pred_22, s=3, alpha=0.2, c='#2196F3')
lims = [min(y_22.min(),pred_22.min()), max(y_22.max(),pred_22.max())]
ax.plot(lims, lims, 'r--', linewidth=2)
ax.set_xlabel('Actual (MW)'); ax.set_ylabel('Predicted (MW)')
ax.set_title(f'Actual vs Predicted (R²={m_22["R2"]:.3f}, Skill={skill:+.1f}%)', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG,"fig_2022_validation.png"))
plt.close()
print("  -> fig_2022_validation.png (updated)")

print("\n" + "="*78)
print(f"  DONE: 2022 OOS R²={m_22['R2']:.4f}, Skill={skill:+.1f}%")
print("="*78)
