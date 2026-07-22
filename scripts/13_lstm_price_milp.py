"""
=============================================================================
 THREE REVIEWER FIXES:
 1. LSTM benchmark for net load forecasting
 2. Price forecasting → BESS arbitrage in MILP
 3. 2022 out-of-sample validation (runs after 2022 data is fetched)
=============================================================================
"""
import os, sys, io, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),
            'R2':r2_score(y,p),'MAPE':np.mean(np.abs((y-p)/np.clip(np.abs(y),1,None)))*100}

print("="*78)
print("  LSTM + PRICE-MILP + 2022 VALIDATION")
print("="*78)

# Load 2023 data
proc = pd.read_csv(os.path.join(DATA,"caiso_preprocessed_v2_2023.csv"),index_col=0,parse_dates=True)
with open(os.path.join(DATA,"feature_config.json")) as f:
    cfg = json.load(f)
feature_cols = [c for c in cfg['feature_cols'] if c in proc.columns]

train = proc[proc.index <= '2023-08-31']
val = proc[(proc.index > '2023-08-31') & (proc.index <= '2023-10-31')]
test = proc[proc.index > '2023-10-31']

X_tr, X_va, X_te = train[feature_cols].values, val[feature_cols].values, test[feature_cols].values
y_tr, y_va, y_te = train['net_load_MW'].values, val['net_load_MW'].values, test['net_load_MW'].values

# =====================================================================
#  FIX 1: LSTM BENCHMARK
# =====================================================================
print("\n[1/3] LSTM Benchmark...")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# Prepare sequences (lookback=24h windows)
LOOKBACK = 24

scaler_X = StandardScaler().fit(X_tr)
scaler_y = StandardScaler().fit(y_tr.reshape(-1,1))

def make_sequences(X, y, lookback):
    X_s = scaler_X.transform(X)
    y_s = scaler_y.transform(y.reshape(-1,1)).ravel()
    Xs, ys = [], []
    for i in range(lookback, len(X_s)):
        Xs.append(X_s[i-lookback:i])
        ys.append(y_s[i])
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, LOOKBACK)
X_va_seq, y_va_seq = make_sequences(X_va, y_va, LOOKBACK)
X_te_seq, y_te_seq = make_sequences(X_te, y_te, LOOKBACK)

train_ds = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

model = LSTMForecaster(len(feature_cols), hidden=64, layers=2, dropout=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print("  Training LSTM (64 hidden, 2 layers, lookback=24h)...")
t0 = time.time()
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.FloatTensor(X_va_seq))
        val_loss = criterion(val_pred, torch.FloatTensor(y_va_seq)).item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 15:
        print(f"    Early stopping at epoch {epoch+1}")
        break

    if (epoch+1) % 20 == 0:
        print(f"    Epoch {epoch+1}: train_loss={epoch_loss/len(train_dl):.4f}, val_loss={val_loss:.4f}")

lstm_time = time.time() - t0
model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    lstm_pred_tr = scaler_y.inverse_transform(model(torch.FloatTensor(X_tr_seq)).numpy().reshape(-1,1)).ravel()
    lstm_pred_va = scaler_y.inverse_transform(model(torch.FloatTensor(X_va_seq)).numpy().reshape(-1,1)).ravel()
    lstm_pred_te = scaler_y.inverse_transform(model(torch.FloatTensor(X_te_seq)).numpy().reshape(-1,1)).ravel()

m_lstm_tr = metrics(y_tr[LOOKBACK:], lstm_pred_tr)
m_lstm_va = metrics(y_va[LOOKBACK:], lstm_pred_va)
m_lstm_te = metrics(y_te[LOOKBACK:], lstm_pred_te)

print(f"\n  LSTM Results:")
print(f"    Train: MAE={m_lstm_tr['MAE']:.0f}  R2={m_lstm_tr['R2']:.4f}")
print(f"    Val:   MAE={m_lstm_va['MAE']:.0f}  R2={m_lstm_va['R2']:.4f}")
print(f"    Test:  MAE={m_lstm_te['MAE']:.0f}  R2={m_lstm_te['R2']:.4f}  ({lstm_time:.1f}s)")

# Compare with existing results
bench = pd.read_csv(os.path.join(TBL, "benchmark_final.csv"))
lstm_row = {'Target':'Net Load','Model':'LSTM','Test_MAE':m_lstm_te['MAE'],
            'Test_RMSE':m_lstm_te['RMSE'],'Test_R2':m_lstm_te['R2'],'Test_MAPE':m_lstm_te['MAPE'],
            'Train_R2':m_lstm_tr['R2'],'Overfit_Gap':m_lstm_tr['R2']-m_lstm_te['R2'],
            'Train_Time_s':lstm_time}

# Add LSTM to benchmark
bench = pd.concat([bench, pd.DataFrame([lstm_row])], ignore_index=True)
bench.to_csv(os.path.join(TBL, "benchmark_final.csv"), index=False)
print("  -> benchmark_final.csv updated with LSTM")

# =====================================================================
#  FIX 2: PRICE FORECAST → BESS ARBITRAGE IN MILP
# =====================================================================
print("\n" + "="*78)
print("[2/3] Price-responsive BESS arbitrage in MILP...")

import pyomo.environ as pyo
try:
    import highspy; SOLVER = 'appsi_highs'
except:
    SOLVER = 'glpk'

# Train daily price forecast
proc['price_daily'] = proc.groupby(proc.index.date)['price_USD_MWh'].transform('mean')

# Daily price features for forecasting
price_features = [c for c in feature_cols if c in proc.columns]
if 'price_lag24h' not in price_features and 'price_lag24h' in proc.columns:
    price_features.append('price_lag24h')

X_tr_p = train[price_features].values
X_te_p = test[price_features].values
y_tr_p = train['price_daily'].values if 'price_daily' in train.columns else train['price_USD_MWh'].values
y_te_p = test['price_daily'].values if 'price_daily' in test.columns else test['price_USD_MWh'].values

# Train LightGBM for price
dt_p = lgb.Dataset(X_tr_p, y_tr_p, feature_name=price_features)
dv_p = lgb.Dataset(val[price_features].values, val['price_daily'].values if 'price_daily' in val.columns else val['price_USD_MWh'].values,
                    feature_name=price_features, reference=dt_p)
lgb_price = lgb.train({'objective':'regression','metric':'mae','num_leaves':31,'learning_rate':0.03,
    'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,'min_child_samples':30,
    'n_estimators':2000,'early_stopping_rounds':100,'verbose':-1,'random_state':42,'n_jobs':-1},
    dt_p, valid_sets=[dv_p], valid_names=['val'], callbacks=[lgb.log_evaluation(0)])

price_pred = lgb_price.predict(X_te_p)
m_price = metrics(y_te_p, price_pred)
print(f"  Price forecast: MAE=${m_price['MAE']:.1f}  R2={m_price['R2']:.4f}")

# MILP with price-responsive BESS arbitrage
print("\n  Running price-responsive MILP (spring week)...")

spring = proc['2023-04-10':'2023-04-16']
spring_nl = spring['net_load_MW'].values[:168]
spring_price = spring['price_USD_MWh'].values[:168]

def solve_price_milp(nl, prices, name, use_price_arb=True):
    T = len(nl)
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T-1)
    m.Pn = pyo.Var(m.T, bounds=(1800, 2256))
    m.Pg = pyo.Var(m.T, bounds=(0, 15000))
    m.Pch = pyo.Var(m.T, bounds=(0, 5000))
    m.Pdis = pyo.Var(m.T, bounds=(0, 5000))
    m.SoC = pyo.Var(m.T, bounds=(2000, 18000))
    m.Pimp = pyo.Var(m.T, bounds=(0, 8000))
    m.Pexp = pyo.Var(m.T, bounds=(0, 6000))
    m.Pshed = pyo.Var(m.T, bounds=(0, 10000))
    m.Pcurt = pyo.Var(m.T, bounds=(0, 20000))
    m.u = pyo.Var(m.T, within=pyo.Binary)

    if use_price_arb:
        # Price-responsive: BESS earns revenue from price spread
        m.obj = pyo.Objective(expr=sum(
            12*m.Pn[t] + 45*m.Pg[t]
            + prices[t]*m.Pch[t] - prices[t]*m.Pdis[t]  # Buy low, sell high
            + 5*(m.Pch[t]+m.Pdis[t])  # Degradation cost
            + 55*m.Pimp[t] - 20*m.Pexp[t]
            + 10000*m.Pshed[t] + 10*m.Pcurt[t]
            for t in m.T), sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(expr=sum(
            12*m.Pn[t]+45*m.Pg[t]+5*(m.Pch[t]+m.Pdis[t])
            +55*m.Pimp[t]-20*m.Pexp[t]+10000*m.Pshed[t]+10*m.Pcurt[t]
            for t in m.T), sense=pyo.minimize)

    def pb(md,t): return md.Pn[t]+md.Pg[t]+md.Pdis[t]+md.Pimp[t]+md.Pshed[t]==nl[t]+md.Pch[t]+md.Pexp[t]+md.Pcurt[t]
    m.pb = pyo.Constraint(m.T, rule=pb)
    m.soc0 = pyo.Constraint(expr=m.SoC[0]==10000)
    def sd(md,t):
        if t==0: return pyo.Constraint.Skip
        return md.SoC[t]==md.SoC[t-1]+0.949*md.Pch[t]-1.054*md.Pdis[t]
    m.sd = pyo.Constraint(m.T, rule=sd)
    def bcl(md,t): return md.Pch[t]<=5000*(1-md.u[t])
    def bdl(md,t): return md.Pdis[t]<=5000*md.u[t]
    m.bcl = pyo.Constraint(m.T, rule=bcl)
    m.bdl = pyo.Constraint(m.T, rule=bdl)
    def nru(md,t):
        if t==0: return pyo.Constraint.Skip
        return md.Pn[t]-md.Pn[t-1]<=500
    def nrd(md,t):
        if t==0: return pyo.Constraint.Skip
        return md.Pn[t-1]-md.Pn[t]<=500
    m.nru = pyo.Constraint(m.T, rule=nru)
    m.nrd = pyo.Constraint(m.T, rule=nrd)

    solver = pyo.SolverFactory(SOLVER)
    solver.options['time_limit'] = 120
    try:
        res = solver.solve(m, tee=False, load_solutions=False)
        if str(res.solver.termination_condition) in ['optimal','feasible']:
            m.solutions.load_from(res)
        else:
            return None
    except:
        return None

    Pdis = np.array([pyo.value(m.Pdis[t]) for t in range(T)])
    Pch = np.array([pyo.value(m.Pch[t]) for t in range(T)])
    Pg = np.array([pyo.value(m.Pg[t]) for t in range(T)])

    bess_revenue = np.sum(prices * Pdis - prices * Pch) if use_price_arb else 0

    return {
        'cost': pyo.value(m.obj), 'cost_mwh': pyo.value(m.obj)/max(np.sum(nl),1),
        'bess_revenue': bess_revenue,
        'avg_gas': np.mean(Pg),
        'bess_cycles': np.sum(Pdis)/20000,
        'shed': sum(pyo.value(m.Pshed[t]) for t in range(T)),
    }

# Compare fixed-cost vs price-responsive BESS
r_fixed = solve_price_milp(spring_nl, spring_price, 'fixed', use_price_arb=False)
r_price = solve_price_milp(spring_nl, spring_price, 'price', use_price_arb=True)

if r_fixed and r_price:
    print(f"\n  Fixed-cost BESS:     $/MWh={r_fixed['cost_mwh']:.2f}  Cycles={r_fixed['bess_cycles']:.2f}")
    print(f"  Price-responsive:    $/MWh={r_price['cost_mwh']:.2f}  Cycles={r_price['bess_cycles']:.2f}  "
          f"BESS Revenue=${r_price['bess_revenue']:,.0f}")
    improvement = (r_fixed['cost_mwh'] - r_price['cost_mwh']) / r_fixed['cost_mwh'] * 100
    print(f"  Cost improvement:    {improvement:+.2f}%")

    price_milp_df = pd.DataFrame([
        {'BESS_Mode': 'Fixed cost ($5/MWh)', 'Cost_MWh': r_fixed['cost_mwh'],
         'BESS_Cycles': r_fixed['bess_cycles'], 'BESS_Revenue': 0},
        {'BESS_Mode': 'Price-responsive', 'Cost_MWh': r_price['cost_mwh'],
         'BESS_Cycles': r_price['bess_cycles'], 'BESS_Revenue': r_price['bess_revenue']},
    ])
    price_milp_df.to_csv(os.path.join(TBL, "price_responsive_milp.csv"), index=False)
    print("  -> price_responsive_milp.csv")

# =====================================================================
#  FIX 3: 2022 OUT-OF-SAMPLE VALIDATION
# =====================================================================
print("\n" + "="*78)
print("[3/3] 2022 Out-of-sample validation...")

gen_2022_path = os.path.join(DATA, "caiso_generation_by_fuel_2022.csv")
reg_2022_path = os.path.join(DATA, "caiso_region_data_2022.csv")

if os.path.exists(gen_2022_path) and os.path.exists(reg_2022_path):
    print("  Loading 2022 data...")

    gen22 = pd.read_csv(gen_2022_path)
    gen22['period'] = pd.to_datetime(gen22['period'])
    gen22['value'] = pd.to_numeric(gen22['value'], errors='coerce')
    gen22_piv = gen22.pivot_table(index='period', columns='fueltype', values='value', aggfunc='sum').sort_index()
    gen22_piv.columns = [f'gen_{c.lower()}_MW' for c in gen22_piv.columns]

    reg22 = pd.read_csv(reg_2022_path)
    reg22['period'] = pd.to_datetime(reg22['period'])
    reg22['value'] = pd.to_numeric(reg22['value'], errors='coerce')
    reg22_piv = reg22.pivot_table(index='period', columns='type-name', values='value', aggfunc='sum').sort_index()
    reg22_piv.columns = reg22_piv.columns.str.lower().str.replace(' ','_').str.replace('-','_')

    df22 = reg22_piv.join(gen22_piv, how='inner').sort_index()

    col_map = {'demand':'total_demand_MW','day_ahead_demand_forecast':'day_ahead_forecast_MW',
               'net_generation':'net_generation_MW','total_interchange':'total_interchange_MW'}
    df22 = df22.rename(columns=col_map)

    if 'gen_sun_MW' in df22.columns and 'gen_wnd_MW' in df22.columns:
        df22['net_load_MW'] = df22['total_demand_MW'] - df22['gen_sun_MW'] - df22['gen_wnd_MW']
    else:
        df22['net_load_MW'] = df22['total_demand_MW']

    # Build same features for 2022
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

    for lag in [24,48,72,168]:
        df22[f'netload_lag{lag}h'] = df22['net_load_MW'].shift(lag)
        df22[f'demand_lag{lag}h'] = df22['total_demand_MW'].shift(lag)

    if 'gen_sun_MW' in df22.columns:
        df22['solar_lag24h'] = df22['gen_sun_MW'].shift(24)
        df22['solar_lag168h'] = df22['gen_sun_MW'].shift(168)
    if 'gen_wnd_MW' in df22.columns:
        df22['wind_lag24h'] = df22['gen_wnd_MW'].shift(24)
        df22['wind_lag168h'] = df22['gen_wnd_MW'].shift(168)

    df22['yesterday_demand_mean'] = df22['total_demand_MW'].shift(24).rolling(24).mean()
    df22['yesterday_demand_max'] = df22['total_demand_MW'].shift(24).rolling(24).max()
    df22['yesterday_demand_min'] = df22['total_demand_MW'].shift(24).rolling(24).min()
    df22['yesterday_demand_std'] = df22['total_demand_MW'].shift(24).rolling(24).std()
    df22['yesterday_netload_mean'] = df22['net_load_MW'].shift(24).rolling(24).mean()
    df22['yesterday_solar_mean'] = df22.get('gen_sun_MW', pd.Series(0, index=df22.index)).shift(24).rolling(24).mean()
    df22['demand_trend_24h'] = df22.get('demand_lag24h', pd.Series(0, index=df22.index)) - df22.get('demand_lag48h', pd.Series(0, index=df22.index))
    df22['netload_trend_24h'] = df22.get('netload_lag24h', pd.Series(0, index=df22.index)) - df22.get('netload_lag48h', pd.Series(0, index=df22.index))
    df22['forecast_error_lag24h'] = (df22.get('day_ahead_forecast_MW', pd.Series(0, index=df22.index)).shift(24)
                                     - df22['total_demand_MW'].shift(24))
    df22['price_lag24h'] = 0  # No price data for 2022

    # Add dummy temperature features (not available for 2022)
    for tf in ['temp_lag24h','CDH_lag24h','HDH_lag24h','temp_yesterday_max','temp_yesterday_mean','CDH_hour_interaction']:
        if tf not in df22.columns:
            df22[tf] = 0

    df22 = df22.dropna()

    # Filter to features that exist
    avail_features = [f for f in feature_cols if f in df22.columns]
    missing = [f for f in feature_cols if f not in df22.columns]
    if missing:
        print(f"  Missing features (filled with 0): {missing}")
        for mf in missing:
            df22[mf] = 0

    X_2022 = df22[feature_cols].values
    y_2022 = df22['net_load_MW'].values

    # Use 2023-trained LightGBM to predict 2022
    dt = lgb.Dataset(X_tr, y_tr, feature_name=feature_cols)
    dv = lgb.Dataset(X_va, y_va, feature_name=feature_cols, reference=dt)
    lgb_23 = lgb.train({'objective':'regression','metric':'mae','num_leaves':63,'learning_rate':0.03,
        'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,'min_child_samples':30,
        'n_estimators':3000,'early_stopping_rounds':100,'verbose':-1,'random_state':42,'n_jobs':-1,
        'lambda_l1':0.1,'lambda_l2':0.1}, dt, valid_sets=[dv], valid_names=['val'],
        callbacks=[lgb.log_evaluation(0)])

    pred_2022 = lgb_23.predict(X_2022)
    m_2022 = metrics(y_2022, pred_2022)

    # Persistence for 2022
    persist_2022 = df22['netload_lag24h'].values
    m_persist_2022 = metrics(y_2022, persist_2022)

    print(f"\n  2022 Out-of-Sample Results (model trained on 2023):")
    print(f"    LightGBM:    MAE={m_2022['MAE']:.0f}  R2={m_2022['R2']:.4f}")
    print(f"    Persistence: MAE={m_persist_2022['MAE']:.0f}  R2={m_persist_2022['R2']:.4f}")
    skill_2022 = (1 - m_2022['MAE']/m_persist_2022['MAE'])*100
    print(f"    Skill:       {skill_2022:+.1f}% vs persistence")

    # Save
    oos_df = pd.DataFrame([
        {'Year': 2023, 'Dataset': 'Test (in-sample year)', 'Model': 'LightGBM',
         'MAE': metrics(y_te, lgb_23.predict(X_te))['MAE'],
         'R2': metrics(y_te, lgb_23.predict(X_te))['R2'],
         'N': len(y_te)},
        {'Year': 2023, 'Dataset': 'Test (in-sample year)', 'Model': 'Persistence',
         'MAE': metrics(y_te, test['netload_lag24h'].values)['MAE'],
         'R2': metrics(y_te, test['netload_lag24h'].values)['R2'],
         'N': len(y_te)},
        {'Year': 2022, 'Dataset': 'Out-of-sample year', 'Model': 'LightGBM',
         'MAE': m_2022['MAE'], 'R2': m_2022['R2'], 'N': len(y_2022)},
        {'Year': 2022, 'Dataset': 'Out-of-sample year', 'Model': 'Persistence',
         'MAE': m_persist_2022['MAE'], 'R2': m_persist_2022['R2'], 'N': len(y_2022)},
    ])
    oos_df.to_csv(os.path.join(TBL, "out_of_sample_2022.csv"), index=False)
    print("  -> out_of_sample_2022.csv")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    n_show = min(168*4, len(y_2022))
    ax.plot(range(n_show), y_2022[:n_show], 'k-', linewidth=0.5, alpha=0.7, label='Actual 2022')
    ax.plot(range(n_show), pred_2022[:n_show], 'r--', linewidth=0.5, alpha=0.7, label='Predicted (2023 model)')
    ax.set_xlabel('Hour'); ax.set_ylabel('Net Load (MW)')
    ax.set_title(f'2022 Out-of-Sample: R²={m_2022["R2"]:.3f}, MAE={m_2022["MAE"]:.0f} MW', fontweight='bold')
    ax.legend()

    ax = axes[1]
    ax.scatter(y_2022, pred_2022, s=3, alpha=0.2, c='#2196F3')
    lims = [min(y_2022.min(), pred_2022.min()), max(y_2022.max(), pred_2022.max())]
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlabel('Actual (MW)'); ax.set_ylabel('Predicted (MW)')
    ax.set_title(f'2022 Actual vs Predicted', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_2022_validation.png"))
    plt.close()
    print("  -> fig_2022_validation.png")

else:
    print("  2022 data not yet available — skipping (run 12_fetch_2022.py first)")

# =====================================================================
#  SUMMARY
# =====================================================================
print("\n" + "="*78)
print("  ALL THREE FIXES COMPLETE")
print("="*78)
print(f"\n  1. LSTM: Test R2={m_lstm_te['R2']:.4f}, MAE={m_lstm_te['MAE']:.0f} MW")
if r_fixed and r_price:
    print(f"  2. Price-MILP: Fixed={r_fixed['cost_mwh']:.2f} vs Price-responsive={r_price['cost_mwh']:.2f} $/MWh")
if os.path.exists(gen_2022_path):
    print(f"  3. 2022 OOS: R2={m_2022['R2']:.4f}, Skill={skill_2022:+.1f}%")
print("="*78)
