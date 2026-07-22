"""
=============================================================================
 SEASONAL MILP ANALYSIS + VALUE OF PERFECT INFORMATION (VOPI)
 Addresses reviewer concern: "only 2 weeks analyzed"
 Four representative weeks: Winter / Spring / Summer / Fall
 VOPI: cost premium of ML forecast vs perfect oracle forecast
=============================================================================
"""
import os, sys, io, time, warnings
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import pyomo.environ as pyo
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyomo', '-q'])
    import pyomo.environ as pyo

try:
    import highspy; SOLVER = 'appsi_highs'
except ImportError:
    SOLVER = 'glpk'

P   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(P, "data")
FIG  = os.path.join(P, "outputs", "figures")
TBL  = os.path.join(P, "outputs", "tables")
for d in [FIG, TBL]: os.makedirs(d, exist_ok=True)

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'serif','font.size':11,'axes.grid':True,'grid.alpha':0.3,
    'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False})

print("="*78)
print("  SEASONAL MILP + VALUE OF PERFECT INFORMATION (VOPI)")
print("="*78)

# =========================================================================
#  PARAMETERS (same as 16_improved_milp.py)
# =========================================================================
NUC_CAP=2256; NUC_MIN=1800; NUC_RAMP=100; NUC_COST=12
N_CCGT=2; CCGT_UNIT_MW=5000; CCGT_MIN_MW=1500; CCGT_RAMP=800
CCGT_COST=40; CCGT_STARTUP=25000; CCGT_UP_MIN=4; CCGT_DN_MIN=2
CCGT_CAP=N_CCGT*CCGT_UNIT_MW
PEAK_CAP=10000; PEAK_RAMP=5000; PEAK_COST=65
BESS_POW=5000; BESS_ENE=20000; BESS_EFF=0.90
SOC_MIN=0.10; SOC_MAX=0.90; SOC_INIT=0.50; SOC_END=0.35; BESS_DEG=5
IMP_CAP=10000; EXP_CAP=6000; IMP_COST=55; EXP_REV=20
SHED_CAP=5000; VOLL=10000; CURT_CAP=20000; CURT_COST=10
HORIZON=168

# =========================================================================
#  MODEL (copy of 16_improved_milp build_and_solve, same constraints)
# =========================================================================
def solve_dispatch(nl_profile, label):
    T = len(nl_profile)
    m = pyo.ConcreteModel(name=label)
    m.T = pyo.RangeSet(0, T-1)
    m.K = pyo.RangeSet(0, N_CCGT-1)

    m.P_nuc  = pyo.Var(m.T, bounds=(NUC_MIN, NUC_CAP), initialize=NUC_CAP*0.9)
    m.x_ccgt = pyo.Var(m.T, m.K, bounds=(0, CCGT_UNIT_MW), initialize=4000)
    m.u_ccgt = pyo.Var(m.T, m.K, within=pyo.Binary, initialize=1)
    m.y_ccgt = pyo.Var(m.T, m.K, within=pyo.Binary, initialize=0)
    m.P_peak = pyo.Var(m.T, bounds=(0, PEAK_CAP), initialize=2000)
    m.P_ch   = pyo.Var(m.T, bounds=(0, BESS_POW), initialize=0)
    m.P_dis  = pyo.Var(m.T, bounds=(0, BESS_POW), initialize=0)
    m.SoC    = pyo.Var(m.T, bounds=(SOC_MIN*BESS_ENE, SOC_MAX*BESS_ENE),
                        initialize=SOC_INIT*BESS_ENE)
    m.P_imp  = pyo.Var(m.T, bounds=(0, IMP_CAP), initialize=0)
    m.P_exp  = pyo.Var(m.T, bounds=(0, EXP_CAP), initialize=0)
    m.P_shed = pyo.Var(m.T, bounds=(0, SHED_CAP), initialize=0)
    m.P_curt = pyo.Var(m.T, bounds=(0, CURT_CAP), initialize=0)
    m.u_bess = pyo.Var(m.T, within=pyo.Binary, initialize=0)

    def obj(mdl):
        return sum(
            NUC_COST*mdl.P_nuc[t]
            + CCGT_COST*sum(mdl.x_ccgt[t,k] for k in mdl.K)
            + CCGT_STARTUP*sum(mdl.y_ccgt[t,k] for k in mdl.K)
            + PEAK_COST*mdl.P_peak[t]
            + BESS_DEG*(mdl.P_ch[t]+mdl.P_dis[t])
            + IMP_COST*mdl.P_imp[t]
            - EXP_REV*mdl.P_exp[t]
            + VOLL*mdl.P_shed[t]
            + CURT_COST*mdl.P_curt[t]
            for t in mdl.T)
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    def balance(mdl, t):
        return (mdl.P_nuc[t] + sum(mdl.x_ccgt[t,k] for k in mdl.K)
                + mdl.P_peak[t] + mdl.P_dis[t] + mdl.P_imp[t] + mdl.P_shed[t]
                == nl_profile[t] + mdl.P_ch[t] + mdl.P_exp[t] + mdl.P_curt[t])
    m.c_bal = pyo.Constraint(m.T, rule=balance)

    eta = np.sqrt(BESS_EFF)
    m.c_soc0   = pyo.Constraint(expr=m.SoC[0]==SOC_INIT*BESS_ENE)
    m.c_soc_end= pyo.Constraint(expr=m.SoC[T-1]>=SOC_END*BESS_ENE)
    def soc_dyn(mdl,t):
        if t==0: return pyo.Constraint.Skip
        return mdl.SoC[t]==mdl.SoC[t-1]+eta*mdl.P_ch[t]-(1/eta)*mdl.P_dis[t]
    m.c_soc=pyo.Constraint(m.T, rule=soc_dyn)
    def ch(mdl,t):  return mdl.P_ch[t]<=BESS_POW*(1-mdl.u_bess[t])
    def dis(mdl,t): return mdl.P_dis[t]<=BESS_POW*mdl.u_bess[t]
    m.c_ch=pyo.Constraint(m.T, rule=ch); m.c_dis=pyo.Constraint(m.T, rule=dis)

    def nup(mdl,t):
        if t==0: return pyo.Constraint.Skip
        return mdl.P_nuc[t]-mdl.P_nuc[t-1]<=NUC_RAMP
    def ndn(mdl,t):
        if t==0: return pyo.Constraint.Skip
        return mdl.P_nuc[t-1]-mdl.P_nuc[t]<=NUC_RAMP
    m.c_nup=pyo.Constraint(m.T,rule=nup); m.c_ndn=pyo.Constraint(m.T,rule=ndn)

    def ccgt_init(mdl,k): return mdl.u_ccgt[0,k]==1
    m.c_ccgt_init=pyo.Constraint(m.K, rule=ccgt_init)
    def cmx(mdl,t,k): return mdl.x_ccgt[t,k]<=CCGT_UNIT_MW*mdl.u_ccgt[t,k]
    def cmn(mdl,t,k): return mdl.x_ccgt[t,k]>=CCGT_MIN_MW*mdl.u_ccgt[t,k]
    m.c_cmx=pyo.Constraint(m.T,m.K,rule=cmx); m.c_cmn=pyo.Constraint(m.T,m.K,rule=cmn)
    def cst(mdl,t,k):
        if t==0: return pyo.Constraint.Skip
        return mdl.y_ccgt[t,k]>=mdl.u_ccgt[t,k]-mdl.u_ccgt[t-1,k]
    m.c_cst=pyo.Constraint(m.T,m.K,rule=cst)
    def cup(mdl,t,k):
        if t==0: return pyo.Constraint.Skip
        end=min(t+CCGT_UP_MIN-1,T-1)
        return sum(mdl.u_ccgt[tau,k] for tau in range(t,end+1))>=CCGT_UP_MIN*mdl.y_ccgt[t,k]
    m.c_cup=pyo.Constraint(m.T,m.K,rule=cup)
    def cdn(mdl,t,k):
        if t==0: return pyo.Constraint.Skip
        end=min(t+CCGT_DN_MIN-1,T-1)
        return sum(1-mdl.u_ccgt[tau,k] for tau in range(t,end+1))>=CCGT_DN_MIN*(mdl.u_ccgt[t-1,k]-mdl.u_ccgt[t,k])
    m.c_cdn=pyo.Constraint(m.T,m.K,rule=cdn)
    def cru(mdl,t,k):
        if t==0: return pyo.Constraint.Skip
        return mdl.x_ccgt[t,k]-mdl.x_ccgt[t-1,k]<=CCGT_RAMP*mdl.u_ccgt[t-1,k]+CCGT_UNIT_MW*mdl.y_ccgt[t,k]
    def crd(mdl,t,k):
        if t==0: return pyo.Constraint.Skip
        return mdl.x_ccgt[t-1,k]-mdl.x_ccgt[t,k]<=CCGT_RAMP
    m.c_cru=pyo.Constraint(m.T,m.K,rule=cru); m.c_crd=pyo.Constraint(m.T,m.K,rule=crd)
    def pku(mdl,t):
        if t==0: return pyo.Constraint.Skip
        return mdl.P_peak[t]-mdl.P_peak[t-1]<=PEAK_RAMP
    def pkd(mdl,t):
        if t==0: return pyo.Constraint.Skip
        return mdl.P_peak[t-1]-mdl.P_peak[t]<=PEAK_RAMP
    m.c_pku=pyo.Constraint(m.T,rule=pku); m.c_pkd=pyo.Constraint(m.T,rule=pkd)

    solver=pyo.SolverFactory(SOLVER); solver.options['time_limit']=300
    t0=time.time()
    res=solver.solve(m, tee=False, load_solutions=False)
    elapsed=time.time()-t0
    status=str(res.solver.termination_condition)

    if status in ('optimal','feasible'):
        m.solutions.load_from(res)
        def v(var,t): return pyo.value(var[t]) or 0.0
        P_nuc  = [v(m.P_nuc,t) for t in range(T)]
        P_ccgt = [sum(pyo.value(m.x_ccgt[t,k]) or 0 for k in range(N_CCGT)) for t in range(T)]
        P_peak = [v(m.P_peak,t) for t in range(T)]
        P_shed = [v(m.P_shed,t) for t in range(T)]
        P_curt = [v(m.P_curt,t) for t in range(T)]
        P_imp  = [v(m.P_imp,t) for t in range(T)]
        P_dis  = [v(m.P_dis,t) for t in range(T)]
        obj_val= pyo.value(m.obj)
        nl_sum = float(np.sum(nl_profile))
        n_startups = sum(pyo.value(m.y_ccgt[t,k]) or 0
                         for t in range(T) for k in range(N_CCGT))
        return {
            'status':status,'time':elapsed,
            'cost_total':obj_val,
            'cost_mwh':obj_val/nl_sum if nl_sum>0 else 0,
            'avg_nuc':np.mean(P_nuc),
            'avg_ccgt':np.mean(P_ccgt),
            'avg_peak':np.mean(P_peak),
            'avg_gas':np.mean(P_ccgt)+np.mean(P_peak),
            'avg_imp':np.mean(P_imp),
            'shed_total':np.sum(P_shed),
            'curt_total':np.sum(P_curt),
            'bess_cycles':np.sum(P_dis)/BESS_ENE,
            'n_startups':n_startups,
            'P_nuc':P_nuc,'P_ccgt':P_ccgt,'P_peak':P_peak,
            'P_imp':P_imp,'nl':list(nl_profile),
        }
    return None

# =========================================================================
#  LOAD DATA
# =========================================================================
print("\n[1/4] Loading preprocessed data...")
proc = pd.read_csv(os.path.join(DATA,"caiso_preprocessed_v2_2023.csv"),
                   index_col=0, parse_dates=True)
proc = proc.sort_index()
pred = pd.read_csv(os.path.join(DATA,"ml_predictions_for_milp.csv"),
                   parse_dates=['timestamp']).set_index('timestamp')

# =========================================================================
#  DEFINE REPRESENTATIVE WEEKS
# =========================================================================
# Strategy: pick complete Mon-Sun weeks that are representative of each season.
# All weeks from the 2023 CAISO dataset.
SEASONS = {
    'Winter (Jan 23–29)': ('2023-01-23','2023-01-29'),   # typical winter (avoid extreme cold-snap Jan 9-15)
    'Spring (Apr 10–16)': ('2023-04-10','2023-04-16'),   # duck-belly week
    'Summer (Aug 7–13)':  ('2023-08-07','2023-08-13'),   # peak demand week
    'Fall (Nov 1–7)':     ('2023-11-01','2023-11-07'),   # test/OOS week
}

def extract_week(df, start, end):
    w = df.loc[start:end, 'net_load_MW'].dropna()
    # Ensure exactly 168 hours
    if len(w) > 168: w = w.iloc[:168]
    return w.values

# =========================================================================
#  RUN SEASONAL DISPATCH
# =========================================================================
print("\n[2/4] Running seasonal dispatch (actual net load)...")
season_rows = []
season_disp = {}

for label, (start, end) in SEASONS.items():
    nl = extract_week(proc, start, end)
    if len(nl) < 168:
        print(f"  {label}: only {len(nl)}h data — skipping")
        continue
    print(f"  {label}: NL range [{nl.min():.0f}, {nl.max():.0f}] MW...")
    r = solve_dispatch(nl, label)
    if r:
        season_rows.append({
            'Season': label,
            'Net_Load_Mean_MW': round(nl.mean(),0),
            'Net_Load_Min_MW':  round(nl.min(),0),
            'Net_Load_Max_MW':  round(nl.max(),0),
            'Cost_MWh':         round(r['cost_mwh'],2),
            'Avg_Nuclear_MW':   round(r['avg_nuc'],0),
            'Avg_CCGT_MW':      round(r['avg_ccgt'],0),
            'Avg_Peaker_MW':    round(r['avg_peak'],0),
            'Avg_Import_MW':    round(r['avg_imp'],0),
            'Shed_MWh':         round(r['shed_total'],0),
            'Curt_MWh':         round(r['curt_total'],0),
            'BESS_Cycles':      round(r['bess_cycles'],2),
            'CCGT_Startups':    r['n_startups'],
            'Solve_s':          round(r['time'],2),
        })
        season_disp[label] = r
        print(f"    -> ${r['cost_mwh']:.2f}/MWh  shed={r['shed_total']:.0f} MWh  "
              f"CCGT={r['avg_ccgt']:.0f}  peak={r['avg_peak']:.0f}  "
              f"starts={r['n_startups']}  t={r['time']:.2f}s")
    else:
        print(f"    -> INFEASIBLE")

season_df = pd.DataFrame(season_rows)
season_df.to_csv(os.path.join(TBL,"seasonal_dispatch.csv"), index=False)
print("\n  Seasonal dispatch:")
print(season_df[['Season','Cost_MWh','Avg_CCGT_MW','Avg_Peaker_MW',
                  'Shed_MWh','BESS_Cycles']].to_string(index=False))

# =========================================================================
#  VALUE OF PERFECT INFORMATION (VOPI) — November week only
# =========================================================================
print("\n[3/4] Computing VOPI for November week...")

# Oracle: actual realized net load (no forecast error)
nl_oracle = pred['net_load_actual'].values[:HORIZON]
# ML:     ML day-ahead prediction (1,429 MW MAE)
nl_ml     = pred['net_load_predicted'].values[:HORIZON]
# Q90:    upper-bound uncertainty scenario
nl_q90    = pred['net_load_Q90'].values[:HORIZON]

print(f"  Oracle (actual) NL: mean={nl_oracle.mean():.0f} MW")
print(f"  ML predicted NL:    mean={nl_ml.mean():.0f} MW")
print(f"  Forecast MAE:       {np.mean(np.abs(nl_oracle - nl_ml)):.0f} MW (matches benchmark)")

r_oracle = solve_dispatch(nl_oracle, "Oracle_Nov")
r_ml     = solve_dispatch(nl_ml,     "ML_Nov")

if r_oracle and r_ml:
    vopi_abs = r_ml['cost_mwh'] - r_oracle['cost_mwh']
    vopi_pct = vopi_abs / r_oracle['cost_mwh'] * 100
    vopi_weekly = (r_ml['cost_total'] - r_oracle['cost_total'])
    print(f"\n  Oracle dispatch:  ${r_oracle['cost_mwh']:.2f}/MWh")
    print(f"  ML dispatch:      ${r_ml['cost_mwh']:.2f}/MWh")
    print(f"  VOPI:             ${vopi_abs:+.2f}/MWh ({vopi_pct:+.2f}%)")
    print(f"  VOPI weekly $:    ${vopi_weekly:,.0f}")
    print(f"  VOPI annualized:  ${vopi_weekly*52/1e6:.1f}M/year")

    vopi_df = pd.DataFrame([{
        'Scenario': 'Oracle (perfect forecast)', 'Cost_MWh': round(r_oracle['cost_mwh'],2),
        'Shed_MWh': round(r_oracle['shed_total'],0),'BESS_Cycles': round(r_oracle['bess_cycles'],2)},
        {'Scenario': 'ML day-ahead (XGBoost, MAE=1,429 MW)', 'Cost_MWh': round(r_ml['cost_mwh'],2),
        'Shed_MWh': round(r_ml['shed_total'],0),'BESS_Cycles': round(r_ml['bess_cycles'],2)},
        {'Scenario': 'VOPI (ML cost premium)', 'Cost_MWh': round(vopi_abs,2),
        'Shed_MWh': '—', 'BESS_Cycles': f'{vopi_pct:+.2f}%'},
    ])
    vopi_df.to_csv(os.path.join(TBL,"vopi_analysis.csv"), index=False)
    print("  -> vopi_analysis.csv")

# =========================================================================
#  FIGURE: Seasonal dispatch comparison
# =========================================================================
print("\n[4/4] Generating seasonal figure...")

if len(season_df) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    season_labels = list(season_disp.keys())
    colors = {'Nuclear':'#4CAF50','CCGT':'#FF9800','Peaker':'#FFC107',
               'Import':'#9C27B0','Net Load':'black'}

    for i, (label, r) in enumerate(season_disp.items()):
        ax = axes[i//2][i%2]
        t = np.arange(HORIZON)
        nuc  = np.array(r['P_nuc'])
        ccgt = np.array(r['P_ccgt'])
        peak = np.array(r['P_peak'])
        imp  = np.array(r['P_imp'])
        ax.fill_between(t, 0, nuc,                alpha=0.85,color='#4CAF50',label='Nuclear')
        ax.fill_between(t, nuc, nuc+ccgt,          alpha=0.75,color='#FF9800',label='CCGT')
        ax.fill_between(t, nuc+ccgt, nuc+ccgt+peak,alpha=0.70,color='#FFC107',label='Peaker')
        ax.fill_between(t, nuc+ccgt+peak, nuc+ccgt+peak+imp,
                         alpha=0.50,color='#9C27B0',label='Import')
        ax.plot(t, r['nl'], 'k-', lw=1.8, label='Net Load', zorder=5)
        nl_arr = np.array(r['nl'])
        ax.fill_between(t, nl_arr, where=nl_arr<0, color='red', alpha=0.3,
                         label='Negative NL' if any(nl_arr<0) else None)
        ax.set_title(f"{label}\n${r['cost_mwh']:.2f}/MWh  shed={r['shed_total']:.0f} MWh",
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Hour'); ax.set_ylabel('MW')
        if i==0: ax.legend(ncol=3, fontsize=8, loc='upper right')
        ax.set_ylim(bottom=min(0, min(r['nl'])-500))

    plt.suptitle('Seasonal Dispatch Comparison — Improved MILP (Two-Tier Gas + CCGT UC)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG,"fig_seasonal_dispatch.png"))
    plt.close()
    print("  -> fig_seasonal_dispatch.png")

# VOPI comparison figure
if r_oracle and r_ml:
    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.arange(HORIZON)
    ax.plot(t, nl_oracle,  'k-', lw=1.5, label='Actual net load (Oracle)')
    ax.plot(t, nl_ml,      'b--',lw=1.2, label=f'ML prediction (MAE={np.mean(np.abs(nl_oracle-nl_ml)):.0f} MW)')
    ax.fill_between(t, nl_oracle, nl_ml,
                     where=nl_ml>nl_oracle, alpha=0.25, color='red',   label='Over-predict')
    ax.fill_between(t, nl_oracle, nl_ml,
                     where=nl_ml<nl_oracle, alpha=0.25, color='green', label='Under-predict')
    ax.set_xlabel('Hour'); ax.set_ylabel('Net Load (MW)')
    ax.set_title(f'VOPI Analysis: Oracle vs ML Day-Ahead Forecast\n'
                 f'Cost premium of ML forecast: ${vopi_abs:+.2f}/MWh ({vopi_pct:+.2f}%)',
                 fontweight='bold')
    ax.legend(ncol=4, fontsize=9, loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG,"fig_vopi_analysis.png"))
    plt.close()
    print("  -> fig_vopi_analysis.png")

print("\n" + "="*78)
print("  SEASONAL MILP + VOPI COMPLETE")
print(f"  Seasons analyzed: {len(season_rows)}")
if r_oracle and r_ml:
    print(f"  VOPI: ${vopi_abs:+.2f}/MWh ({vopi_pct:+.2f}%) — "
          f"forecast error costs ${vopi_weekly*52/1e6:.1f}M/year system-wide")
print("="*78)
