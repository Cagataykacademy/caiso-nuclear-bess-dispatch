"""
Gamma sweep: robustness parameter γ ∈ {0, 0.25, 0.50, 0.75, 1.00}
Uses November ML predictions (same as S1/S6/S2 scenarios) with two-tier CCGT UC model.
Outputs: outputs/tables/gamma_sweep.csv
"""
import os, sys, io, time, warnings
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

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
TBL  = os.path.join(P, "outputs", "tables")

# ── PARAMETERS (identical to scripts 16 and 17) ───────────────────────────
NUC_CAP=2256; NUC_MIN=1800; NUC_RAMP=100; NUC_COST=12
N_CCGT=2; CCGT_UNIT_MW=5000; CCGT_MIN_MW=1500; CCGT_RAMP=800
CCGT_COST=40; CCGT_STARTUP=25000; CCGT_UP_MIN=4; CCGT_DN_MIN=2
PEAK_CAP=10000; PEAK_RAMP=5000; PEAK_COST=65
BESS_POW=5000; BESS_ENE=20000; BESS_EFF=0.90
SOC_MIN=0.10; SOC_MAX=0.90; SOC_INIT=0.50; SOC_END=0.35; BESS_DEG=5
IMP_CAP=10000; EXP_CAP=6000; IMP_COST=55; EXP_REV=20
SHED_CAP=5000; VOLL=10000; CURT_CAP=20000; CURT_COST=10
HORIZON=168

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
    m.SoC    = pyo.Var(m.T, bounds=(SOC_MIN*BESS_ENE, SOC_MAX*BESS_ENE), initialize=SOC_INIT*BESS_ENE)
    m.P_imp  = pyo.Var(m.T, bounds=(0, IMP_CAP), initialize=0)
    m.P_exp  = pyo.Var(m.T, bounds=(0, EXP_CAP), initialize=0)
    m.P_shed = pyo.Var(m.T, bounds=(0, SHED_CAP), initialize=0)
    m.P_curt = pyo.Var(m.T, bounds=(0, CURT_CAP), initialize=0)
    m.u_bess = pyo.Var(m.T, within=pyo.Binary, initialize=0)

    def obj(mdl):
        return sum(NUC_COST*mdl.P_nuc[t]
            + CCGT_COST*sum(mdl.x_ccgt[t,k] for k in mdl.K)
            + CCGT_STARTUP*sum(mdl.y_ccgt[t,k] for k in mdl.K)
            + PEAK_COST*mdl.P_peak[t]
            + BESS_DEG*(mdl.P_ch[t]+mdl.P_dis[t])
            + IMP_COST*mdl.P_imp[t] - EXP_REV*mdl.P_exp[t]
            + VOLL*mdl.P_shed[t] + CURT_COST*mdl.P_curt[t] for t in mdl.T)
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    def balance(mdl,t):
        return (mdl.P_nuc[t]+sum(mdl.x_ccgt[t,k] for k in mdl.K)
                +mdl.P_peak[t]+mdl.P_dis[t]+mdl.P_imp[t]+mdl.P_shed[t]
                ==nl_profile[t]+mdl.P_ch[t]+mdl.P_exp[t]+mdl.P_curt[t])
    m.c_bal=pyo.Constraint(m.T, rule=balance)
    eta=np.sqrt(BESS_EFF)
    m.c_soc0=pyo.Constraint(expr=m.SoC[0]==SOC_INIT*BESS_ENE)
    m.c_soc_end=pyo.Constraint(expr=m.SoC[T-1]>=SOC_END*BESS_ENE)
    def soc_dyn(mdl,t):
        if t==0: return pyo.Constraint.Skip
        return mdl.SoC[t]==mdl.SoC[t-1]+eta*mdl.P_ch[t]-(1/eta)*mdl.P_dis[t]
    m.c_soc=pyo.Constraint(m.T, rule=soc_dyn)
    def ch(mdl,t): return mdl.P_ch[t]<=BESS_POW*(1-mdl.u_bess[t])
    def dis(mdl,t): return mdl.P_dis[t]<=BESS_POW*mdl.u_bess[t]
    m.c_ch=pyo.Constraint(m.T,rule=ch); m.c_dis=pyo.Constraint(m.T,rule=dis)
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
        obj_val=pyo.value(m.obj)
        nl_sum=float(np.sum(nl_profile))
        shed=sum(pyo.value(m.P_shed[t]) or 0 for t in range(T))
        gas=sum(sum(pyo.value(m.x_ccgt[t,k]) or 0 for k in range(N_CCGT))
               +pyo.value(m.P_peak[t]) or 0 for t in range(T))/T
        return {'cost_mwh': obj_val/nl_sum if nl_sum>0 else 0,
                'shed': shed, 'avg_gas': gas, 'time': elapsed}
    return None

# ── LOAD DATA ─────────────────────────────────────────────────────────────
pred = pd.read_csv(os.path.join(DATA,'ml_predictions_for_milp.csv'),
                   parse_dates=['timestamp']).set_index('timestamp')

nl_point = pred['net_load_predicted'].values[:HORIZON]  # ML point forecast
nl_q90   = pred['net_load_Q90'].values[:HORIZON]        # CQR upper bound

print("="*60)
print("  GAMMA SWEEP — November week, two-tier CCGT UC model")
print("="*60)
print(f"  Point NL mean: {nl_point.mean():.0f} MW")
print(f"  Q90  NL mean:  {nl_q90.mean():.0f} MW")
print(f"  Mean PI width: {(nl_q90 - nl_point).mean():.0f} MW")
print()

rows = []
for gamma in [0.00, 0.25, 0.50, 0.75, 1.00]:
    nl = nl_point + gamma * (nl_q90 - nl_point)
    label = f"gamma={gamma:.2f}"
    print(f"  Running {label} (NL mean={nl.mean():.0f} MW)...", end=' ', flush=True)
    r = solve_dispatch(nl, label)
    if r:
        print(f"${r['cost_mwh']:.2f}/MWh  shed={r['shed']:.0f}  t={r['time']:.2f}s")
        rows.append({
            'gamma': gamma,
            'gamma_label': '0.00 (deterministic)' if gamma==0 else
                           '1.00 (worst-case Q90)' if gamma==1 else f'{gamma:.2f}',
            'cost_mwh': round(r['cost_mwh'], 2),
            'shed_mwh': round(r['shed'], 0),
            'avg_gas_mw': round(r['avg_gas'], 0),
        })
    else:
        print("INFEASIBLE")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(TBL,'gamma_sweep.csv'), index=False)

print()
print("  RESULTS:")
print(df[['gamma_label','cost_mwh','shed_mwh','avg_gas_mw']].to_string(index=False))
print()
print(f"  Cost range: ${df.cost_mwh.min():.2f} – ${df.cost_mwh.max():.2f}/MWh")
d = df.cost_mwh.max() - df.cost_mwh.min()
print(f"  Cost of full robustness: +${d:.2f}/MWh (+{d/df.cost_mwh.min()*100:.1f}%)")
print("="*60)
