"""
=============================================================================
 PHASE 3 (IMPROVED): MILP with Two-Tier Gas + End-SoC Constraint
 Key improvements over 05b:
  1. Two-tier gas: CCGT (efficient, slow ramp) + Peaker (flexible, expensive)
  2. End-of-period SoC constraint (prevents end-game BESS depletion)
  3. Consistent nuclear ramp (100 MW/h, reflecting PWR load-following limit)
=============================================================================
Horizon justification:
  168 hours (one week) captures the full weekly periodicity of CAISO net load
  and BESS cycling behavior. A 24h horizon undervalues inter-day arbitrage
  (e.g., Sunday low-demand charging for Monday evening ramp). A 168h horizon
  contains one complete weekday/weekend cycle and aligns with CAISO's weekly
  energy storage planning reports, while remaining computationally tractable.
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
    import highspy
    SOLVER = 'appsi_highs'
except ImportError:
    SOLVER = 'glpk'

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(P, "data")
FIG  = os.path.join(P, "outputs", "figures")
TBL  = os.path.join(P, "outputs", "tables")
for d in [FIG, TBL]: os.makedirs(d, exist_ok=True)

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'serif','font.size':11,'axes.grid':True,'grid.alpha':0.3,
    'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9,
    'axes.spines.top':False,'axes.spines.right':False})

print("="*78)
print("  IMPROVED MILP: TWO-TIER GAS + END-SOC CONSTRAINT")
print("="*78)

# =========================================================================
#  SYSTEM PARAMETERS
# =========================================================================
# Nuclear (Diablo Canyon — 2 × 1,128 MW PWR)
NUC_CAP  = 2256    # MW maximum
NUC_MIN  = 1800    # MW minimum (80% stable output floor)
NUC_RAMP = 100     # MW/h — PWR load-following limit (~4.4%/h of rated capacity)
NUC_COST = 12      # $/MWh fuel + O&M (zero-carbon baseload)

# TIER 1 — CCGT (Combined Cycle Gas Turbines) — 2 aggregate units of 5,000 MW each
# Efficient, but slower to ramp; modeled with unit commitment (startup cost, min up/down time)
N_CCGT       = 2        # number of aggregate CCGT units
CCGT_UNIT_MW = 5000     # MW per unit
CCGT_MIN_MW  = 1500     # MW minimum load per unit when committed (30%)
CCGT_RAMP    = 800      # MW/h per unit (steam turbine thermal inertia)
CCGT_COST    = 40       # $/MWh variable fuel + O&M
CCGT_STARTUP = 25000    # $ per startup event (cold-start fuel + wear costs, aggregate)
CCGT_UP_MIN  = 4        # h minimum time online after startup
CCGT_DN_MIN  = 2        # h minimum time offline after shutdown
CCGT_CAP     = N_CCGT * CCGT_UNIT_MW  # 10,000 MW total

# TIER 2 — Peakers (Simple Cycle GTs + reciprocating engines)
# Expensive but very fast; deployed only when CCGT cannot ramp fast enough
PEAK_CAP  = 10000  # MW (total peaker fleet)
PEAK_RAMP = 5000   # MW/h (simple cycle GT can ramp near-instantaneously)
PEAK_COST = 65     # $/MWh (heat rate ~10.5 MMBtu/MWh × $5.5 + $7 O&M)

# BESS (California's 5 GW / 4-hour Li-ion fleet as of 2023)
BESS_POW  = 5000   # MW
BESS_ENE  = 20000  # MWh (4-hour duration)
BESS_EFF  = 0.90   # round-trip efficiency
SOC_MIN   = 0.10   # % — lower bound to protect battery life
SOC_MAX   = 0.90   # % — upper bound
SOC_INIT  = 0.50   # % — initial state of charge
SOC_END   = 0.35   # % — end-of-horizon minimum (prevents end-game depletion)
BESS_DEG  = 5      # $/MWh throughput degradation cost

# Grid services
IMP_CAP   = 10000  # MW import limit (CAISO tie-line capacity)
EXP_CAP   = 6000   # MW export limit
IMP_COST  = 55     # $/MWh import price
EXP_REV   = 20     # $/MWh export revenue
SHED_CAP  = 5000   # MW maximum curtailable load
VOLL      = 10000  # $/MWh value of lost load penalty
CURT_CAP  = 20000  # MW curtailment cap
CURT_COST = 10     # $/MWh renewable curtailment opportunity cost

HORIZON   = 168    # hours = 1 week (see module docstring for justification)

# =========================================================================
#  LOAD PREDICTIONS
# =========================================================================
print("\n[1/4] Loading ML predictions...")
pred = pd.read_csv(os.path.join(DATA, "ml_predictions_for_milp.csv"))
pred['timestamp'] = pd.to_datetime(pred['timestamp'])
week = pred.head(HORIZON).copy()
print(f"  Period: {week['timestamp'].iloc[0]} → {week['timestamp'].iloc[-1]}")

# =========================================================================
#  MODEL BUILDER
# =========================================================================
def build_and_solve(nl_profile, name, nuc_cap=NUC_CAP, nuc_min=NUC_MIN,
                    use_nuc=True, bess_pow=BESS_POW, bess_ene=BESS_ENE,
                    ccgt_cost=CCGT_COST, peak_cost=PEAK_COST):
    """
    Build and solve improved MILP economic dispatch.

    Two-tier gas model:
      - CCGT: efficient (low cost), constrained ramp — compensates for nuclear variability
      - Peaker: flexible (high cost), fast ramp — handles rapid net load swings

    End-of-horizon SoC constraint:
      SoC[T-1] >= SOC_END * bess_ene
      Prevents the optimizer from gaming the finite horizon by draining BESS
      in the final hours (a known artifact of rolling-horizon dispatch formulations).
    """
    T = len(nl_profile)
    m = pyo.ConcreteModel(name=f"Dispatch_{name}")
    m.T = pyo.RangeSet(0, T-1)

    # --- Decision variables ---
    if use_nuc:
        m.P_nuc = pyo.Var(m.T, bounds=(nuc_min, nuc_cap), initialize=nuc_cap*0.9)
    else:
        m.P_nuc = pyo.Var(m.T, bounds=(0, 0), initialize=0)

    # CCGT unit commitment variables (k=0,1 for 2 aggregate units of 5,000 MW each)
    m.K = pyo.RangeSet(0, N_CCGT - 1)
    m.x_ccgt = pyo.Var(m.T, m.K, bounds=(0, CCGT_UNIT_MW), initialize=4000)  # output MW
    m.u_ccgt = pyo.Var(m.T, m.K, within=pyo.Binary, initialize=1)  # commitment (1=online)
    m.y_ccgt = pyo.Var(m.T, m.K, within=pyo.Binary, initialize=0)  # startup indicator
    m.P_peak = pyo.Var(m.T, bounds=(0, PEAK_CAP), initialize=2000)
    m.P_ch   = pyo.Var(m.T, bounds=(0, bess_pow), initialize=0)
    m.P_dis  = pyo.Var(m.T, bounds=(0, bess_pow), initialize=0)
    m.SoC    = pyo.Var(m.T, bounds=(SOC_MIN*bess_ene, SOC_MAX*bess_ene),
                       initialize=SOC_INIT*bess_ene)
    m.P_imp  = pyo.Var(m.T, bounds=(0, IMP_CAP), initialize=0)
    m.P_exp  = pyo.Var(m.T, bounds=(0, EXP_CAP), initialize=0)
    m.P_shed = pyo.Var(m.T, bounds=(0, SHED_CAP), initialize=0)
    m.P_curt = pyo.Var(m.T, bounds=(0, CURT_CAP), initialize=0)
    m.u      = pyo.Var(m.T, within=pyo.Binary, initialize=0)  # BESS mode (1=discharge)

    # --- Objective: minimise total system cost (includes CCGT startup costs) ---
    def obj(mdl):
        return sum(
            NUC_COST     * mdl.P_nuc[t]
            + ccgt_cost  * sum(mdl.x_ccgt[t, k] for k in mdl.K)
            + CCGT_STARTUP * sum(mdl.y_ccgt[t, k] for k in mdl.K)
            + peak_cost  * mdl.P_peak[t]
            + BESS_DEG   * (mdl.P_ch[t] + mdl.P_dis[t])
            + IMP_COST   * mdl.P_imp[t]
            - EXP_REV    * mdl.P_exp[t]
            + VOLL       * mdl.P_shed[t]
            + CURT_COST  * mdl.P_curt[t]
            for t in mdl.T)
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    # --- C1: Power balance (every hour) ---
    def balance(mdl, t):
        ccgt_total = sum(mdl.x_ccgt[t, k] for k in mdl.K)
        supply = mdl.P_nuc[t] + ccgt_total + mdl.P_peak[t] \
                 + mdl.P_dis[t] + mdl.P_imp[t] + mdl.P_shed[t]
        demand = nl_profile[t] + mdl.P_ch[t] + mdl.P_exp[t] + mdl.P_curt[t]
        return supply == demand
    m.c_balance = pyo.Constraint(m.T, rule=balance)

    # --- C2: BESS SoC dynamics (round-trip η = 90%) ---
    eta = np.sqrt(BESS_EFF)
    m.c_soc0 = pyo.Constraint(expr=m.SoC[0] == SOC_INIT * bess_ene)

    def soc_dyn(mdl, t):
        if t == 0: return pyo.Constraint.Skip
        return mdl.SoC[t] == mdl.SoC[t-1] + eta*mdl.P_ch[t] - (1/eta)*mdl.P_dis[t]
    m.c_soc = pyo.Constraint(m.T, rule=soc_dyn)

    # --- C3: End-of-horizon SoC floor (prevent end-game depletion) ---
    m.c_soc_end = pyo.Constraint(expr=m.SoC[T-1] >= SOC_END * bess_ene)

    # --- C4: BESS cannot charge and discharge simultaneously (binary mutex) ---
    def ch_lim(mdl, t):  return mdl.P_ch[t]  <= bess_pow * (1 - mdl.u[t])
    def dis_lim(mdl, t): return mdl.P_dis[t] <= bess_pow * mdl.u[t]
    m.c_ch  = pyo.Constraint(m.T, rule=ch_lim)
    m.c_dis = pyo.Constraint(m.T, rule=dis_lim)

    # --- C5: Nuclear ramp rate (±100 MW/h, PWR load-following physical limit) ---
    if use_nuc:
        def nuc_up(mdl, t):
            if t == 0: return pyo.Constraint.Skip
            return mdl.P_nuc[t] - mdl.P_nuc[t-1] <= NUC_RAMP
        def nuc_dn(mdl, t):
            if t == 0: return pyo.Constraint.Skip
            return mdl.P_nuc[t-1] - mdl.P_nuc[t] <= NUC_RAMP
        m.c_nuc_up = pyo.Constraint(m.T, rule=nuc_up)
        m.c_nuc_dn = pyo.Constraint(m.T, rule=nuc_dn)

    # --- C6: CCGT Unit Commitment constraints ---
    # Both units start online (warm state at beginning of week)
    def ccgt_init(mdl, k):
        return mdl.u_ccgt[0, k] == 1
    m.c_ccgt_init = pyo.Constraint(m.K, rule=ccgt_init)

    # Output bounded by commitment status
    def ccgt_max(mdl, t, k):
        return mdl.x_ccgt[t, k] <= CCGT_UNIT_MW * mdl.u_ccgt[t, k]
    def ccgt_min(mdl, t, k):
        return mdl.x_ccgt[t, k] >= CCGT_MIN_MW * mdl.u_ccgt[t, k]
    m.c_ccgt_max = pyo.Constraint(m.T, m.K, rule=ccgt_max)
    m.c_ccgt_min = pyo.Constraint(m.T, m.K, rule=ccgt_min)

    # Startup indicator: y[t,k]=1 when u transitions 0→1
    def ccgt_startup(mdl, t, k):
        if t == 0: return pyo.Constraint.Skip
        return mdl.y_ccgt[t, k] >= mdl.u_ccgt[t, k] - mdl.u_ccgt[t-1, k]
    m.c_ccgt_startup = pyo.Constraint(m.T, m.K, rule=ccgt_startup)

    # Minimum up-time: once started, must stay on for CCGT_UP_MIN hours
    def ccgt_minup(mdl, t, k):
        if t == 0: return pyo.Constraint.Skip
        end = min(t + CCGT_UP_MIN - 1, T - 1)
        return sum(mdl.u_ccgt[tau, k] for tau in range(t, end + 1)) >= \
               CCGT_UP_MIN * mdl.y_ccgt[t, k]
    m.c_ccgt_minup = pyo.Constraint(m.T, m.K, rule=ccgt_minup)

    # Minimum down-time: once shut down, must stay off for CCGT_DN_MIN hours
    def ccgt_mindn(mdl, t, k):
        if t == 0: return pyo.Constraint.Skip
        end = min(t + CCGT_DN_MIN - 1, T - 1)
        # shutdown indicator: 1 - u[t,k] >= u[t-1,k] - u[t,k]
        return sum((1 - mdl.u_ccgt[tau, k]) for tau in range(t, end + 1)) >= \
               CCGT_DN_MIN * (mdl.u_ccgt[t-1, k] - mdl.u_ccgt[t, k])
    m.c_ccgt_mindn = pyo.Constraint(m.T, m.K, rule=ccgt_mindn)

    # CCGT ramp rate (per unit, ±800 MW/h when committed)
    def ccgt_up(mdl, t, k):
        if t == 0: return pyo.Constraint.Skip
        # Allow large ramp on startup (from 0 to min load)
        return mdl.x_ccgt[t, k] - mdl.x_ccgt[t-1, k] <= \
               CCGT_RAMP * mdl.u_ccgt[t-1, k] + CCGT_UNIT_MW * mdl.y_ccgt[t, k]
    def ccgt_dn(mdl, t, k):
        if t == 0: return pyo.Constraint.Skip
        return mdl.x_ccgt[t-1, k] - mdl.x_ccgt[t, k] <= CCGT_RAMP
    m.c_ccgt_up = pyo.Constraint(m.T, m.K, rule=ccgt_up)
    m.c_ccgt_dn = pyo.Constraint(m.T, m.K, rule=ccgt_dn)

    # --- C7: Peaker ramp rate (±5,000 MW/h — effectively unconstrained) ---
    def peak_up(mdl, t):
        if t == 0: return pyo.Constraint.Skip
        return mdl.P_peak[t] - mdl.P_peak[t-1] <= PEAK_RAMP
    def peak_dn(mdl, t):
        if t == 0: return pyo.Constraint.Skip
        return mdl.P_peak[t-1] - mdl.P_peak[t] <= PEAK_RAMP
    m.c_peak_up = pyo.Constraint(m.T, rule=peak_up)
    m.c_peak_dn = pyo.Constraint(m.T, rule=peak_dn)

    # --- Solve ---
    solver = pyo.SolverFactory(SOLVER)
    solver.options['time_limit'] = 300
    t0 = time.time()
    result = solver.solve(m, tee=False, load_solutions=False)
    elapsed = time.time() - t0
    status = str(result.solver.termination_condition)

    metrics = {
        'status': status, 'solve_time_s': elapsed,
        'n_vars': m.nvariables(), 'n_constrs': m.nconstraints(),
        'n_binary': T,  # one u[t] per hour
    }

    if status in ('optimal', 'feasible'):
        m.solutions.load_from(result)
        def val(v, t): return pyo.value(v[t]) or 0.0

        P_nuc  = [val(m.P_nuc, t) for t in range(T)]
        P_ccgt = [sum(pyo.value(m.x_ccgt[t, k]) or 0.0 for k in range(N_CCGT))
                  for t in range(T)]
        P_peak = [val(m.P_peak, t) for t in range(T)]
        n_startups = sum(pyo.value(m.y_ccgt[t, k]) or 0
                         for t in range(T) for k in range(N_CCGT))
        P_ch   = [val(m.P_ch, t)   for t in range(T)]
        P_dis  = [val(m.P_dis, t)  for t in range(T)]
        SoC    = [val(m.SoC, t)    for t in range(T)]
        P_imp  = [val(m.P_imp, t)  for t in range(T)]
        P_exp  = [val(m.P_exp, t)  for t in range(T)]
        P_shed = [val(m.P_shed, t) for t in range(T)]
        P_curt = [val(m.P_curt, t) for t in range(T)]

        obj_val = pyo.value(m.obj)
        nl_sum  = float(np.sum(nl_profile))

        dispatch = {
            'P_nuc': P_nuc, 'P_ccgt': P_ccgt, 'P_peak': P_peak,
            'P_ch': P_ch, 'P_dis': P_dis, 'SoC': SoC,
            'P_imp': P_imp, 'P_exp': P_exp,
            'P_shed': P_shed, 'P_curt': P_curt,
            'net_load': nl_profile.tolist() if hasattr(nl_profile, 'tolist') else list(nl_profile),
            'total_cost': obj_val,
            'cost_per_mwh': obj_val / nl_sum if nl_sum > 0 else 0,
            'avg_nuc': np.mean(P_nuc),
            'avg_ccgt': np.mean(P_ccgt),
            'avg_peak': np.mean(P_peak),
            'avg_gas': np.mean(P_ccgt) + np.mean(P_peak),
            'n_startups': n_startups,
            'avg_imp': np.mean(P_imp),
            'shed_total': np.sum(P_shed),
            'curt_total': np.sum(P_curt),
            'bess_cycles': np.sum(P_dis) / bess_ene if bess_ene > 0 else 0,
            'soc_end': SoC[-1],
        }
        metrics['obj'] = obj_val
        metrics['n_binary_total'] = T + N_CCGT * T * 2  # BESS u(t) + CCGT u,y
        metrics['gap_pct'] = 0.0
        try:
            lb = result.problem[0].lower_bound
            ub = result.problem[0].upper_bound
            if lb and ub and ub > 0:
                metrics['gap_pct'] = (ub - lb) / ub * 100
        except: pass
    else:
        dispatch = None
        metrics['obj'] = float('inf')
        metrics['gap_pct'] = float('nan')

    return metrics, dispatch


# =========================================================================
#  6 CORE SCENARIOS (same definitions as 05b)
# =========================================================================
print("\n[2/4] Running 6 core scenarios...")

scenarios = {
    'S1_Deterministic': {
        'nl': week['net_load_predicted'].values, 'use_nuc': True,
        'bess_pow': BESS_POW, 'bess_ene': BESS_ENE, 'desc': 'Point prediction'},
    'S2_Worst_Case': {
        'nl': week['net_load_Q90'].values, 'use_nuc': True,
        'bess_pow': BESS_POW, 'bess_ene': BESS_ENE, 'desc': 'CQR Q90 upper bound'},
    'S3_Best_Case': {
        'nl': week['net_load_Q10'].values, 'use_nuc': True,
        'bess_pow': BESS_POW, 'bess_ene': BESS_ENE, 'desc': 'CQR Q10 lower bound'},
    'S4_No_Nuclear': {
        'nl': week['net_load_predicted'].values, 'use_nuc': False,
        'bess_pow': BESS_POW, 'bess_ene': BESS_ENE, 'desc': 'Nuclear removed'},
    'S5_Small_BESS': {
        'nl': week['net_load_predicted'].values, 'use_nuc': True,
        'bess_pow': 1000, 'bess_ene': 4000, 'desc': 'BESS 1 GW / 4 GWh'},
    'S6_Robust': {
        'nl': 0.5*week['net_load_predicted'].values + 0.5*week['net_load_Q90'].values,
        'use_nuc': True, 'bess_pow': BESS_POW, 'bess_ene': BESS_ENE,
        'desc': '50% point + 50% Q90 (γ=0.5)'},
}

sm_all, disp_all = {}, {}
for sname, cfg in scenarios.items():
    sm, disp = build_and_solve(cfg['nl'], sname, use_nuc=cfg['use_nuc'],
                               bess_pow=cfg['bess_pow'], bess_ene=cfg['bess_ene'])
    sm_all[sname] = sm
    disp_all[sname] = disp
    status_str = sm['status']
    if disp:
        print(f"  {sname:<22} ${disp['cost_per_mwh']:5.2f}/MWh  "
              f"shed={disp['shed_total']:6.0f} MWh  "
              f"starts={disp.get('n_startups',0):2.0f}  "
              f"SoCend={disp['soc_end']/BESS_ENE*100:.0f}%  "
              f"t={sm['solve_time_s']:.2f}s  gap={sm['gap_pct']:.4f}%")
    else:
        print(f"  {sname:<22} INFEASIBLE ({status_str})")


# Nuclear cost premium (S4 vs S1)
s1 = disp_all.get('S1_Deterministic')
s4 = disp_all.get('S4_No_Nuclear')
if s1 and s4:
    pct = (s4['cost_per_mwh'] - s1['cost_per_mwh']) / s1['cost_per_mwh'] * 100
    print(f"\n  Nuclear cost premium (S4 vs S1): "
          f"${s1['cost_per_mwh']:.2f} → ${s4['cost_per_mwh']:.2f}/MWh (+{pct:.1f}%)")


# =========================================================================
#  NUCLEAR SENSITIVITY SWEEP (0 → 4,000 MW)
# =========================================================================
print("\n[3/4] Nuclear capacity sensitivity sweep...")
nl_det = week['net_load_predicted'].values
nuc_rows = []
for cap in [0, 500, 1000, 1500, 1800, 2256, 3000, 4000]:
    n_min = max(0, int(cap * 0.80))
    sm, d = build_and_solve(nl_det, f"Nuc_{cap}", nuc_cap=cap, nuc_min=n_min,
                            use_nuc=(cap > 0))
    if d:
        nuc_rows.append({'Nuclear_MW': cap, 'Cost_MWh': d['cost_per_mwh'],
                         'Avg_CCGT': d['avg_ccgt'], 'Avg_Peak': d['avg_peak'],
                         'Avg_Gas': d['avg_gas'], 'Avg_Import': d['avg_imp'],
                         'Shed_MWh': d['shed_total'],
                         'Solve_Time_s': sm['solve_time_s'], 'Gap_Pct': sm['gap_pct']})
        print(f"  {cap:5} MW → ${d['cost_per_mwh']:.2f}/MWh  "
              f"CCGT={d['avg_ccgt']:.0f} Peak={d['avg_peak']:.0f}")
    else:
        print(f"  {cap:5} MW → INFEASIBLE")

nuc_df = pd.DataFrame(nuc_rows)
nuc_df.to_csv(os.path.join(TBL, "sensitivity_nuclear.csv"), index=False)
print("  -> sensitivity_nuclear.csv")


# =========================================================================
#  SAVE SCENARIO COMPARISON TABLE
# =========================================================================
print("\n[4/4] Saving tables and figures...")

# Solver quality
sq_rows = []
for sname, sm in sm_all.items():
    d = disp_all[sname]
    sq_rows.append({'Scenario': sname, 'Status': sm['status'],
                    'Solve_Time_s': round(sm['solve_time_s'], 3),
                    'Objective': round(sm['obj'], 0),
                    'Gap_Pct': round(sm['gap_pct'], 4),
                    'Variables': sm['n_vars'],
                    'Constraints': sm['n_constrs'],
                    'Binary_Vars': sm['n_binary'],
                    'Cost_MWh': round(d['cost_per_mwh'], 2) if d else None,
                    'Shed_MWh': round(d['shed_total'], 0) if d else None,
                    'SoC_End_pct': round(d['soc_end']/BESS_ENE*100, 1) if d else None})
sq_df = pd.DataFrame(sq_rows)
sq_df.to_csv(os.path.join(TBL, "solver_quality.csv"), index=False)

# Scenario comparison (extended with CCGT/Peaker split)
comp_rows = []
for sname, d in disp_all.items():
    if d:
        comp_rows.append({'Scenario': sname, 'Cost_MWh': round(d['cost_per_mwh'], 2),
                          'Avg_Nuclear_MW': round(d['avg_nuc'], 0),
                          'Avg_CCGT_MW': round(d['avg_ccgt'], 0),
                          'Avg_Peaker_MW': round(d['avg_peak'], 0),
                          'Avg_Gas_Total_MW': round(d['avg_gas'], 0),
                          'Avg_Import_MW': round(d['avg_imp'], 0),
                          'Shed_MWh': round(d['shed_total'], 0),
                          'Curt_MWh': round(d['curt_total'], 0),
                          'BESS_Cycles': round(d['bess_cycles'], 2),
                          'SoC_End_pct': round(d['soc_end']/BESS_ENE*100, 1)})
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(TBL, "scenario_comparison_improved.csv"), index=False)

# Print full table
print("\n  SCENARIO COMPARISON:")
print(f"  {'Scenario':<22} {'$/MWh':>7} {'CCGT':>6} {'Peak':>6} {'Imp':>6} {'Shed':>8} {'SoC%':>6}")
print("  " + "-"*70)
for _, row in comp_df.iterrows():
    print(f"  {row['Scenario']:<22} {row['Cost_MWh']:>7.2f} "
          f"{row['Avg_CCGT_MW']:>6.0f} {row['Avg_Peaker_MW']:>6.0f} "
          f"{row['Avg_Import_MW']:>6.0f} {row['Shed_MWh']:>8.0f} "
          f"{row['SoC_End_pct']:>6.1f}")


# Nuclear premium summary
if len(nuc_df) > 1:
    row_0   = nuc_df[nuc_df['Nuclear_MW']==0].iloc[0]
    row_cur = nuc_df[nuc_df['Nuclear_MW']==2256].iloc[0]
    prem = (row_0['Cost_MWh'] - row_cur['Cost_MWh']) / row_cur['Cost_MWh'] * 100
    print(f"\n  Nuclear sensitivity: 0 MW → ${row_0['Cost_MWh']:.2f}/MWh, "
          f"2256 MW → ${row_cur['Cost_MWh']:.2f}/MWh (+{prem:.1f}% removal premium)")


# =========================================================================
#  FIGURE: Dispatch Stack (S1)
# =========================================================================
if disp_all.get('S1_Deterministic'):
    d = disp_all['S1_Deterministic']
    t = np.arange(HORIZON)
    fig2, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    ax = axes[0]
    nuc  = np.array(d['P_nuc'])
    ccgt = np.array(d['P_ccgt'])
    peak = np.array(d['P_peak'])
    dis  = np.array(d['P_dis'])
    imp  = np.array(d['P_imp'])
    base = np.zeros(HORIZON)
    ax.fill_between(t, base, nuc,           alpha=0.85, color='#4CAF50', label='Nuclear')
    ax.fill_between(t, nuc, nuc+ccgt,       alpha=0.75, color='#FF9800', label='CCGT')
    ax.fill_between(t, nuc+ccgt, nuc+ccgt+peak, alpha=0.70, color='#FFC107', label='Peaker')
    ax.fill_between(t, nuc+ccgt+peak, nuc+ccgt+peak+dis, alpha=0.70, color='#2196F3', label='BESS Dis')
    ax.fill_between(t, nuc+ccgt+peak+dis, nuc+ccgt+peak+dis+imp, alpha=0.50, color='#9C27B0', label='Import')
    ax.plot(t, d['net_load'], 'k-', lw=2, label='Net Load')
    ax.set_ylabel('Power (MW)'); ax.legend(ncol=6, loc='upper right')
    ax.set_title('Optimal Dispatch: Two-Tier Gas Stack (S1 Deterministic)', fontweight='bold')
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ch = np.array(d['P_ch'])
    ax.bar(t, dis, width=0.9, alpha=0.7, color='#4CAF50', label='Discharge')
    ax.bar(t, -ch, width=0.9, alpha=0.7, color='#E53935', label='Charge')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel('BESS Power (MW)')
    ax.set_title('BESS Charge / Discharge', fontweight='bold')
    ax2 = ax.twinx()
    ax2.plot(t, np.array(d['SoC'])/BESS_ENE*100, 'b-', lw=2, label='SoC (%)')
    ax2.axhline(SOC_END*100, color='red', lw=1, ls='--', alpha=0.5, label=f'SoC floor ({SOC_END*100:.0f}%)')
    ax2.set_ylabel('SoC (%)', color='blue'); ax2.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, ncol=4, loc='upper right')

    ax = axes[2]
    c_nuc  = NUC_COST   * nuc
    c_ccgt = CCGT_COST  * ccgt
    c_peak = PEAK_COST  * peak
    c_bess = BESS_DEG   * (ch + dis)
    c_imp  = IMP_COST   * imp
    base = np.zeros(HORIZON)
    ax.fill_between(t, base,                           (base+c_nuc)/1e3,            color='#4CAF50', alpha=0.85, label='Nuclear')
    ax.fill_between(t, c_nuc/1e3,                     (c_nuc+c_ccgt)/1e3,          color='#FF9800', alpha=0.75, label='CCGT')
    ax.fill_between(t, (c_nuc+c_ccgt)/1e3,            (c_nuc+c_ccgt+c_peak)/1e3,   color='#FFC107', alpha=0.70, label='Peaker')
    ax.fill_between(t, (c_nuc+c_ccgt+c_peak)/1e3,     (c_nuc+c_ccgt+c_peak+c_bess)/1e3, color='#2196F3', alpha=0.70, label='BESS')
    ax.fill_between(t, (c_nuc+c_ccgt+c_peak+c_bess)/1e3, (c_nuc+c_ccgt+c_peak+c_bess+c_imp)/1e3, color='#9C27B0', alpha=0.50, label='Import')
    ax.set_xlabel('Hour'); ax.set_ylabel('Hourly Cost ($k)')
    ax.set_title('Hourly System Cost Breakdown', fontweight='bold')
    ax.legend(ncol=5, loc='upper right')

    plt.tight_layout()
    fig2.savefig(os.path.join(FIG, "fig13_dispatch_stack.png"))
    plt.close()
    print("  -> fig13_dispatch_stack.png (updated: two-tier gas)")


# Nuclear sensitivity figure
if len(nuc_df) > 0:
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nuc_df['Nuclear_MW'], nuc_df['Cost_MWh'], 'o-', color='#4CAF50', lw=2, ms=8)
    ax.axvline(NUC_CAP, color='red', ls='--', alpha=0.6, label=f'Current Diablo Canyon ({NUC_CAP} MW)')
    ax.set_xlabel('Nuclear Capacity (MW)')
    ax.set_ylabel('System Cost ($/MWh)')
    ax.set_title('Nuclear Capacity Sensitivity\n(Two-Tier Gas + End-SoC Constraint)', fontweight='bold')
    ax.legend()

    # Annotate premium
    if len(nuc_df) >= 2:
        r0 = nuc_df[nuc_df['Nuclear_MW']==0].iloc[0]
        rc = nuc_df[nuc_df['Nuclear_MW']==2256].iloc[0]
        ax.annotate(f"+{(r0['Cost_MWh']-rc['Cost_MWh'])/rc['Cost_MWh']*100:.1f}%\nremoval cost",
                    xy=(0, r0['Cost_MWh']), xytext=(500, r0['Cost_MWh']-0.5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    fig2.savefig(os.path.join(FIG, "fig_nuclear_sensitivity.png"))
    plt.close()
    print("  -> fig_nuclear_sensitivity.png")


print("\n" + "="*78)
print("  IMPROVED MILP COMPLETE")
n_bin = HORIZON + N_CCGT * HORIZON * 2
print(f"  Model size: ~{HORIZON*13} vars ({n_bin} binary: {HORIZON} BESS + {N_CCGT*HORIZON*2} CCGT UC)")
print(f"  Improvements: two-tier gas with UC (startup ${CCGT_STARTUP:,}/event, min {CCGT_UP_MIN}h up/{CCGT_DN_MIN}h down)")
print(f"               end-SoC≥{SOC_END*100:.0f}% constraint, nuclear ramp {NUC_RAMP} MW/h")
print("="*78)
