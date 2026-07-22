"""
=============================================================================
 PHASE 3 (ENHANCED): MILP OPTIMIZATION + SENSITIVITY ANALYSIS
 Nuclear Baseload + BESS Dispatch — Solver Metrics & Parametric Sweep
=============================================================================
"""

import os
import sys
import io
import time
import warnings
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    try:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'highspy', '-q'])
        SOLVER = 'appsi_highs'
    except:
        SOLVER = 'glpk'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR  = os.path.join(PROJECT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "outputs", "tables")
for d in [FIG_DIR, TABLE_DIR]:
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
print("  PHASE 3 (ENHANCED): MILP + SENSITIVITY ANALYSIS")
print("=" * 78)

# =====================================================================
#  SYSTEM PARAMETERS (baseline)
# =====================================================================
NUCLEAR_CAPACITY_MW = 2256
NUCLEAR_MIN_MW = 1800
NUCLEAR_RAMP_RATE = 100
NUCLEAR_MARGINAL_COST = 12

GAS_CAPACITY_MW = 20000
GAS_RAMP_RATE = 5000
GAS_MARGINAL_COST = 45

BESS_POWER_MW = 5000
BESS_ENERGY_MWH = 20000
BESS_EFFICIENCY = 0.90
BESS_SOC_MIN = 0.10
BESS_SOC_MAX = 0.90
BESS_CYCLING_COST = 5
BESS_INITIAL_SOC = 0.50

CURTAILMENT_COST = 10
VOLL = 10000
IMPORT_COST = 55
EXPORT_REVENUE = 20

# =====================================================================
#  LOAD UPDATED PREDICTIONS
# =====================================================================
print("\n[1/5] Loading updated (leakage-free) predictions...")

preds = pd.read_csv(os.path.join(DATA_DIR, "ml_predictions_for_milp.csv"))
preds['timestamp'] = pd.to_datetime(preds['timestamp'])

HORIZON = 168
preds_week = preds.head(HORIZON).copy()
print(f"  Loaded {len(preds)} predictions, using {HORIZON}h horizon")
print(f"  Period: {preds_week['timestamp'].iloc[0]} -> {preds_week['timestamp'].iloc[-1]}")

# =====================================================================
#  MILP MODEL BUILDER (with solver metrics)
# =====================================================================
def build_and_solve(net_load_profile, price_profile, scenario_name,
                    nuclear_cap=NUCLEAR_CAPACITY_MW, nuclear_min=NUCLEAR_MIN_MW,
                    bess_power=BESS_POWER_MW, bess_energy=BESS_ENERGY_MWH,
                    use_nuclear=True, time_limit=300):
    T = len(net_load_profile)
    model = pyo.ConcreteModel(name=f"Dispatch_{scenario_name}")
    model.T = pyo.RangeSet(0, T-1)

    if use_nuclear:
        model.P_nuc = pyo.Var(model.T, bounds=(nuclear_min, nuclear_cap), initialize=nuclear_cap*0.9)
    else:
        model.P_nuc = pyo.Var(model.T, bounds=(0, 0), initialize=0)

    model.P_gas = pyo.Var(model.T, bounds=(0, GAS_CAPACITY_MW), initialize=10000)
    model.P_bess_ch = pyo.Var(model.T, bounds=(0, bess_power), initialize=0)
    model.P_bess_dis = pyo.Var(model.T, bounds=(0, bess_power), initialize=0)
    model.SoC = pyo.Var(model.T, bounds=(BESS_SOC_MIN * bess_energy, BESS_SOC_MAX * bess_energy))
    model.P_import = pyo.Var(model.T, bounds=(0, 10000), initialize=0)
    model.P_export = pyo.Var(model.T, bounds=(0, 6000), initialize=0)
    model.P_shed = pyo.Var(model.T, bounds=(0, 5000), initialize=0)
    model.P_curt = pyo.Var(model.T, bounds=(0, 15000), initialize=0)
    model.u_bess = pyo.Var(model.T, within=pyo.Binary, initialize=0)

    def obj_rule(m):
        return sum(
            NUCLEAR_MARGINAL_COST * m.P_nuc[t]
            + GAS_MARGINAL_COST * m.P_gas[t]
            + BESS_CYCLING_COST * (m.P_bess_ch[t] + m.P_bess_dis[t])
            + IMPORT_COST * m.P_import[t]
            - EXPORT_REVENUE * m.P_export[t]
            + VOLL * m.P_shed[t]
            + CURTAILMENT_COST * m.P_curt[t]
            for t in m.T
        )
    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def power_balance_rule(m, t):
        supply = m.P_nuc[t] + m.P_gas[t] + m.P_bess_dis[t] + m.P_import[t] + m.P_shed[t]
        demand_side = net_load_profile[t] + m.P_bess_ch[t] + m.P_export[t] + m.P_curt[t]
        return supply == demand_side
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    eta = np.sqrt(BESS_EFFICIENCY)

    def soc_init_rule(m):
        return m.SoC[0] == BESS_INITIAL_SOC * bess_energy
    model.soc_init = pyo.Constraint(rule=soc_init_rule)

    def soc_dynamics_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.SoC[t] == m.SoC[t-1] + eta * m.P_bess_ch[t] - (1/eta) * m.P_bess_dis[t]
    model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    def bess_ch_limit(m, t):
        return m.P_bess_ch[t] <= bess_power * (1 - m.u_bess[t])
    model.bess_ch_lim = pyo.Constraint(model.T, rule=bess_ch_limit)

    def bess_dis_limit(m, t):
        return m.P_bess_dis[t] <= bess_power * m.u_bess[t]
    model.bess_dis_lim = pyo.Constraint(model.T, rule=bess_dis_limit)

    if use_nuclear:
        def nuc_ramp_up(m, t):
            if t == 0: return pyo.Constraint.Skip
            return m.P_nuc[t] - m.P_nuc[t-1] <= NUCLEAR_RAMP_RATE
        model.nuc_ramp_up = pyo.Constraint(model.T, rule=nuc_ramp_up)

        def nuc_ramp_down(m, t):
            if t == 0: return pyo.Constraint.Skip
            return m.P_nuc[t-1] - m.P_nuc[t] <= NUCLEAR_RAMP_RATE
        model.nuc_ramp_dn = pyo.Constraint(model.T, rule=nuc_ramp_down)

    def gas_ramp_up(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.P_gas[t] - m.P_gas[t-1] <= GAS_RAMP_RATE
    model.gas_ramp_up = pyo.Constraint(model.T, rule=gas_ramp_up)

    def gas_ramp_down(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.P_gas[t-1] - m.P_gas[t] <= GAS_RAMP_RATE
    model.gas_ramp_dn = pyo.Constraint(model.T, rule=gas_ramp_down)

    # Solve with timing
    solver = pyo.SolverFactory(SOLVER)
    solver.options['time_limit'] = time_limit

    t0 = time.time()
    result = solver.solve(model, tee=False)
    solve_time = time.time() - t0

    status = result.solver.termination_condition

    # Extract solver quality metrics
    solver_metrics = {
        'status': str(status),
        'solve_time_s': solve_time,
        'n_variables': model.nvariables(),
        'n_constraints': model.nconstraints(),
        'n_binary_vars': sum(1 for v in model.component_data_objects(pyo.Var) if v.domain == pyo.Binary),
    }

    try:
        solver_metrics['objective_value'] = pyo.value(model.objective)
    except:
        solver_metrics['objective_value'] = float('inf')

    try:
        solver_metrics['lower_bound'] = result.problem[0].lower_bound
        solver_metrics['upper_bound'] = result.problem[0].upper_bound
        if solver_metrics['lower_bound'] > 0 and solver_metrics['upper_bound'] < float('inf'):
            solver_metrics['gap_pct'] = (solver_metrics['upper_bound'] - solver_metrics['lower_bound']) / solver_metrics['upper_bound'] * 100
        else:
            solver_metrics['gap_pct'] = 0.0
    except:
        solver_metrics['lower_bound'] = solver_metrics['objective_value']
        solver_metrics['upper_bound'] = solver_metrics['objective_value']
        solver_metrics['gap_pct'] = 0.0

    dispatch = None
    if str(status) in ['optimal', 'feasible']:
        T_h = len(net_load_profile)
        dispatch = {
            'P_nuc': [pyo.value(model.P_nuc[t]) for t in range(T_h)],
            'P_gas': [pyo.value(model.P_gas[t]) for t in range(T_h)],
            'P_bess_ch': [pyo.value(model.P_bess_ch[t]) for t in range(T_h)],
            'P_bess_dis': [pyo.value(model.P_bess_dis[t]) for t in range(T_h)],
            'SoC': [pyo.value(model.SoC[t]) for t in range(T_h)],
            'P_import': [pyo.value(model.P_import[t]) for t in range(T_h)],
            'P_export': [pyo.value(model.P_export[t]) for t in range(T_h)],
            'P_shed': [pyo.value(model.P_shed[t]) for t in range(T_h)],
            'P_curt': [pyo.value(model.P_curt[t]) for t in range(T_h)],
            'net_load': net_load_profile.tolist() if hasattr(net_load_profile, 'tolist') else list(net_load_profile),
        }
        dispatch['total_cost'] = solver_metrics['objective_value']
        dispatch['avg_nuclear'] = np.mean(dispatch['P_nuc'])
        dispatch['avg_gas'] = np.mean(dispatch['P_gas'])
        dispatch['avg_import'] = np.mean(dispatch['P_import'])
        dispatch['total_shed'] = np.sum(dispatch['P_shed'])
        dispatch['total_curt'] = np.sum(dispatch['P_curt'])
        bess_e = bess_energy if bess_energy > 0 else 1
        dispatch['bess_cycles'] = np.sum(dispatch['P_bess_dis']) / bess_e
        dispatch['cost_per_mwh'] = dispatch['total_cost'] / np.sum(net_load_profile) if np.sum(net_load_profile) > 0 else 0

    return solver_metrics, dispatch

# =====================================================================
#  RUN 6 CORE SCENARIOS
# =====================================================================
print("\n[2/5] Running 6 core scenarios with solver metrics...")

scenarios = {
    'S1_Deterministic': {
        'net_load': preds_week['net_load_predicted'].values,
        'price': preds_week['price_predicted'].values,
        'desc': 'Point prediction (deterministic)',
        'use_nuclear': True, 'bess_power': BESS_POWER_MW, 'bess_energy': BESS_ENERGY_MWH,
    },
    'S2_Worst_Case': {
        'net_load': preds_week['net_load_Q90'].values,
        'price': preds_week['price_Q90'].values,
        'desc': 'Upper bound (Q90)',
        'use_nuclear': True, 'bess_power': BESS_POWER_MW, 'bess_energy': BESS_ENERGY_MWH,
    },
    'S3_Best_Case': {
        'net_load': preds_week['net_load_Q10'].values,
        'price': preds_week['price_Q10'].values,
        'desc': 'Lower bound (Q10)',
        'use_nuclear': True, 'bess_power': BESS_POWER_MW, 'bess_energy': BESS_ENERGY_MWH,
    },
    'S4_No_Nuclear': {
        'net_load': preds_week['net_load_predicted'].values,
        'price': preds_week['price_predicted'].values,
        'desc': 'Without nuclear baseload',
        'use_nuclear': False, 'bess_power': BESS_POWER_MW, 'bess_energy': BESS_ENERGY_MWH,
    },
    'S5_Small_BESS': {
        'net_load': preds_week['net_load_predicted'].values,
        'price': preds_week['price_predicted'].values,
        'desc': 'Undersized BESS (1000 MW / 4000 MWh)',
        'use_nuclear': True, 'bess_power': 1000, 'bess_energy': 4000,
    },
    'S6_Robust': {
        'net_load': preds_week['net_load_Q90'].values * 0.5 + preds_week['net_load_predicted'].values * 0.5,
        'price': preds_week['price_Q90'].values * 0.5 + preds_week['price_predicted'].values * 0.5,
        'desc': 'Robust (50% point + 50% worst case)',
        'use_nuclear': True, 'bess_power': BESS_POWER_MW, 'bess_energy': BESS_ENERGY_MWH,
    },
}

all_solver_metrics = {}
all_dispatch = {}

for name, cfg in scenarios.items():
    print(f"\n  --- {name}: {cfg['desc']} ---")

    sm, dispatch = build_and_solve(
        net_load_profile=cfg['net_load'], price_profile=cfg['price'],
        scenario_name=name, use_nuclear=cfg['use_nuclear'],
        bess_power=cfg['bess_power'], bess_energy=cfg['bess_energy'],
    )

    all_solver_metrics[name] = sm
    all_dispatch[name] = dispatch

    print(f"    Status:      {sm['status']}")
    print(f"    Solve time:  {sm['solve_time_s']:.2f}s")
    print(f"    Variables:   {sm['n_variables']}")
    print(f"    Constraints: {sm['n_constraints']}")
    print(f"    Gap:         {sm['gap_pct']:.4f}%")
    if dispatch:
        print(f"    Total cost:  ${dispatch['total_cost']:>15,.0f}")
        print(f"    Cost/MWh:    ${dispatch['cost_per_mwh']:>15.2f}")

# =====================================================================
#  SENSITIVITY ANALYSIS: BESS SIZE SWEEP
# =====================================================================
print("\n" + "=" * 78)
print("[3/5] Sensitivity Analysis: BESS Size Parametric Sweep...")

bess_configs = [
    (500, 2000, '500 MW / 2h'),
    (1000, 4000, '1 GW / 4h'),
    (2000, 8000, '2 GW / 4h'),
    (3000, 12000, '3 GW / 4h'),
    (5000, 20000, '5 GW / 4h (baseline)'),
    (7500, 30000, '7.5 GW / 4h'),
    (10000, 40000, '10 GW / 4h'),
]

bess_sensitivity = []
net_load_det = preds_week['net_load_predicted'].values

for bess_mw, bess_mwh, label in bess_configs:
    print(f"  BESS {label}...", end=' ')
    sm, disp = build_and_solve(
        net_load_profile=net_load_det, price_profile=preds_week['price_predicted'].values,
        scenario_name=f"BESS_{bess_mw}", bess_power=bess_mw, bess_energy=bess_mwh,
    )
    if disp:
        bess_sensitivity.append({
            'BESS_MW': bess_mw, 'BESS_MWh': bess_mwh, 'Label': label,
            'Total_Cost': disp['total_cost'], 'Cost_MWh': disp['cost_per_mwh'],
            'BESS_Cycles': disp['bess_cycles'], 'Avg_Gas': disp['avg_gas'],
            'Avg_Import': disp['avg_import'], 'Total_Shed': disp['total_shed'],
            'Solve_Time': sm['solve_time_s'], 'Gap_Pct': sm['gap_pct'],
        })
        print(f"Cost: ${disp['cost_per_mwh']:.2f}/MWh, Time: {sm['solve_time_s']:.2f}s")
    else:
        print("INFEASIBLE")

bess_sens_df = pd.DataFrame(bess_sensitivity)

# =====================================================================
#  SENSITIVITY ANALYSIS: NUCLEAR CAPACITY SWEEP
# =====================================================================
print("\n[3b/5] Sensitivity Analysis: Nuclear Capacity Sweep...")

nuclear_configs = [0, 500, 1000, 1500, 1800, 2256, 3000, 4000]
nuclear_sensitivity = []

for nuc_cap in nuclear_configs:
    nuc_min = max(0, int(nuc_cap * 0.8))
    use_nuc = nuc_cap > 0
    label = f'{nuc_cap} MW'
    print(f"  Nuclear {label}...", end=' ')

    sm, disp = build_and_solve(
        net_load_profile=net_load_det, price_profile=preds_week['price_predicted'].values,
        scenario_name=f"Nuc_{nuc_cap}", nuclear_cap=nuc_cap, nuclear_min=nuc_min,
        use_nuclear=use_nuc,
    )
    if disp:
        nuclear_sensitivity.append({
            'Nuclear_MW': nuc_cap, 'Total_Cost': disp['total_cost'],
            'Cost_MWh': disp['cost_per_mwh'], 'Avg_Nuclear': disp['avg_nuclear'],
            'Avg_Gas': disp['avg_gas'], 'Avg_Import': disp['avg_import'],
            'Total_Shed': disp['total_shed'],
            'Solve_Time': sm['solve_time_s'], 'Gap_Pct': sm['gap_pct'],
        })
        print(f"Cost: ${disp['cost_per_mwh']:.2f}/MWh, Gas: {disp['avg_gas']:.0f} MW")
    else:
        print("INFEASIBLE")

nuc_sens_df = pd.DataFrame(nuclear_sensitivity)

# =====================================================================
#  SENSITIVITY: GAS PRICE SWEEP
# =====================================================================
print("\n[3c/5] Sensitivity Analysis: Gas Price Sweep...")

gas_prices = [25, 35, 45, 55, 65, 75, 100]
gas_sensitivity = []

for gp in gas_prices:
    print(f"  Gas price ${gp}/MWh...", end=' ')
    old_gc = GAS_MARGINAL_COST

    # Temporarily override gas cost — we need to rebuild with the modified cost
    # Instead of modifying global, pass via a wrapper
    # The cleanest way is to modify the objective, but since our function uses global,
    # we modify and restore
    import types
    # Create a local scope version
    exec_globals = globals().copy()
    exec_globals['GAS_MARGINAL_COST'] = gp

    T_local = len(net_load_det)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T_local-1)
    model.P_nuc = pyo.Var(model.T, bounds=(NUCLEAR_MIN_MW, NUCLEAR_CAPACITY_MW))
    model.P_gas = pyo.Var(model.T, bounds=(0, GAS_CAPACITY_MW))
    model.P_bess_ch = pyo.Var(model.T, bounds=(0, BESS_POWER_MW))
    model.P_bess_dis = pyo.Var(model.T, bounds=(0, BESS_POWER_MW))
    model.SoC = pyo.Var(model.T, bounds=(BESS_SOC_MIN * BESS_ENERGY_MWH, BESS_SOC_MAX * BESS_ENERGY_MWH))
    model.P_import = pyo.Var(model.T, bounds=(0, 10000))
    model.P_export = pyo.Var(model.T, bounds=(0, 6000))
    model.P_shed = pyo.Var(model.T, bounds=(0, 5000))
    model.P_curt = pyo.Var(model.T, bounds=(0, 15000))
    model.u_bess = pyo.Var(model.T, within=pyo.Binary)

    gas_cost_local = gp
    def obj_rule_gp(m):
        return sum(
            NUCLEAR_MARGINAL_COST * m.P_nuc[t]
            + gas_cost_local * m.P_gas[t]
            + BESS_CYCLING_COST * (m.P_bess_ch[t] + m.P_bess_dis[t])
            + IMPORT_COST * m.P_import[t]
            - EXPORT_REVENUE * m.P_export[t]
            + VOLL * m.P_shed[t]
            + CURTAILMENT_COST * m.P_curt[t]
            for t in m.T
        )
    model.objective = pyo.Objective(rule=obj_rule_gp, sense=pyo.minimize)

    nl = net_load_det
    def pb(m, t):
        return m.P_nuc[t] + m.P_gas[t] + m.P_bess_dis[t] + m.P_import[t] + m.P_shed[t] == \
               nl[t] + m.P_bess_ch[t] + m.P_export[t] + m.P_curt[t]
    model.pb = pyo.Constraint(model.T, rule=pb)

    eta_l = np.sqrt(BESS_EFFICIENCY)
    model.soc_init = pyo.Constraint(expr=model.SoC[0] == BESS_INITIAL_SOC * BESS_ENERGY_MWH)
    def soc_dyn(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.SoC[t] == m.SoC[t-1] + eta_l * m.P_bess_ch[t] - (1/eta_l) * m.P_bess_dis[t]
    model.soc_dyn = pyo.Constraint(model.T, rule=soc_dyn)

    def bch(m, t): return m.P_bess_ch[t] <= BESS_POWER_MW * (1 - m.u_bess[t])
    def bdis(m, t): return m.P_bess_dis[t] <= BESS_POWER_MW * m.u_bess[t]
    model.bch = pyo.Constraint(model.T, rule=bch)
    model.bdis = pyo.Constraint(model.T, rule=bdis)

    def nru(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.P_nuc[t] - m.P_nuc[t-1] <= NUCLEAR_RAMP_RATE
    def nrd(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.P_nuc[t-1] - m.P_nuc[t] <= NUCLEAR_RAMP_RATE
    model.nru = pyo.Constraint(model.T, rule=nru)
    model.nrd = pyo.Constraint(model.T, rule=nrd)

    def gru(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.P_gas[t] - m.P_gas[t-1] <= GAS_RAMP_RATE
    def grd(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.P_gas[t-1] - m.P_gas[t] <= GAS_RAMP_RATE
    model.gru = pyo.Constraint(model.T, rule=gru)
    model.grd = pyo.Constraint(model.T, rule=grd)

    solver = pyo.SolverFactory(SOLVER)
    solver.options['time_limit'] = 300
    t0 = time.time()
    result = solver.solve(model, tee=False)
    st = time.time() - t0

    status = str(result.solver.termination_condition)
    if status in ['optimal', 'feasible']:
        obj_val = pyo.value(model.objective)
        avg_gas = np.mean([pyo.value(model.P_gas[t]) for t in range(T_local)])
        avg_nuc = np.mean([pyo.value(model.P_nuc[t]) for t in range(T_local)])
        cost_mwh = obj_val / np.sum(nl)
        gas_sensitivity.append({
            'Gas_Price': gp, 'Total_Cost': obj_val, 'Cost_MWh': cost_mwh,
            'Avg_Gas': avg_gas, 'Avg_Nuclear': avg_nuc, 'Solve_Time': st,
        })
        print(f"Cost: ${cost_mwh:.2f}/MWh, Gas: {avg_gas:.0f} MW")
    else:
        print("INFEASIBLE")

gas_sens_df = pd.DataFrame(gas_sensitivity)

# =====================================================================
#  SAVE SOLVER QUALITY TABLE
# =====================================================================
print("\n" + "=" * 78)
print("[4/5] Saving solver quality and scenario tables...")

# Solver metrics table
solver_rows = []
for name, sm in all_solver_metrics.items():
    row = {
        'Scenario': name,
        'Status': sm['status'],
        'Solve_Time_s': sm['solve_time_s'],
        'Objective': sm['objective_value'],
        'Lower_Bound': sm.get('lower_bound', ''),
        'Upper_Bound': sm.get('upper_bound', ''),
        'Gap_Pct': sm.get('gap_pct', 0),
        'Variables': sm['n_variables'],
        'Constraints': sm['n_constraints'],
        'Binary_Vars': sm['n_binary_vars'],
    }
    solver_rows.append(row)

solver_df = pd.DataFrame(solver_rows)
solver_df.to_csv(os.path.join(TABLE_DIR, "solver_quality.csv"), index=False)
print("  -> solver_quality.csv")

# Scenario comparison (updated)
comp_rows = []
for name, disp in all_dispatch.items():
    if disp:
        comp_rows.append({
            'Scenario': name,
            'Total Cost ($)': disp['total_cost'],
            'Cost/MWh ($/MWh)': disp['cost_per_mwh'],
            'Avg Nuclear (MW)': disp['avg_nuclear'],
            'Avg Gas (MW)': disp['avg_gas'],
            'Avg Import (MW)': disp['avg_import'],
            'Load Shed (MWh)': disp['total_shed'],
            'Curtailment (MWh)': disp['total_curt'],
            'BESS Cycles': disp['bess_cycles'],
        })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(TABLE_DIR, "scenario_comparison.csv"), index=False)
print("  -> scenario_comparison.csv")

# Sensitivity tables
bess_sens_df.to_csv(os.path.join(TABLE_DIR, "sensitivity_bess.csv"), index=False)
nuc_sens_df.to_csv(os.path.join(TABLE_DIR, "sensitivity_nuclear.csv"), index=False)
gas_sens_df.to_csv(os.path.join(TABLE_DIR, "sensitivity_gas_price.csv"), index=False)
print("  -> sensitivity_bess.csv, sensitivity_nuclear.csv, sensitivity_gas_price.csv")

# =====================================================================
#  FIGURES
# =====================================================================
print("\n" + "=" * 78)
print("[5/5] Generating MILP figures...")

timestamps = preds_week['timestamp'].values

# --- Fig 13: Dispatch Stack (updated) ---
if 'S1_Deterministic' in all_dispatch and all_dispatch['S1_Deterministic']:
    res = all_dispatch['S1_Deterministic']
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    t = np.arange(HORIZON)
    nuc = np.array(res['P_nuc'])
    gas = np.array(res['P_gas'])
    bess_dis = np.array(res['P_bess_dis'])
    imp = np.array(res['P_import'])

    ax = axes[0]
    ax.fill_between(t, 0, nuc, alpha=0.8, color='#4CAF50', label='Nuclear')
    ax.fill_between(t, nuc, nuc+gas, alpha=0.7, color='#FF9800', label='Natural Gas')
    ax.fill_between(t, nuc+gas, nuc+gas+bess_dis, alpha=0.7, color='#2196F3', label='BESS Discharge')
    ax.fill_between(t, nuc+gas+bess_dis, nuc+gas+bess_dis+imp, alpha=0.5, color='#9C27B0', label='Import')
    ax.plot(t, res['net_load'], 'k-', linewidth=2, label='Net Load')
    ax.set_ylabel('Power (MW)')
    ax.set_title('Optimal Dispatch: Generation Stack (S1 Deterministic)', fontweight='bold')
    ax.legend(loc='upper right', ncol=5)
    ax.set_ylim(0)

    ax = axes[1]
    bess_ch = np.array(res['P_bess_ch'])
    soc = np.array(res['SoC'])
    ax.bar(t, bess_dis, width=0.9, alpha=0.7, color='#4CAF50', label='Discharge')
    ax.bar(t, -bess_ch, width=0.9, alpha=0.7, color='#E53935', label='Charge')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('BESS Power (MW)')
    ax.set_title('BESS Charge/Discharge Profile', fontweight='bold')
    ax2 = ax.twinx()
    ax2.plot(t, soc / BESS_ENERGY_MWH * 100, 'b-', linewidth=2, label='SoC (%)')
    ax2.set_ylabel('State of Charge (%)', color='blue')
    ax2.set_ylim(0, 100)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=3)

    ax = axes[2]
    cost_nuc = NUCLEAR_MARGINAL_COST * nuc
    cost_gas = GAS_MARGINAL_COST * gas
    cost_bess = BESS_CYCLING_COST * (bess_ch + bess_dis)
    cost_imp = IMPORT_COST * imp
    ax.fill_between(t, 0, cost_nuc/1000, alpha=0.8, color='#4CAF50', label='Nuclear')
    ax.fill_between(t, cost_nuc/1000, (cost_nuc+cost_gas)/1000, alpha=0.7, color='#FF9800', label='Gas')
    ax.fill_between(t, (cost_nuc+cost_gas)/1000, (cost_nuc+cost_gas+cost_bess)/1000, alpha=0.7, color='#2196F3', label='BESS')
    ax.fill_between(t, (cost_nuc+cost_gas+cost_bess)/1000, (cost_nuc+cost_gas+cost_bess+cost_imp)/1000, alpha=0.5, color='#9C27B0', label='Import')
    ax.set_ylabel('Hourly Cost ($k)')
    ax.set_xlabel('Hour')
    ax.set_title('Hourly System Cost Breakdown', fontweight='bold')
    ax.legend(loc='upper right', ncol=4)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig13_dispatch_stack.png"))
    plt.close()
    print("  -> fig13_dispatch_stack.png")

# --- Fig 14: Scenario Comparison (updated) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors_sc = ['#4CAF50', '#E53935', '#2196F3', '#FF9800', '#9C27B0', '#795548']

ax = axes[0]
bars = ax.bar(range(len(comp_df)), comp_df['Total Cost ($)'] / 1e6,
              color=colors_sc[:len(comp_df)], alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(comp_df)))
ax.set_xticklabels([s.replace('_', '\n') for s in comp_df['Scenario']], fontsize=8)
ax.set_ylabel('Total Cost (Million $)')
ax.set_title('Scenario Total Cost Comparison', fontweight='bold')
for bar, val in zip(bars, comp_df['Total Cost ($)'] / 1e6):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'${val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax = axes[1]
bars = ax.bar(range(len(comp_df)), comp_df['Cost/MWh ($/MWh)'],
              color=colors_sc[:len(comp_df)], alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(comp_df)))
ax.set_xticklabels([s.replace('_', '\n') for s in comp_df['Scenario']], fontsize=8)
ax.set_ylabel('Cost per MWh ($/MWh)')
ax.set_title('Scenario Unit Cost Comparison', fontweight='bold')
for bar, val in zip(bars, comp_df['Cost/MWh ($/MWh)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'${val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig14_scenario_comparison.png"))
plt.close()
print("  -> fig14_scenario_comparison.png")

# --- Fig 15: Nuclear Impact ---
if all_dispatch.get('S1_Deterministic') and all_dispatch.get('S4_No_Nuclear'):
    s1 = all_dispatch['S1_Deterministic']
    s4 = all_dispatch['S4_No_Nuclear']

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    t = np.arange(HORIZON)

    ax = axes[0]
    nuc = np.array(s1['P_nuc'])
    gas_s1 = np.array(s1['P_gas'])
    ax.fill_between(t, 0, nuc, alpha=0.8, color='#4CAF50', label='Nuclear')
    ax.fill_between(t, nuc, nuc+gas_s1, alpha=0.7, color='#FF9800', label='Gas')
    ax.plot(t, s1['net_load'], 'k-', linewidth=2, label='Net Load')
    ax.set_ylabel('Power (MW)')
    ax.set_title(f'WITH Nuclear (Cost: ${s1["total_cost"]:,.0f})', fontweight='bold', color='green')
    ax.legend(loc='upper right')
    ax.set_ylim(0)

    ax = axes[1]
    gas_s4 = np.array(s4['P_gas'])
    imp_s4 = np.array(s4['P_import'])
    ax.fill_between(t, 0, gas_s4, alpha=0.7, color='#FF9800', label='Gas')
    ax.fill_between(t, gas_s4, gas_s4+imp_s4, alpha=0.5, color='#9C27B0', label='Import')
    ax.plot(t, s4['net_load'], 'k-', linewidth=2, label='Net Load')
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Hour')
    ax.set_title(f'WITHOUT Nuclear (Cost: ${s4["total_cost"]:,.0f})', fontweight='bold', color='red')
    ax.legend(loc='upper right')
    ax.set_ylim(0)

    cost_diff = s4['total_cost'] - s1['total_cost']
    pct_diff = cost_diff / s1['total_cost'] * 100
    fig.suptitle(f'Nuclear Impact: Removing nuclear increases cost by ${cost_diff:,.0f} (+{pct_diff:.1f}%)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig15_nuclear_impact.png"))
    plt.close()
    print("  -> fig15_nuclear_impact.png")

# --- Fig 20: Sensitivity Analysis (3 subplots) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# BESS sensitivity
ax = axes[0]
ax.plot(bess_sens_df['BESS_MW'], bess_sens_df['Cost_MWh'], 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.set_xlabel('BESS Capacity (MW)')
ax.set_ylabel('System Cost ($/MWh)')
ax.set_title('BESS Size Sensitivity', fontweight='bold')
ax.axvline(x=BESS_POWER_MW, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({BESS_POWER_MW} MW)')
ax.legend()

# Nuclear sensitivity
ax = axes[1]
ax.plot(nuc_sens_df['Nuclear_MW'], nuc_sens_df['Cost_MWh'], 's-', color='#4CAF50', linewidth=2, markersize=8)
ax.set_xlabel('Nuclear Capacity (MW)')
ax.set_ylabel('System Cost ($/MWh)')
ax.set_title('Nuclear Capacity Sensitivity', fontweight='bold')
ax.axvline(x=NUCLEAR_CAPACITY_MW, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({NUCLEAR_CAPACITY_MW} MW)')
ax.legend()

# Gas price sensitivity
ax = axes[2]
ax.plot(gas_sens_df['Gas_Price'], gas_sens_df['Cost_MWh'], 'D-', color='#FF9800', linewidth=2, markersize=8)
ax.set_xlabel('Gas Marginal Cost ($/MWh)')
ax.set_ylabel('System Cost ($/MWh)')
ax.set_title('Gas Price Sensitivity', fontweight='bold')
ax.axvline(x=GAS_MARGINAL_COST, color='red', linestyle='--', alpha=0.5, label=f'Baseline (${GAS_MARGINAL_COST})')
ax.legend()

fig.suptitle('Parametric Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig20_sensitivity_analysis.png"))
plt.close()
print("  -> fig20_sensitivity_analysis.png")

# --- Fig 21: Solver Performance ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
names = solver_df['Scenario'].values
times = solver_df['Solve_Time_s'].values
colors_t = ['#4CAF50' if t < 1 else '#FF9800' if t < 5 else '#E53935' for t in times]
bars = ax.barh(range(len(names)), times, color=colors_t, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace('_', '\n') for n in names], fontsize=8)
ax.set_xlabel('Solve Time (seconds)')
ax.set_title('Solver Performance by Scenario', fontweight='bold')
for bar, t_val in zip(bars, times):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{t_val:.2f}s', va='center', fontsize=9)

ax = axes[1]
ax.axis('off')
table_data = []
for _, row in solver_df.iterrows():
    table_data.append([
        row['Scenario'].replace('_', ' '),
        row['Status'],
        f"{row['Solve_Time_s']:.2f}s",
        f"{row['Gap_Pct']:.4f}%",
        str(row['Variables']),
        str(row['Constraints']),
    ])
table = ax.table(
    cellText=table_data,
    colLabels=['Scenario', 'Status', 'Time', 'Gap%', 'Vars', 'Constrs'],
    loc='center', cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f0f0f0')
    cell.set_edgecolor('#cccccc')
ax.set_title('Solver Quality Metrics', fontweight='bold', pad=20)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig21_solver_performance.png"))
plt.close()
print("  -> fig21_solver_performance.png")

# =====================================================================
#  FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 78)
print("  PHASE 3 (ENHANCED) COMPLETE")
print("=" * 78)

print(f"\n  Scenario Results:")
print(f"  {'Scenario':<20} {'Cost':>12} {'$/MWh':>8} {'Time':>8} {'Gap':>8}")
print(f"  {'-'*60}")
for _, row in solver_df.iterrows():
    disp = all_dispatch.get(row['Scenario'])
    cost_str = f"${disp['cost_per_mwh']:.2f}" if disp else 'N/A'
    print(f"  {row['Scenario']:<20} {cost_str:>12} {row['Solve_Time_s']:>7.2f}s {row['Gap_Pct']:>7.4f}%")

print(f"\n  Sensitivity Analyses: 3 parametric sweeps completed")
print(f"    BESS:    {len(bess_sensitivity)} configurations")
print(f"    Nuclear: {len(nuclear_sensitivity)} configurations")
print(f"    Gas:     {len(gas_sensitivity)} configurations")

print(f"\n  New tables: solver_quality.csv, sensitivity_*.csv")
print(f"  New figures: fig20, fig21")
print("=" * 78)
