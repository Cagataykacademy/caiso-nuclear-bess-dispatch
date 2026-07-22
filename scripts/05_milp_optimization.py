"""
=============================================================================
 PHASE 3: ROBUST MILP OPTIMIZATION
 Nuclear Baseload + BESS Dispatch under Duck Curve Uncertainty
=============================================================================
 Formulation:
   - Objective: Minimize total system cost (generation + BESS cycling)
   - Decision Variables: 
       P_nuclear(t), P_gas(t), P_BESS_charge(t), P_BESS_discharge(t), SoC(t)
   - Uncertainty: Net load prediction intervals from CQR (Phase 2)
   - 6 Scenarios: Deterministic, Worst-case, Best-case, 
                  No Nuclear, Undersized BESS, All-Renewables
=============================================================================
"""

import os
import sys
import io
import json
import warnings
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Check for Pyomo
try:
    import pyomo.environ as pyo
    print(f"  Pyomo loaded")
except ImportError:
    print("  Installing Pyomo...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyomo', '-q'])
    import pyomo.environ as pyo

# Check for HiGHS solver
try:
    import highspy
    SOLVER = 'appsi_highs'
    print(f"  HiGHS solver available")
except ImportError:
    try:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'highspy', '-q'])
        SOLVER = 'appsi_highs'
    except:
        SOLVER = 'glpk'
        print(f"  Using GLPK solver (fallback)")

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR  = os.path.join(PROJECT_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

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
print("  PHASE 3: ROBUST MILP OPTIMIZATION")
print("  Nuclear Baseload + BESS under Duck Curve Uncertainty")
print("=" * 78)

# =====================================================================
#  SYSTEM PARAMETERS
# =====================================================================
# Nuclear plant parameters (based on Diablo Canyon, California)
NUCLEAR_CAPACITY_MW = 2256       # MW (Diablo Canyon: 2 x 1128 MW)
NUCLEAR_MIN_MW = 1800            # MW (minimum stable output ~80%)
NUCLEAR_RAMP_RATE = 100          # MW/h (limited ramp capability)
NUCLEAR_MARGINAL_COST = 12       # $/MWh (fuel + O&M, very low)

# Natural gas parameters (flexible peaking/CCGT)
GAS_CAPACITY_MW = 20000          # MW (CAISO total gas capacity)
GAS_MIN_MW = 0                   # MW 
GAS_RAMP_RATE = 5000             # MW/h (fast ramping)
GAS_MARGINAL_COST = 45           # $/MWh (fuel + O&M + carbon)

# Battery Energy Storage System (BESS)
BESS_POWER_MW = 5000             # MW (charging/discharging capacity)
BESS_ENERGY_MWH = 20000          # MWh (4-hour duration battery)
BESS_EFFICIENCY = 0.90           # Round-trip efficiency (sqrt for each direction)
BESS_SOC_MIN = 0.10              # Minimum state of charge (10%)
BESS_SOC_MAX = 0.90              # Maximum state of charge (90%)
BESS_CYCLING_COST = 5            # $/MWh (degradation cost)
BESS_INITIAL_SOC = 0.50          # Starting SoC (50%)

# Renewables curtailment cost
CURTAILMENT_COST = 10            # $/MWh (cost of curtailing renewables)

# Unserved energy penalty (load shedding)
VOLL = 10000                     # $/MWh (Value of Lost Load)

# Import/Export pricing
IMPORT_COST = 55                 # $/MWh
EXPORT_REVENUE = 20              # $/MWh

print("\n  System Parameters:")
print(f"    Nuclear: {NUCLEAR_CAPACITY_MW} MW (min {NUCLEAR_MIN_MW} MW)")
print(f"    Gas:     {GAS_CAPACITY_MW} MW")
print(f"    BESS:    {BESS_POWER_MW} MW / {BESS_ENERGY_MWH} MWh")
print(f"    BESS Eff: {BESS_EFFICIENCY*100:.0f}%")

# =====================================================================
#  LOAD ML PREDICTIONS
# =====================================================================
print("\n[1/4] Loading ML predictions...")

pred_path = os.path.join(DATA_DIR, "ml_predictions_for_milp.csv")
if not os.path.exists(pred_path):
    print(f"  ERROR: {pred_path} not found. Run Phase 2 first.")
    sys.exit(1)

preds = pd.read_csv(pred_path)
preds['timestamp'] = pd.to_datetime(preds['timestamp'])

# Use first 7 days for optimization (168 hours)
HORIZON = 168  # hours (1 week)
preds_week = preds.head(HORIZON).copy()

print(f"  Loaded {len(preds)} predictions")
print(f"  Optimization horizon: {HORIZON} hours (first week of test period)")
print(f"  Period: {preds_week['timestamp'].iloc[0]} -> {preds_week['timestamp'].iloc[-1]}")

# =====================================================================
#  MILP MODEL BUILDER
# =====================================================================
def build_dispatch_model(net_load_profile, price_profile, scenario_name,
                         nuclear_cap=NUCLEAR_CAPACITY_MW,
                         nuclear_min=NUCLEAR_MIN_MW,
                         bess_power=BESS_POWER_MW,
                         bess_energy=BESS_ENERGY_MWH,
                         use_nuclear=True):
    """
    Build and solve the MILP dispatch model.
    
    Min  sum_t [ C_nuc * P_nuc(t) + C_gas * P_gas(t) 
                 + C_bess * (P_ch(t) + P_dis(t))
                 + C_imp * P_imp(t) - R_exp * P_exp(t)
                 + VOLL * P_shed(t) + C_curt * P_curt(t) ]
    
    s.t. Power balance, ramp limits, BESS dynamics, capacity limits
    """
    T = len(net_load_profile)
    model = pyo.ConcreteModel(name=f"Dispatch_{scenario_name}")
    
    # Sets
    model.T = pyo.RangeSet(0, T-1)
    
    # Decision Variables
    if use_nuclear:
        model.P_nuc = pyo.Var(model.T, bounds=(nuclear_min, nuclear_cap), initialize=nuclear_cap*0.9)
    else:
        model.P_nuc = pyo.Var(model.T, bounds=(0, 0), initialize=0)
    
    model.P_gas = pyo.Var(model.T, bounds=(0, GAS_CAPACITY_MW), initialize=10000)
    model.P_bess_ch = pyo.Var(model.T, bounds=(0, bess_power), initialize=0)   # Charging
    model.P_bess_dis = pyo.Var(model.T, bounds=(0, bess_power), initialize=0)  # Discharging
    model.SoC = pyo.Var(model.T, bounds=(BESS_SOC_MIN * bess_energy, BESS_SOC_MAX * bess_energy))
    model.P_import = pyo.Var(model.T, bounds=(0, 10000), initialize=0)
    model.P_export = pyo.Var(model.T, bounds=(0, 6000), initialize=0)
    model.P_shed = pyo.Var(model.T, bounds=(0, 5000), initialize=0)        # Load shedding
    model.P_curt = pyo.Var(model.T, bounds=(0, 15000), initialize=0)       # Curtailment
    
    # Binary for BESS mode (can't charge and discharge simultaneously)
    model.u_bess = pyo.Var(model.T, within=pyo.Binary, initialize=0)  # 1=discharging
    
    # Objective: Minimize total cost
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
    
    # Constraints
    
    # C1: Power balance
    def power_balance_rule(m, t):
        supply = m.P_nuc[t] + m.P_gas[t] + m.P_bess_dis[t] + m.P_import[t] + m.P_shed[t]
        demand_side = net_load_profile[t] + m.P_bess_ch[t] + m.P_export[t] + m.P_curt[t]
        return supply == demand_side
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)
    
    # C2: BESS State of Charge dynamics
    eta = np.sqrt(BESS_EFFICIENCY)  # One-way efficiency
    
    def soc_init_rule(m):
        return m.SoC[0] == BESS_INITIAL_SOC * bess_energy
    model.soc_init = pyo.Constraint(rule=soc_init_rule)
    
    def soc_dynamics_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.SoC[t] == m.SoC[t-1] + eta * m.P_bess_ch[t] - (1/eta) * m.P_bess_dis[t]
    model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)
    
    # C3: BESS mutual exclusion (can't charge and discharge at same time)
    def bess_ch_limit(m, t):
        return m.P_bess_ch[t] <= bess_power * (1 - m.u_bess[t])
    model.bess_ch_lim = pyo.Constraint(model.T, rule=bess_ch_limit)
    
    def bess_dis_limit(m, t):
        return m.P_bess_dis[t] <= bess_power * m.u_bess[t]
    model.bess_dis_lim = pyo.Constraint(model.T, rule=bess_dis_limit)
    
    # C4: Nuclear ramp rate
    if use_nuclear:
        def nuc_ramp_up(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.P_nuc[t] - m.P_nuc[t-1] <= NUCLEAR_RAMP_RATE
        model.nuc_ramp_up = pyo.Constraint(model.T, rule=nuc_ramp_up)
        
        def nuc_ramp_down(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.P_nuc[t-1] - m.P_nuc[t] <= NUCLEAR_RAMP_RATE
        model.nuc_ramp_dn = pyo.Constraint(model.T, rule=nuc_ramp_down)
    
    # C5: Gas ramp rate
    def gas_ramp_up(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.P_gas[t] - m.P_gas[t-1] <= GAS_RAMP_RATE
    model.gas_ramp_up = pyo.Constraint(model.T, rule=gas_ramp_up)
    
    def gas_ramp_down(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.P_gas[t-1] - m.P_gas[t] <= GAS_RAMP_RATE
    model.gas_ramp_dn = pyo.Constraint(model.T, rule=gas_ramp_down)
    
    # Solve
    solver = pyo.SolverFactory(SOLVER)
    solver.options['time_limit'] = 300  # 5 minutes max
    
    result = solver.solve(model, tee=False)
    
    status = result.solver.termination_condition
    
    return model, status


# =====================================================================
#  STEP 2: RUN 6 SCENARIOS
# =====================================================================
print("\n[2/4] Running 6 optimization scenarios...")

scenarios = {
    'S1_Deterministic': {
        'net_load': preds_week['net_load_predicted'].values,
        'price': preds_week['price_predicted'].values,
        'desc': 'Point prediction (deterministic)',
        'use_nuclear': True,
        'bess_power': BESS_POWER_MW,
        'bess_energy': BESS_ENERGY_MWH,
    },
    'S2_Worst_Case': {
        'net_load': preds_week['net_load_Q90'].values,
        'price': preds_week['price_Q90'].values,
        'desc': 'Upper bound (Q90) - high demand scenario',
        'use_nuclear': True,
        'bess_power': BESS_POWER_MW,
        'bess_energy': BESS_ENERGY_MWH,
    },
    'S3_Best_Case': {
        'net_load': preds_week['net_load_Q10'].values,
        'price': preds_week['price_Q10'].values,
        'desc': 'Lower bound (Q10) - low demand scenario',
        'use_nuclear': True,
        'bess_power': BESS_POWER_MW,
        'bess_energy': BESS_ENERGY_MWH,
    },
    'S4_No_Nuclear': {
        'net_load': preds_week['net_load_predicted'].values,
        'price': preds_week['price_predicted'].values,
        'desc': 'Without nuclear baseload',
        'use_nuclear': False,
        'bess_power': BESS_POWER_MW,
        'bess_energy': BESS_ENERGY_MWH,
    },
    'S5_Small_BESS': {
        'net_load': preds_week['net_load_predicted'].values,
        'price': preds_week['price_predicted'].values,
        'desc': 'Undersized BESS (1000 MW / 4000 MWh)',
        'use_nuclear': True,
        'bess_power': 1000,
        'bess_energy': 4000,
    },
    'S6_Robust': {
        'net_load': preds_week['net_load_Q90'].values * 0.5 + preds_week['net_load_predicted'].values * 0.5,
        'price': preds_week['price_Q90'].values * 0.5 + preds_week['price_predicted'].values * 0.5,
        'desc': 'Robust (50% point + 50% worst case)',
        'use_nuclear': True,
        'bess_power': BESS_POWER_MW,
        'bess_energy': BESS_ENERGY_MWH,
    },
}

scenario_results = {}
for name, cfg in scenarios.items():
    print(f"\n  --- {name}: {cfg['desc']} ---")
    
    model, status = build_dispatch_model(
        net_load_profile=cfg['net_load'],
        price_profile=cfg['price'],
        scenario_name=name,
        nuclear_cap=NUCLEAR_CAPACITY_MW,
        nuclear_min=NUCLEAR_MIN_MW,
        bess_power=cfg['bess_power'],
        bess_energy=cfg['bess_energy'],
        use_nuclear=cfg['use_nuclear'],
    )
    
    print(f"    Solver status: {status}")
    
    if str(status) in ['optimal', 'feasible']:
        T = HORIZON
        result = {
            'status': str(status),
            'total_cost': pyo.value(model.objective),
            'P_nuc': [pyo.value(model.P_nuc[t]) for t in range(T)],
            'P_gas': [pyo.value(model.P_gas[t]) for t in range(T)],
            'P_bess_ch': [pyo.value(model.P_bess_ch[t]) for t in range(T)],
            'P_bess_dis': [pyo.value(model.P_bess_dis[t]) for t in range(T)],
            'SoC': [pyo.value(model.SoC[t]) for t in range(T)],
            'P_import': [pyo.value(model.P_import[t]) for t in range(T)],
            'P_export': [pyo.value(model.P_export[t]) for t in range(T)],
            'P_shed': [pyo.value(model.P_shed[t]) for t in range(T)],
            'P_curt': [pyo.value(model.P_curt[t]) for t in range(T)],
            'net_load': cfg['net_load'],
        }
        
        # Summary statistics
        result['avg_nuclear'] = np.mean(result['P_nuc'])
        result['avg_gas'] = np.mean(result['P_gas'])
        result['avg_import'] = np.mean(result['P_import'])
        result['total_shed'] = np.sum(result['P_shed'])
        result['total_curt'] = np.sum(result['P_curt'])
        result['bess_cycles'] = np.sum(result['P_bess_dis']) / cfg['bess_energy'] if cfg['bess_energy'] > 0 else 0
        result['cost_per_mwh'] = result['total_cost'] / (np.sum(cfg['net_load']))
        
        print(f"    Total cost:  ${result['total_cost']:>15,.0f}")
        print(f"    Cost/MWh:    ${result['cost_per_mwh']:>15.2f}")
        print(f"    Avg Nuclear: {result['avg_nuclear']:>10,.0f} MW")
        print(f"    Avg Gas:     {result['avg_gas']:>10,.0f} MW")
        print(f"    Avg Import:  {result['avg_import']:>10,.0f} MW")
        print(f"    Load shed:   {result['total_shed']:>10,.0f} MWh")
        print(f"    Curtailment: {result['total_curt']:>10,.0f} MWh")
        print(f"    BESS cycles: {result['bess_cycles']:>10.1f}")
        
        scenario_results[name] = result
    else:
        print(f"    INFEASIBLE or ERROR!")
        scenario_results[name] = {'status': str(status), 'total_cost': float('inf')}

# =====================================================================
#  STEP 3: COMPARISON TABLE
# =====================================================================
print("\n" + "=" * 78)
print("[3/4] Scenario comparison...")

comp_rows = []
for name, res in scenario_results.items():
    if res['status'] in ['optimal', 'feasible']:
        comp_rows.append({
            'Scenario': name,
            'Total Cost ($)': res['total_cost'],
            'Cost/MWh ($/MWh)': res['cost_per_mwh'],
            'Avg Nuclear (MW)': res['avg_nuclear'],
            'Avg Gas (MW)': res['avg_gas'],
            'Avg Import (MW)': res['avg_import'],
            'Load Shed (MWh)': res['total_shed'],
            'Curtailment (MWh)': res['total_curt'],
            'BESS Cycles': res['bess_cycles'],
        })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(TABLE_DIR, "scenario_comparison.csv"), index=False)

print(f"\n  {'Scenario':<20} {'Total Cost':>15} {'$/MWh':>8} {'Nuclear':>8} {'Gas':>8} {'Import':>8} {'Shed':>8}")
print(f"  {'-'*88}")
for _, row in comp_df.iterrows():
    print(f"  {row['Scenario']:<20} ${row['Total Cost ($)']:>13,.0f} ${row['Cost/MWh ($/MWh)']:>6.2f} "
          f"{row['Avg Nuclear (MW)']:>7,.0f} {row['Avg Gas (MW)']:>7,.0f} {row['Avg Import (MW)']:>7,.0f} "
          f"{row['Load Shed (MWh)']:>7,.0f}")

# =====================================================================
#  STEP 4: PUBLICATION FIGURES
# =====================================================================
print("\n" + "=" * 78)
print("[4/4] Generating MILP publication figures...")

timestamps = preds_week['timestamp'].values

# --- Figure 13: Dispatch Stack Chart (S1 Deterministic) ---
if 'S1_Deterministic' in scenario_results and scenario_results['S1_Deterministic']['status'] in ['optimal', 'feasible']:
    res = scenario_results['S1_Deterministic']
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    
    # Top: Generation Stack
    ax = axes[0]
    t = np.arange(HORIZON)
    
    nuc = np.array(res['P_nuc'])
    gas = np.array(res['P_gas'])
    bess_dis = np.array(res['P_bess_dis'])
    imp = np.array(res['P_import'])
    
    ax.fill_between(t, 0, nuc, alpha=0.8, color='#4CAF50', label='Nuclear')
    ax.fill_between(t, nuc, nuc+gas, alpha=0.7, color='#FF9800', label='Natural Gas')
    ax.fill_between(t, nuc+gas, nuc+gas+bess_dis, alpha=0.7, color='#2196F3', label='BESS Discharge')
    ax.fill_between(t, nuc+gas+bess_dis, nuc+gas+bess_dis+imp, alpha=0.5, color='#9C27B0', label='Import')
    
    ax.plot(t, res['net_load'], 'k-', linewidth=2, label='Net Load')
    ax.set_ylabel('Power (MW)')
    ax.set_title('Optimal Dispatch: Generation Stack (S1 Deterministic)', fontweight='bold')
    ax.legend(loc='upper right', ncol=5)
    ax.set_ylim(0)
    
    # Middle: BESS Operation
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
    
    # Bottom: Cost breakdown
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
    print(f"  -> fig13_dispatch_stack.png")

# --- Figure 14: Scenario Cost Comparison ---
if comp_df is not None and len(comp_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Total cost bars
    ax = axes[0]
    colors_sc = ['#4CAF50', '#E53935', '#2196F3', '#FF9800', '#9C27B0', '#795548']
    bars = ax.bar(range(len(comp_df)), comp_df['Total Cost ($)'] / 1e6, 
                  color=colors_sc[:len(comp_df)], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(comp_df)))
    ax.set_xticklabels([s.replace('_', '\n') for s in comp_df['Scenario']], fontsize=8)
    ax.set_ylabel('Total Cost (Million $)')
    ax.set_title('Scenario Total Cost Comparison', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, comp_df['Total Cost ($)'] / 1e6):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'${val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right: Cost per MWh
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
    print(f"  -> fig14_scenario_comparison.png")

# --- Figure 15: Nuclear Impact Analysis ---
if 'S1_Deterministic' in scenario_results and 'S4_No_Nuclear' in scenario_results:
    s1 = scenario_results['S1_Deterministic']
    s4 = scenario_results['S4_No_Nuclear']
    
    if s1['status'] in ['optimal', 'feasible'] and s4['status'] in ['optimal', 'feasible']:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        t = np.arange(HORIZON)
        
        # Top: With Nuclear
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
        
        # Bottom: Without Nuclear
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
        print(f"  -> fig15_nuclear_impact.png")

# =====================================================================
#  FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 78)
print("  PHASE 3 COMPLETE: MILP Optimization Results")
print("=" * 78)

print(f"\n  Solved {len([r for r in scenario_results.values() if r['status'] in ['optimal', 'feasible']])} "
      f"of {len(scenarios)} scenarios successfully")

print(f"\n  Output files:")
for f in sorted(os.listdir(FIG_DIR)):
    if 'fig1' in f and f.endswith('.png'):
        fpath = os.path.join(FIG_DIR, f)
        print(f"    * {f} ({os.path.getsize(fpath)/1024:.0f} KB)")

print(f"\n  Tables:")
for f in sorted(os.listdir(TABLE_DIR)):
    if f.endswith('.csv'):
        print(f"    * {f}")

print("\n" + "=" * 78)
print("  ALL PHASES COMPLETE!")
print("=" * 78)
