"""
=============================================================================
 PHASE 6: TWO-STAGE STOCHASTIC PROGRAMMING BASELINE
 Reviewer question: "With the same information set, what would a classical
 scenario-based two-stage SP deliver compared with the CQR-robust approach?"

 Design
 ------
 Information set: identical to the robust approach — the CQR triplet
   {Q10, point, Q90} for the November test week.
 Scenarios: 3-point Swanson–Megill approximation (Keefer & Bodily, 1983):
   p = {0.30 (Q10), 0.40 (point), 0.30 (Q90)}.
 First stage (here-and-now): CCGT commitment u[t,k], startups y[t,k],
   nuclear trajectory P_nuc[t]  (slow / inflexible decisions).
 Second stage (recourse, per scenario): all continuous dispatch + BESS.

 Out-of-sample evaluation protocol (identical for every plan):
   fix the plan's first-stage decisions, re-solve dispatch against the
   REALIZED net load -> realized cost & shedding.
   Plans compared: Deterministic (γ=0), Robust γ=0.5, Robust γ=1, SP.

 Outputs:
   - outputs/tables/sp_vs_robust.csv
=============================================================================
"""
import os, time
import numpy as np
import pandas as pd
import pyomo.environ as pyo

P   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL = os.path.join(P, "outputs", "tables")

# ---- system parameters (identical to 16_improved_milp.py) ----------------
NUC_CAP, NUC_MIN, NUC_RAMP, NUC_COST = 2256, 1800, 100, 12
N_CCGT, CCGT_UNIT_MW, CCGT_MIN_MW = 2, 5000, 1500
CCGT_RAMP, CCGT_COST, CCGT_STARTUP = 800, 40, 25000
CCGT_UP_MIN, CCGT_DN_MIN = 4, 2
PEAK_CAP, PEAK_RAMP, PEAK_COST = 10000, 5000, 65
BESS_POW, BESS_ENE, BESS_EFF = 5000, 20000, 0.90
SOC_MIN, SOC_MAX, SOC_INIT, SOC_END, BESS_DEG = 0.10, 0.90, 0.50, 0.35, 5
IMP_CAP, EXP_CAP, IMP_COST, EXP_REV = 10000, 6000, 55, 20
SHED_CAP, VOLL, CURT_CAP, CURT_COST = 5000, 10000, 20000, 10
T = 168
ETA = np.sqrt(BESS_EFF)

SOLVER = 'appsi_highs'


def _bess_block(m, sset=None):
    """Attach BESS + grid variables, indexed by (t) or (t, s)."""
    idx = (m.T, sset) if sset is not None else (m.T,)
    m.P_peak = pyo.Var(*idx, bounds=(0, PEAK_CAP))
    m.P_ch   = pyo.Var(*idx, bounds=(0, BESS_POW))
    m.P_dis  = pyo.Var(*idx, bounds=(0, BESS_POW))
    m.SoC    = pyo.Var(*idx, bounds=(SOC_MIN*BESS_ENE, SOC_MAX*BESS_ENE))
    m.P_imp  = pyo.Var(*idx, bounds=(0, IMP_CAP))
    m.P_exp  = pyo.Var(*idx, bounds=(0, EXP_CAP))
    m.P_shed = pyo.Var(*idx, bounds=(0, SHED_CAP))
    m.P_curt = pyo.Var(*idx, bounds=(0, CURT_CAP))
    m.u_bess = pyo.Var(*idx, within=pyo.Binary)


def build_model(nl_scenarios, probs, fix_uc=None, fix_nuc=None):
    """
    Generic builder.
      nl_scenarios: list of 168-length arrays (1 -> deterministic model,
                    >1 -> two-stage SP with shared first stage).
      probs:        scenario probabilities (sum to 1).
      fix_uc:       optional (u[t][k], y[t][k]) arrays to freeze commitment.
      fix_nuc:      optional 168-array to freeze the nuclear trajectory.
    """
    S = len(nl_scenarios)
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T-1)
    m.K = pyo.RangeSet(0, N_CCGT-1)
    m.S = pyo.RangeSet(0, S-1)

    # ---- first stage ----
    m.P_nuc  = pyo.Var(m.T, bounds=(NUC_MIN, NUC_CAP))
    m.u_ccgt = pyo.Var(m.T, m.K, within=pyo.Binary)
    m.y_ccgt = pyo.Var(m.T, m.K, within=pyo.Binary)

    # ---- second stage ----
    m.x_ccgt = pyo.Var(m.T, m.K, m.S, bounds=(0, CCGT_UNIT_MW))
    _bess_block(m, m.S)

    def obj(md):
        first = sum(NUC_COST*md.P_nuc[t]
                    + CCGT_STARTUP*sum(md.y_ccgt[t, k] for k in md.K)
                    for t in md.T)
        second = sum(probs[s]*sum(
            CCGT_COST*sum(md.x_ccgt[t, k, s] for k in md.K)
            + PEAK_COST*md.P_peak[t, s]
            + BESS_DEG*(md.P_ch[t, s] + md.P_dis[t, s])
            + IMP_COST*md.P_imp[t, s] - EXP_REV*md.P_exp[t, s]
            + VOLL*md.P_shed[t, s] + CURT_COST*md.P_curt[t, s]
            for t in md.T) for s in md.S)
        return first + second
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    def balance(md, t, s):
        return (md.P_nuc[t] + sum(md.x_ccgt[t, k, s] for k in md.K)
                + md.P_peak[t, s] + md.P_dis[t, s] + md.P_imp[t, s]
                + md.P_shed[t, s]
                == nl_scenarios[s][t] + md.P_ch[t, s] + md.P_exp[t, s]
                + md.P_curt[t, s])
    m.c_bal = pyo.Constraint(m.T, m.S, rule=balance)

    m.c_soc0 = pyo.Constraint(m.S, rule=lambda md, s:
                              md.SoC[0, s] == SOC_INIT*BESS_ENE)
    m.c_soc = pyo.Constraint(m.T, m.S, rule=lambda md, t, s:
        pyo.Constraint.Skip if t == 0 else
        md.SoC[t, s] == md.SoC[t-1, s] + ETA*md.P_ch[t, s]
        - md.P_dis[t, s]/ETA)
    m.c_socend = pyo.Constraint(m.S, rule=lambda md, s:
                                md.SoC[T-1, s] >= SOC_END*BESS_ENE)
    m.c_ch  = pyo.Constraint(m.T, m.S, rule=lambda md, t, s:
                             md.P_ch[t, s] <= BESS_POW*(1-md.u_bess[t, s]))
    m.c_dis = pyo.Constraint(m.T, m.S, rule=lambda md, t, s:
                             md.P_dis[t, s] <= BESS_POW*md.u_bess[t, s])

    m.c_nup = pyo.Constraint(m.T, rule=lambda md, t:
        pyo.Constraint.Skip if t == 0 else
        md.P_nuc[t] - md.P_nuc[t-1] <= NUC_RAMP)
    m.c_ndn = pyo.Constraint(m.T, rule=lambda md, t:
        pyo.Constraint.Skip if t == 0 else
        md.P_nuc[t-1] - md.P_nuc[t] <= NUC_RAMP)

    m.c_ci = pyo.Constraint(m.K, rule=lambda md, k: md.u_ccgt[0, k] == 1)
    m.c_cmax = pyo.Constraint(m.T, m.K, m.S, rule=lambda md, t, k, s:
                              md.x_ccgt[t, k, s] <= CCGT_UNIT_MW*md.u_ccgt[t, k])
    m.c_cmin = pyo.Constraint(m.T, m.K, m.S, rule=lambda md, t, k, s:
                              md.x_ccgt[t, k, s] >= CCGT_MIN_MW*md.u_ccgt[t, k])
    m.c_start = pyo.Constraint(m.T, m.K, rule=lambda md, t, k:
        pyo.Constraint.Skip if t == 0 else
        md.y_ccgt[t, k] >= md.u_ccgt[t, k] - md.u_ccgt[t-1, k])

    def minup(md, t, k):
        if t == 0: return pyo.Constraint.Skip
        end = min(t + CCGT_UP_MIN - 1, T-1)
        return sum(md.u_ccgt[tau, k] for tau in range(t, end+1)) >= \
               CCGT_UP_MIN*md.y_ccgt[t, k]
    m.c_minup = pyo.Constraint(m.T, m.K, rule=minup)

    def mindn(md, t, k):
        if t == 0: return pyo.Constraint.Skip
        end = min(t + CCGT_DN_MIN - 1, T-1)
        return sum(1 - md.u_ccgt[tau, k] for tau in range(t, end+1)) >= \
               CCGT_DN_MIN*(md.u_ccgt[t-1, k] - md.u_ccgt[t, k])
    m.c_mindn = pyo.Constraint(m.T, m.K, rule=mindn)

    m.c_cup = pyo.Constraint(m.T, m.K, m.S, rule=lambda md, t, k, s:
        pyo.Constraint.Skip if t == 0 else
        md.x_ccgt[t, k, s] - md.x_ccgt[t-1, k, s]
        <= CCGT_RAMP*md.u_ccgt[t-1, k] + CCGT_UNIT_MW*md.y_ccgt[t, k])
    m.c_cdn = pyo.Constraint(m.T, m.K, m.S, rule=lambda md, t, k, s:
        pyo.Constraint.Skip if t == 0 else
        md.x_ccgt[t-1, k, s] - md.x_ccgt[t, k, s] <= CCGT_RAMP)
    m.c_pup = pyo.Constraint(m.T, m.S, rule=lambda md, t, s:
        pyo.Constraint.Skip if t == 0 else
        md.P_peak[t, s] - md.P_peak[t-1, s] <= PEAK_RAMP)
    m.c_pdn = pyo.Constraint(m.T, m.S, rule=lambda md, t, s:
        pyo.Constraint.Skip if t == 0 else
        md.P_peak[t-1, s] - md.P_peak[t, s] <= PEAK_RAMP)

    # ---- optional first-stage freezing (for out-of-sample evaluation) ----
    if fix_uc is not None:
        u_fix, y_fix = fix_uc
        for t in range(T):
            for k in range(N_CCGT):
                m.u_ccgt[t, k].fix(round(u_fix[t][k]))
                m.y_ccgt[t, k].fix(round(y_fix[t][k]))
    if fix_nuc is not None:
        for t in range(T):
            m.P_nuc[t].fix(fix_nuc[t])
    return m


def solve(m):
    solver = pyo.SolverFactory(SOLVER)
    solver.options['time_limit'] = 600
    t0 = time.time()
    res = solver.solve(m, tee=False)
    dt = time.time() - t0
    status = str(res.solver.termination_condition)
    return status, dt


def extract_plan(m):
    """First-stage decisions from a solved model."""
    u = [[pyo.value(m.u_ccgt[t, k]) for k in range(N_CCGT)] for t in range(T)]
    y = [[pyo.value(m.y_ccgt[t, k]) for k in range(N_CCGT)] for t in range(T)]
    nuc = [pyo.value(m.P_nuc[t]) for t in range(T)]
    return u, y, nuc


def scenario_cost(m, s, probs_ignored=False):
    """Full cost of scenario s including first-stage cost terms."""
    first = sum(NUC_COST*pyo.value(m.P_nuc[t])
                + CCGT_STARTUP*sum(pyo.value(m.y_ccgt[t, k])
                                   for k in range(N_CCGT))
                for t in range(T))
    second = sum(
        CCGT_COST*sum(pyo.value(m.x_ccgt[t, k, s]) for k in range(N_CCGT))
        + PEAK_COST*pyo.value(m.P_peak[t, s])
        + BESS_DEG*(pyo.value(m.P_ch[t, s]) + pyo.value(m.P_dis[t, s]))
        + IMP_COST*pyo.value(m.P_imp[t, s]) - EXP_REV*pyo.value(m.P_exp[t, s])
        + VOLL*pyo.value(m.P_shed[t, s]) + CURT_COST*pyo.value(m.P_curt[t, s])
        for t in range(T))
    shed = sum(pyo.value(m.P_shed[t, s]) for t in range(T))
    return first + second, shed


# ===========================================================================
print("="*74)
print("  TWO-STAGE SP BASELINE + OUT-OF-SAMPLE PLAN EVALUATION")
print("="*74)

pred = pd.read_csv(os.path.join(P, "data", "ml_predictions_for_milp.csv"))
wk = pred.head(T)
nl_point = wk['net_load_predicted'].values
nl_q10   = wk['net_load_Q10'].values
nl_q90   = wk['net_load_Q90'].values
nl_real  = wk['net_load_actual'].values
real_sum = float(np.sum(nl_real))

PROBS_SM = [0.30, 0.40, 0.30]   # Swanson–Megill for P10/P50/P90

plans = {}

# --- planning models -------------------------------------------------------
print("\n[1/2] Solving planning models...")
plan_specs = {
    'Deterministic (γ=0)':  ([nl_point], [1.0]),
    'Robust (γ=0.5)':       ([0.5*nl_point + 0.5*nl_q90], [1.0]),
    'Robust (γ=1, Q90)':    ([nl_q90], [1.0]),
    'SP (3-scenario)':      ([nl_q10, nl_point, nl_q90], PROBS_SM),
}
for name, (scens, probs) in plan_specs.items():
    m = build_model(scens, probs)
    status, dt = solve(m)
    obj = pyo.value(m.obj)
    nb = sum(1 for v in m.component_data_objects(pyo.Var)
             if v.is_binary() and not v.fixed)
    plan_nl = float(sum(probs[s]*np.sum(scens[s]) for s in range(len(scens))))
    plans[name] = {'uc': extract_plan(m), 'plan_obj_mwh': obj/plan_nl,
                   'binaries': nb, 'solve_s': dt, 'status': status}
    print(f"  {name:<24} plan=${obj/plan_nl:6.2f}/MWh  bin={nb:5d}  "
          f"t={dt:.2f}s  [{status}]")

# --- out-of-sample recourse against realized net load ----------------------
print("\n[2/2] Recourse evaluation against REALIZED net load...")
rows = []
for name, info in plans.items():
    u, y, nuc = info['uc']
    mr = build_model([nl_real], [1.0], fix_uc=(u, y), fix_nuc=nuc)
    status, dt = solve(mr)
    cost, shed = scenario_cost(mr, 0)
    rows.append({'Plan': name,
                 'Plan_Cost_MWh': round(info['plan_obj_mwh'], 2),
                 'Realized_Cost_MWh': round(cost/real_sum, 2),
                 'Realized_Shed_MWh': round(shed, 0),
                 'Binaries': info['binaries'],
                 'Plan_Solve_s': round(info['solve_s'], 2)})
    print(f"  {name:<24} realized=${cost/real_sum:6.2f}/MWh  "
          f"shed={shed:6.0f} MWh  [{status}]")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(TBL, "sp_vs_robust.csv"), index=False)
print("\n  -> sp_vs_robust.csv")
print(df.to_string(index=False))
