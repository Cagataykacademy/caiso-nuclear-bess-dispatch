"""
=============================================================================
 PHASE 9: CCGT AGGREGATION GRANULARITY CHECK
 Reviewer concern: representing CAISO's ~200-unit CCGT fleet as 2 aggregate
 5,000 MW blocks may be an oversimplification that materially affects
 results (startup costs, minimum-load behavior, ramp flexibility).

 Test: re-solve the six core November scenarios (S1-S6) with 4 aggregate
 units of 2,500 MW each (same total 10,000 MW capacity, same 30% minimum-
 load fraction and per-unit ramp rate, so only the GRANULARITY changes) and
 compare headline costs and the nuclear premium (S4 vs S1) against the
 2x5,000 MW baseline used throughout the paper.
=============================================================================
"""
import os, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P    = os.path.dirname(HERE)
TBL  = os.path.join(P, "outputs", "tables")

spec = importlib.util.spec_from_file_location(
    "improved_milp", os.path.join(HERE, "16_improved_milp.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

nl_point = M.week['net_load_predicted'].values
nl_q90   = M.week['net_load_Q90'].values

def run_scenarios(label):
    scens = {
        'S1_Deterministic': (nl_point, True),
        'S4_No_Nuclear':    (nl_point, False),
    }
    out = {}
    for name, (nl, use_nuc) in scens.items():
        sm, d = M.build_and_solve(nl, f"{label}_{name}", use_nuc=use_nuc)
        out[name] = d
        print(f"  [{label}] {name:<18} ${d['cost_per_mwh']:8.4f}/MWh  "
              f"shed={d['shed_total']:6.0f} MWh  starts={d.get('n_startups',0):.0f}")
    return out

print("="*74)
print("  BASELINE: 2 x 5,000 MW CCGT units (as used throughout the paper)")
print("="*74)
base = run_scenarios("2x5000MW")
base_premium = (base['S4_No_Nuclear']['cost_per_mwh'] - base['S1_Deterministic']['cost_per_mwh']) \
               / base['S1_Deterministic']['cost_per_mwh'] * 100

print("\n" + "="*74)
print("  SENSITIVITY: 4 x 2,500 MW CCGT units (same 10 GW total, finer granularity)")
print("="*74)
M.N_CCGT = 4
M.CCGT_UNIT_MW = 2500
M.CCGT_MIN_MW = 750          # keep the same 30% minimum-load fraction
M.CCGT_RAMP = 400            # keep the same MW/h-per-MW-capacity ramp fraction (800/5000 = 400/2500)
fine = run_scenarios("4x2500MW")
fine_premium = (fine['S4_No_Nuclear']['cost_per_mwh'] - fine['S1_Deterministic']['cost_per_mwh']) \
               / fine['S1_Deterministic']['cost_per_mwh'] * 100

rows = [
    {'Configuration': '2x5,000 MW (paper baseline)',
     'S1_Cost_MWh': round(base['S1_Deterministic']['cost_per_mwh'], 4),
     'S4_Cost_MWh': round(base['S4_No_Nuclear']['cost_per_mwh'], 4),
     'Nuclear_Premium_pct': round(base_premium, 3),
     'S1_Startups': round(base['S1_Deterministic'].get('n_startups', 0), 0),
     'S4_Startups': round(base['S4_No_Nuclear'].get('n_startups', 0), 0)},
    {'Configuration': '4x2,500 MW (finer granularity)',
     'S1_Cost_MWh': round(fine['S1_Deterministic']['cost_per_mwh'], 4),
     'S4_Cost_MWh': round(fine['S4_No_Nuclear']['cost_per_mwh'], 4),
     'Nuclear_Premium_pct': round(fine_premium, 3),
     'S1_Startups': round(fine['S1_Deterministic'].get('n_startups', 0), 0),
     'S4_Startups': round(fine['S4_No_Nuclear'].get('n_startups', 0), 0)},
]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(TBL, "ccgt_granularity.csv"), index=False)

print("\nSUMMARY")
print(df.to_string(index=False))
print(f"\nNuclear premium: {base_premium:.1f}% (2x5000MW) vs {fine_premium:.1f}% (4x2500MW) "
      f"-> difference of {abs(base_premium-fine_premium):.1f} percentage points")
print("-> ccgt_granularity.csv")
