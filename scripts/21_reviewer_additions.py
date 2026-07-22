"""
=============================================================================
 PHASE 5 (REVIEWER ADDITIONS):
  A) Uncalibrated QR vs CQR comparison (coverage / width / Winkler).
     The uncalibrated interval is recovered exactly from the shipped
     predictions by removing the constant conformal correction Q_hat=1,044 MW
     (CQR shifts both bounds by a constant, so no retraining is needed).
  B) Import price sensitivity ($45-$70/MWh) for S1 (with nuclear) and
     S4 (no nuclear) -> stability of the nuclear premium.

 Outputs:
   - outputs/tables/qr_vs_cqr.csv
   - outputs/tables/sensitivity_import_price.csv
=============================================================================
"""
import os, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P    = os.path.dirname(HERE)
TBL  = os.path.join(P, "outputs", "tables")
Q_HAT = 1044.0
ALPHA = 0.10

# =========================================================================
# A) QR vs CQR
# =========================================================================
pred = pd.read_csv(os.path.join(P, "data", "ml_predictions_for_milp.csv"))
y  = pred['net_load_actual'].values
lo_c, hi_c = pred['net_load_Q10'].values, pred['net_load_Q90'].values   # calibrated
lo_u, hi_u = lo_c + Q_HAT, hi_c - Q_HAT                                 # uncalibrated

def interval_metrics(lo, hi, y, alpha=ALPHA):
    cov = ((y >= lo) & (y <= hi)).mean() * 100
    width = np.mean(hi - lo)
    winkler = np.mean((hi - lo) + (2/alpha)*np.maximum(lo - y, 0)
                      + (2/alpha)*np.maximum(y - hi, 0))
    return cov, width, winkler

rows = []
for name, lo, hi in [("QR (uncalibrated, q10-q90)", lo_u, hi_u),
                     ("CQR (conformalized)",        lo_c, hi_c)]:
    cov, width, wink = interval_metrics(lo, hi, y)
    rows.append({'Interval': name, 'Coverage_pct': round(cov, 1),
                 'Mean_Width_MW': round(width, 0),
                 'Winkler_MW': round(wink, 0)})
    print(f"  {name:<30} cov={cov:5.1f}%  width={width:7,.0f}  Winkler={wink:7,.0f}")
pd.DataFrame(rows).to_csv(os.path.join(TBL, "qr_vs_cqr.csv"), index=False)
print("  -> qr_vs_cqr.csv")

# =========================================================================
# B) Import price sensitivity (loads improved MILP module; re-runs its base)
# =========================================================================
spec = importlib.util.spec_from_file_location(
    "improved_milp", os.path.join(HERE, "16_improved_milp.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

nl_det = M.week['net_load_predicted'].values
rows = []
for imp_cost in [45, 50, 55, 60, 65, 70]:
    M.IMP_COST = imp_cost   # module-level constant read by the objective
    _, d1 = M.build_and_solve(nl_det, f"S1_imp{imp_cost}", use_nuc=True)
    _, d4 = M.build_and_solve(nl_det, f"S4_imp{imp_cost}", use_nuc=False)
    prem_abs = d4['cost_per_mwh'] - d1['cost_per_mwh']
    prem_pct = prem_abs / d1['cost_per_mwh'] * 100
    rows.append({'Import_Cost_MWh': imp_cost,
                 'S1_Cost_MWh': round(d1['cost_per_mwh'], 2),
                 'S4_Cost_MWh': round(d4['cost_per_mwh'], 2),
                 'Premium_MWh': round(prem_abs, 2),
                 'Premium_pct': round(prem_pct, 1),
                 'S1_Avg_Import_MW': round(d1['avg_imp'], 0),
                 'S4_Avg_Import_MW': round(d4['avg_imp'], 0),
                 'S4_Shed_MWh': round(d4['shed_total'], 0)})
    print(f"  imp=${imp_cost}/MWh  S1=${d1['cost_per_mwh']:.2f}  "
          f"S4=${d4['cost_per_mwh']:.2f}  premium=+{prem_pct:.1f}%")
M.IMP_COST = 55  # restore default

pd.DataFrame(rows).to_csv(os.path.join(TBL, "sensitivity_import_price.csv"),
                          index=False)
print("  -> sensitivity_import_price.csv")
