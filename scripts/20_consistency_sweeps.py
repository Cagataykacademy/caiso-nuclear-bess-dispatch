"""
=============================================================================
 PHASE 4 (CONSISTENCY): Re-run BESS / gas-price / CO2 / gamma-CO2 analyses
 with the IMPROVED two-tier gas + UC MILP (script 16), replacing legacy
 results that were produced with the old single-tier model.

 Outputs (tables):
   - sensitivity_bess.csv        (overwrites legacy single-tier version)
   - sensitivity_gas_price.csv   (overwrites legacy single-tier version)
   - co2_analysis.csv            (overwrites legacy version; S1..S6, Nov week)
   - gamma_sweep.csv             (adds CO2 column)
 Outputs (figures):
   - fig15_sensitivity_analysis.png  (3-panel: BESS / nuclear / gas price)
   - fig_co2_analysis.png            (CO2 by scenario + cost-emissions plane)
   - fig_robust_pareto.png           (gamma vs cost & CO2 tradeoff)

 CO2 emission factors (t CO2 / MWh):
   CCGT   0.37  (heat rate ~7.0 MMBtu/MWh x 53.07 kg/MMBtu, EPA eGRID 2023)
   Peaker 0.55  (heat rate ~10.5 MMBtu/MWh, simple-cycle GT)
   Import 0.428 (CARB default emission factor for unspecified imports,
                 MRR Section 95111; conservative upper bound)
   Nuclear / BESS / renewables: 0
=============================================================================
"""
import os, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P    = os.path.dirname(HERE)
FIG  = os.path.join(P, "outputs", "figures")
TBL  = os.path.join(P, "outputs", "tables")

# --- load improved MILP module (re-runs its 6 scenarios + nuclear sweep) ---
spec = importlib.util.spec_from_file_location(
    "improved_milp", os.path.join(HERE, "16_improved_milp.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

import matplotlib.pyplot as plt

EF_CCGT, EF_PEAK, EF_IMP = 0.37, 0.55, 0.428  # t CO2 / MWh

def co2_of(d):
    """Weekly CO2 (tonnes) of a dispatch dict from build_and_solve."""
    return (np.sum(d['P_ccgt']) * EF_CCGT
            + np.sum(d['P_peak']) * EF_PEAK
            + np.sum(d['P_imp'])  * EF_IMP)

nl_det = M.week['net_load_predicted'].values
nl_q90 = M.week['net_load_Q90'].values

# =========================================================================
# 1) BESS SWEEP (improved model)
# =========================================================================
print("\n[1/4] BESS capacity sweep (two-tier UC model)...")
bess_rows = []
for pow_mw in [500, 1000, 2000, 3000, 5000, 7500, 10000]:
    ene = pow_mw * 4
    sm, d = M.build_and_solve(nl_det, f"BESS_{pow_mw}",
                              bess_pow=pow_mw, bess_ene=ene)
    if d:
        bess_rows.append({
            'BESS_Power_MW': pow_mw, 'BESS_Energy_MWh': ene,
            'Cost_MWh': round(d['cost_per_mwh'], 3),
            'Avg_CCGT_MW': round(d['avg_ccgt'], 0),
            'Avg_Peaker_MW': round(d['avg_peak'], 0),
            'Avg_Import_MW': round(d['avg_imp'], 0),
            'BESS_Cycles': round(d['bess_cycles'], 2),
            'Shed_MWh': round(d['shed_total'], 0),
            'CO2_tonnes': round(co2_of(d), 0),
            'Solve_s': round(sm['solve_time_s'], 2)})
        print(f"  {pow_mw:6,} MW  ${d['cost_per_mwh']:6.2f}/MWh  "
              f"cycles={d['bess_cycles']:.2f}")
bess_df = pd.DataFrame(bess_rows)
bess_df.to_csv(os.path.join(TBL, "sensitivity_bess.csv"), index=False)

# =========================================================================
# 2) GAS PRICE SWEEP (scale both tiers, keep $25 peaker premium)
# =========================================================================
print("\n[2/4] Gas price sweep (two-tier UC model)...")
gas_rows = []
for g in [25, 35, 45, 55, 65, 75, 85, 100]:
    sm, d = M.build_and_solve(nl_det, f"Gas_{g}",
                              ccgt_cost=g, peak_cost=g + 25)
    if d:
        gas_rows.append({
            'CCGT_Cost_MWh': g, 'Peaker_Cost_MWh': g + 25,
            'Cost_MWh': round(d['cost_per_mwh'], 3),
            'Avg_Gas_Total_MW': round(d['avg_gas'], 0),
            'Avg_CCGT_MW': round(d['avg_ccgt'], 0),
            'Avg_Peaker_MW': round(d['avg_peak'], 0),
            'Avg_Import_MW': round(d['avg_imp'], 0),
            'CO2_tonnes': round(co2_of(d), 0),
            'Solve_s': round(sm['solve_time_s'], 2)})
        print(f"  ${g:3}/MWh  cost=${d['cost_per_mwh']:6.2f}  "
              f"gas={d['avg_gas']:7,.0f} MW  imp={d['avg_imp']:6,.0f} MW")
gas_df = pd.DataFrame(gas_rows)
gas_df.to_csv(os.path.join(TBL, "sensitivity_gas_price.csv"), index=False)

# =========================================================================
# 3) CO2 ANALYSIS for the six November scenarios (S1..S6)
# =========================================================================
print("\n[3/4] CO2 analysis (S1..S6, improved model)...")
co2_rows = []
for sname, d in M.disp_all.items():
    if d:
        nl_sum = float(np.sum(M.scenarios[sname]['nl']))
        co2 = co2_of(d)
        co2_rows.append({'Scenario': sname,
                         'Cost_MWh': round(d['cost_per_mwh'], 2),
                         'CO2_tonnes': round(co2, 0),
                         'CO2_kg_MWh': round(co2 * 1000 / nl_sum, 1),
                         'Shed_MWh': round(d['shed_total'], 0)})
co2_df = pd.DataFrame(co2_rows)
co2_df.to_csv(os.path.join(TBL, "co2_analysis.csv"), index=False)

s1co2 = co2_df.loc[co2_df.Scenario == 'S1_Deterministic', 'CO2_tonnes'].iloc[0]
s4co2 = co2_df.loc[co2_df.Scenario == 'S4_No_Nuclear',   'CO2_tonnes'].iloc[0]
annual_saving_mt = (s4co2 - s1co2) * 52 / 1e6
print(f"  Nuclear weekly CO2 saving: {s4co2 - s1co2:,.0f} t  "
      f"(~{annual_saving_mt:.1f} Mt/yr annualized)")

# =========================================================================
# 4) GAMMA SWEEP with CO2 (improved model)
# =========================================================================
print("\n[4/4] Gamma sweep with CO2...")
gam_rows = []
for g in [0.0, 0.25, 0.50, 0.75, 1.00]:
    nl = (1 - g) * nl_det + g * nl_q90
    sm, d = M.build_and_solve(nl, f"Gamma_{g}")
    if d:
        gam_rows.append({'gamma': g,
                         'cost_mwh': round(d['cost_per_mwh'], 2),
                         'shed_mwh': round(d['shed_total'], 0),
                         'avg_gas_mw': round(d['avg_gas'], 0),
                         'co2_tonnes': round(co2_of(d), 0)})
        print(f"  gamma={g:.2f}  ${d['cost_per_mwh']:.2f}/MWh  "
              f"CO2={co2_of(d):,.0f} t")
gam_df = pd.DataFrame(gam_rows)
gam_df.to_csv(os.path.join(TBL, "gamma_sweep.csv"), index=False)

# =========================================================================
# FIGURES
# =========================================================================
print("\nRendering figures...")

# --- fig15: 3-panel sensitivity ---
nuc_df = pd.read_csv(os.path.join(TBL, "sensitivity_nuclear.csv"))
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
ax.plot(bess_df['BESS_Power_MW'] / 1000, bess_df['Cost_MWh'],
        'o-', color='#2196F3', lw=2, ms=7)
ax.axvline(5, color='red', ls='--', alpha=0.6, label='CAISO 2023 (5 GW)')
ax.set_xlabel('BESS Power Capacity (GW)')
ax.set_ylabel('System Cost ($/MWh)')
ax.set_title('(a) BESS Capacity Sensitivity', fontweight='bold')
ax.legend()

ax = axes[1]
ax.plot(nuc_df['Nuclear_MW'] / 1000, nuc_df['Cost_MWh'],
        'o-', color='#4CAF50', lw=2, ms=7)
ax.axvline(2.256, color='red', ls='--', alpha=0.6,
           label='Diablo Canyon (2.26 GW)')
ax.set_xlabel('Nuclear Capacity (GW)')
ax.set_ylabel('System Cost ($/MWh)')
ax.set_title('(b) Nuclear Capacity Sensitivity', fontweight='bold')
ax.legend()

ax = axes[2]
ax.plot(gas_df['CCGT_Cost_MWh'], gas_df['Avg_Gas_Total_MW'] / 1000,
        'o-', color='#FF9800', lw=2, ms=7, label='Gas generation')
ax.plot(gas_df['CCGT_Cost_MWh'], gas_df['Avg_Import_MW'] / 1000,
        's--', color='#9C27B0', lw=2, ms=7, label='Imports')
ax.axvline(55, color='red', ls=':', alpha=0.7, label='Structural break (~$55)')
ax.annotate('Stage 1:\npeakers → imports', xy=(30, 12.6), xytext=(38, 13.8),
            fontsize=9, color='#555555',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1))
ax.annotate('Stage 2:\nCCGT → imports', xy=(55, 8.2), xytext=(63, 11.5),
            fontsize=9, color='#555555',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1))
ax.set_xlabel('CCGT Marginal Cost ($/MWh)')
ax.set_ylabel('Average Dispatch (GW)')
ax.set_title('(c) Gas Price Sensitivity', fontweight='bold')
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig15_sensitivity_analysis.png"))
plt.close()
print("  -> fig15_sensitivity_analysis.png")

# --- fig_co2_analysis ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
labels = [s.replace('_', '\n') for s in co2_df['Scenario']]
colors = ['#4CAF50', '#E53935', '#2196F3', '#9E9E9E', '#FF9800', '#9C27B0']

ax = axes[0]
bars = ax.bar(labels, co2_df['CO2_tonnes'] / 1000, color=colors,
              alpha=0.85, edgecolor='black', lw=0.5)
for b, v in zip(bars, co2_df['CO2_tonnes']):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
            f"{v/1000:,.0f}", ha='center', fontsize=9)
ax.set_ylabel('Weekly CO$_2$ Emissions (kt)')
ax.set_title('(a) CO$_2$ Emissions by Scenario (November week)',
             fontweight='bold')

ax = axes[1]
ax.scatter(co2_df['Cost_MWh'], co2_df['CO2_tonnes'] / 1000, s=160,
           c=colors, edgecolors='black', zorder=5)
for _, row in co2_df.iterrows():
    ax.annotate(row['Scenario'].replace('_', ' '),
                (row['Cost_MWh'], row['CO2_tonnes'] / 1000),
                textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.set_xlabel('System Cost ($/MWh)')
ax.set_ylabel('Weekly CO$_2$ Emissions (kt)')
ax.set_title('(b) Cost vs. Emissions Plane', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig_co2_analysis.png"))
plt.close()
print("  -> fig_co2_analysis.png")

# --- fig_robust_pareto ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(gam_df['gamma'], gam_df['cost_mwh'], 'o-', color='#E53935',
        lw=2, ms=9, label='System cost')
ax.set_xlabel('Robustness Parameter γ  (0 = deterministic, 1 = worst-case Q90)')
ax.set_ylabel('System Cost ($/MWh)', color='#E53935')
ax.tick_params(axis='y', labelcolor='#E53935')
ax2 = ax.twinx()
ax2.plot(gam_df['gamma'], gam_df['co2_tonnes'] / 1000, 's--',
         color='#607D8B', lw=2, ms=8, label='CO$_2$ emissions')
ax2.set_ylabel('Weekly CO$_2$ (kt)', color='#607D8B')
ax2.tick_params(axis='y', labelcolor='#607D8B')
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left')
ax.set_title('Cost–Robustness–Emissions Tradeoff (Two-Tier UC Model)',
             fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig_robust_pareto.png"))
plt.close()
print("  -> fig_robust_pareto.png")

print("\nDONE. All legacy single-tier results replaced with two-tier UC model.")
