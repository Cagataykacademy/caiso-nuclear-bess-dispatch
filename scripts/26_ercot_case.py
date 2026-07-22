"""
=============================================================================
 PHASE 8: ERCOT TRANSFERABILITY CASE STUDY
 Purpose: (i) show the net-load pipeline transfers to a second ISO with no
 code changes; (ii) exploit ERCOT's island topology (DC ties ~1.22 GW only)
 as a natural experiment for the paper's claim that scarce import/recourse
 flexibility amplifies the nuclear premium.

 ERCOT 2023 system parameters (sources in comments):
   Nuclear:  5,150 MW (Comanche Peak 2x~1,215 + South Texas 2x~1,280;
             ramp ~4.4%/h -> 225 MW/h), min 80%.
   Thermal-CC tier: 45,000 MW aggregate dispatchable mid-merit
             (gas CC ~35 GW + coal ~13.6 GW, EIA-860 2023), 9 x 5,000 MW
             UC units, $42/MWh blended.
   Peakers:  15,000 MW simple-cycle/steam GTs, $65/MWh.
   BESS:     3,200 MW / 6,400 MWh (~2 h fleet, ERCOT CDR 2023).
   DC ties:  1,220 MW import/export (Spp North/East + Railroad DC ties).
 Two weeks: deepest-spring-duck week and a November shoulder week
 (mirrors the CAISO analysis), each with and without nuclear.

 Outputs:
   - data/ercot_netload_2023.csv (cached fetch)
   - outputs/tables/ercot_case.csv
   - outputs/figures/fig_ercot_case.png
=============================================================================
"""
import os, time, importlib.util
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
P    = os.path.dirname(HERE)
DATA = os.path.join(P, "data")
TBL  = os.path.join(P, "outputs", "tables")
FIG  = os.path.join(P, "outputs", "figures")

API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")  # set your own key: https://www.eia.gov/opendata/register.php
CACHE = os.path.join(DATA, "ercot_netload_2023.csv")

# =========================================================================
# 1) FETCH (cached)
# =========================================================================
def fetch_series(url, facets, val_name):
    rows, offset = [], 0
    while True:
        params = {'api_key': API_KEY, 'frequency': 'hourly',
                  'data[0]': 'value',
                  'start': '2023-01-01T00', 'end': '2023-12-31T23',
                  'sort[0][column]': 'period', 'sort[0][direction]': 'asc',
                  'length': 5000, 'offset': offset}
        params.update(facets)
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()['response']['data']
        rows.extend(data)
        if len(data) < 5000:
            break
        offset += 5000
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    df['period'] = pd.to_datetime(df['period'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df[['period', 'value']].rename(columns={'value': val_name})

if os.path.exists(CACHE):
    ercot = pd.read_csv(CACHE, parse_dates=['period'])
    print(f"Loaded cached ERCOT data: {len(ercot)} rows")
else:
    print("Fetching ERCOT 2023 from EIA API...")
    base_fuel = 'https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/'
    base_reg  = 'https://api.eia.gov/v2/electricity/rto/region-data/data/'
    dem = fetch_series(base_reg,
        {'facets[respondent][]': 'ERCO', 'facets[type][]': 'D'}, 'demand_MW')
    sun = fetch_series(base_fuel,
        {'facets[respondent][]': 'ERCO', 'facets[fueltype][]': 'SUN'}, 'solar_MW')
    wnd = fetch_series(base_fuel,
        {'facets[respondent][]': 'ERCO', 'facets[fueltype][]': 'WND'}, 'wind_MW')
    ercot = dem.merge(sun, on='period').merge(wnd, on='period')
    ercot = ercot.dropna().sort_values('period').reset_index(drop=True)
    ercot['net_load_MW'] = (ercot['demand_MW'] - ercot['solar_MW']
                            - ercot['wind_MW'])
    ercot.to_csv(CACHE, index=False)
    print(f"  fetched {len(ercot)} hourly rows -> ercot_netload_2023.csv")

nl = ercot['net_load_MW']
print(f"\nERCOT 2023 net load: mean={nl.mean():,.0f}  min={nl.min():,.0f}  "
      f"max={nl.max():,.0f}  std={nl.std():,.0f} MW")
print(f"  demand peak={ercot['demand_MW'].max():,.0f} MW, "
      f"wind max={ercot['wind_MW'].max():,.0f}, solar max={ercot['solar_MW'].max():,.0f}")

# =========================================================================
# 2) ERCOT-PARAMETERIZED MILP (override module-16 globals)
# =========================================================================
spec = importlib.util.spec_from_file_location(
    "improved_milp", os.path.join(HERE, "16_improved_milp.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

# --- ERCOT fleet ---
M.NUC_RAMP = 225          # 5,150 MW x ~4.4%/h
M.N_CCGT = 9              # 9 x 5,000 = 45 GW aggregate mid-merit thermal
M.CCGT_COST = 42          # blended gas-CC + coal variable cost
M.PEAK_CAP = 15000
M.IMP_CAP = 1220          # DC ties only (island grid)
M.EXP_CAP = 1220
M.CURT_CAP = 30000
M.SHED_CAP = 10000
ERCOT_NUC, ERCOT_NUC_MIN = 5150, 4120
ERCOT_BESS_POW, ERCOT_BESS_ENE = 3200, 6400

# --- week selection: deepest spring duck + November shoulder ---
ercot['date'] = ercot['period'].dt.date
idx_min = nl.idxmin()
spring_start = (idx_min // 24) * 24 - 72   # centre the min day in its week
spring_start = max(0, min(spring_start, len(ercot) - 168))
nov_mask = ercot['period'] >= '2023-11-01'
nov_start = ercot.index[nov_mask][0]

weeks = {
    'Spring (deepest duck)': ercot.iloc[spring_start:spring_start+168],
    'November (shoulder)':   ercot.iloc[nov_start:nov_start+168],
}

rows, disps = [], {}
for wname, wdf in weeks.items():
    prof = wdf['net_load_MW'].values
    for use_nuc, tag in [(True, 'with nuclear'), (False, 'no nuclear')]:
        sm, d = M.build_and_solve(
            prof, f"ERCOT_{wname}_{tag}",
            nuc_cap=ERCOT_NUC, nuc_min=ERCOT_NUC_MIN, use_nuc=use_nuc,
            bess_pow=ERCOT_BESS_POW, bess_ene=ERCOT_BESS_ENE)
        if d is None:
            print(f"  {wname} / {tag}: INFEASIBLE"); continue
        disps[(wname, tag)] = d
        rows.append({'Week': wname, 'Case': tag,
                     'Start': str(wdf['period'].iloc[0].date()),
                     'NL_Mean_MW': round(prof.mean(), 0),
                     'NL_Min_MW': round(prof.min(), 0),
                     'NL_Max_MW': round(prof.max(), 0),
                     'Cost_MWh': round(d['cost_per_mwh'], 2),
                     'Shed_MWh': round(d['shed_total'], 0),
                     'Avg_Peaker_MW': round(d['avg_peak'], 0),
                     'Avg_Import_MW': round(d['avg_imp'], 0),
                     'Solve_s': round(sm['solve_time_s'], 2)})
        print(f"  {wname:<22} {tag:<13} ${d['cost_per_mwh']:6.2f}/MWh  "
              f"shed={d['shed_total']:7.0f}  peak={d['avg_peak']:6.0f}  "
              f"t={sm['solve_time_s']:.1f}s")

res = pd.DataFrame(rows)
# premiums
for wname in weeks:
    c1 = res[(res.Week == wname) & (res.Case == 'with nuclear')]['Cost_MWh']
    c0 = res[(res.Week == wname) & (res.Case == 'no nuclear')]['Cost_MWh']
    if len(c1) and len(c0):
        prem = (c0.iloc[0] - c1.iloc[0]) / c1.iloc[0] * 100
        print(f"  {wname}: nuclear premium = +{prem:.1f}%")
        res.loc[res.Week == wname, 'Premium_pct'] = round(prem, 1)
res.to_csv(os.path.join(TBL, "ercot_case.csv"), index=False)
print("  -> ercot_case.csv")

# =========================================================================
# 3) FIGURE: ERCOT duck curve + dispatch comparison
# =========================================================================
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

ax = axes[0]
spring_w = weeks['Spring (deepest duck)']
hrs = np.arange(168)
ax.plot(hrs, spring_w['demand_MW'].values/1000, '-', color='#9E9E9E',
        lw=1.5, label='Demand')
ax.plot(hrs, spring_w['net_load_MW'].values/1000, 'k-', lw=2, label='Net load')
ax.fill_between(hrs, spring_w['net_load_MW'].values/1000,
                spring_w['demand_MW'].values/1000, alpha=0.25,
                color='#FFC107', label='Solar + wind')
ax.set_xlabel('Hour of Week'); ax.set_ylabel('GW')
ax.set_title(f"(a) ERCOT Spring Week ({res['Start'].iloc[0]}): "
             'Wind-Driven "Duck"', fontweight='bold')
ax.legend()

ax = axes[1]
labels, prem_vals = [], []
caiso_spring_prem = 24.4   # placeholder replaced below if computable
for wname in weeks:
    sub = res[res.Week == wname]
    if 'Premium_pct' in sub:
        labels.append(wname.split(' (')[0])
        prem_vals.append(sub['Premium_pct'].iloc[0])
bars = ax.bar(labels, prem_vals, color=['#1565C0', '#6A1B9A'],
              alpha=0.85, edgecolor='black', lw=0.5, width=0.5)
ax.axhline(13.2, color='#4CAF50', ls='--', lw=1.8,
           label='CAISO November (+13.2%)')
for b, v in zip(bars, prem_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'+{v:.1f}%',
            ha='center', fontweight='bold')
ax.set_ylabel('Nuclear Removal Premium (%)')
ax.set_title('(b) Nuclear Premium: ERCOT (island) vs CAISO (interconnected)',
             fontweight='bold')
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ercot_case.png"))
plt.close()
print("  -> fig_ercot_case.png")
