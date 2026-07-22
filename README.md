# Duck Curve Dispatch under Uncertainty: Conformalized ML + Robust MILP (CAISO 2023)

Reproduction package for the manuscript *"Data-Driven Nuclear Baseload and Battery
Storage Dispatch Optimization under Duck Curve Uncertainty: A Conformalized Machine
Learning Approach"* by Cagatay Kuban (submitted to Applied Energy, Elsevier).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21501015.svg)](https://doi.org/10.5281/zenodo.21501015)

## What this does

An end-to-end pipeline on real CAISO 2022–2023 data:

1. **Data** — hourly demand, generation-by-fuel, and interchange from the EIA Form 930
   API; SP15 day-ahead prices (EIA/ICE); Henry Hub gas prices; NOAA ISD-Lite temperatures.
2. **Net load construction** — `demand − solar − wind` from metered fuel-type generation
   (range −10,369 to +44,900 MW in 2023).
3. **Day-ahead forecasting** — LightGBM / XGBoost / Random Forest / LSTM / MLP vs.
   persistence, 35 leakage-free features, Diebold–Mariano tests, expanding-window CV,
   2022 out-of-sample validation.
4. **Uncertainty** — Conformalized Quantile Regression (90% target, 88.8% empirical
   coverage; Winkler-score comparison against uncalibrated QR).
5. **Dispatch** — 168-h MILP (Pyomo + HiGHS): two-tier gas fleet (CCGT with unit
   commitment + peakers), Diablo Canyon nuclear, 5 GW/20 GWh BESS; CQR-driven
   budget-of-uncertainty robustness parameter γ; four-season analysis; CO₂ accounting;
   sensitivity sweeps (nuclear, BESS, gas price, import price); two-zone validation.

## Setup

```bash
pip install -r requirements.txt
```

An EIA API key (free, https://www.eia.gov/opendata/register.php) is required for the
data-acquisition scripts; set it as an environment variable before running them:

```bash
export EIA_API_KEY="your-key-here"
```

All downstream scripts (ML, CQR, MILP, sensitivity sweeps) run entirely from the CSVs
already provided in `data/` — the API key is only needed if you want to re-fetch raw
data from scratch.

## Reproduction order

| Step | Script | Output |
|------|--------|--------|
| Data acquisition | `scripts/01j_fetch_final.py`, `01k`, `01l`, `12_fetch_2022.py` | `data/*.csv` |
| Preprocessing & split | `scripts/03b_preprocess_real.py` | `data/train/val/test_2023.csv`, `feature_config.json` |
| ML benchmark + CQR + predictions | `scripts/09_fix_weaknesses.py`, `scripts/11_fix_all_weaknesses.py`, `scripts/13_lstm_price_milp.py` | `outputs/tables/benchmark_final.csv`, `data/ml_predictions_for_milp.csv`, `hyperparameter_sensitivity.csv` |
| Statistical tests & CV | `scripts/08_statistical_tests.py` | DM tests, CV tables |
| 2022 out-of-sample | `scripts/14_final_2022_validation.py` | `out_of_sample_2022.csv` |
| **Improved MILP (canonical)** | `scripts/16_improved_milp.py` | scenario & nuclear-sweep tables, dispatch figures |
| Four-season MILP | `scripts/17_seasonal_milp.py` | `seasonal_dispatch.csv` |
| γ sweep | `scripts/18_gamma_sweep.py` | `gamma_sweep.csv` |
| Two-zone validation | `scripts/19_zonal_sensitivity.py` | `zonal_sensitivity.csv` |
| Consistency sweeps (BESS / gas price / CO₂ / γ-CO₂) | `scripts/20_consistency_sweeps.py` | sensitivity tables + figs 15, CO₂, Pareto |
| QR-vs-CQR + import-price sensitivity | `scripts/21_reviewer_additions.py` | `qr_vs_cqr.csv`, `sensitivity_import_price.csv` |
| Graphical abstract | `scripts/22_graphical_abstract.py` | `outputs/figures/graphical_abstract.png` |
| Conformal diagnostics + import figures | `scripts/23_extra_figures.py` | `fig_conformal_diagnostics.png`, `fig_import_sensitivity.png` |
| Two-stage SP baseline + out-of-sample plan evaluation | `scripts/24_sp_baseline.py` | `sp_vs_robust.csv` |
| Full-year (50-week) dispatch sweep | `scripts/25_full_year_dispatch.py` | `full_year_dispatch.csv`, `full_year_summary.csv`, `fig_fullyear_dispatch.png` |
| ERCOT transferability check (Appendix B) | `scripts/26_ercot_case.py` | `data/ercot_netload_2023.csv`, `ercot_case.csv`, `fig_ercot_case.png` |
| Robustness: realized-hydro re-solve | `scripts/28_hydro_robustness.py` | `hydro_robustness.csv`, `fig_hydro_robustness.png` |
| Robustness: CCGT aggregation granularity | `scripts/29_ccgt_granularity.py` | `ccgt_granularity.csv` |
| Canonical feature importance (Fig. A.9) | `scripts/27_feature_importance_fix.py` | `feature_importance_netload_final.csv`, `fig09_feature_importance.png` |
| Overview/benchmark/PI figures | `scripts/07_final_figures.py` | `fig02`, `fig03`, `fig06`, `fig07` |

The manuscript itself is not included in this repository; it is under review at Applied Energy and will be linked here upon publication.

> **Note:** this repository contains only the final version of each script, dataset,
> figure, and results table used by the manuscript. Earlier iterations were removed
> for clarity. Rerunning `scripts/07_final_figures.py` regenerates a few exploratory
> figures beyond those used in the paper; this is harmless.

## Data sources (all public)

- EIA Form 930 Hourly Electric Grid Monitor — https://api.eia.gov
- EIA/ICE wholesale electricity prices (SP15 DA LMP peak)
- NOAA ISD-Lite (stations: LAX, SFO, FAT) — https://www.ncei.noaa.gov
- Emission factors: EPA eGRID 2022; CARB MRR default for unspecified imports (0.428 t/MWh)

## License

Code: MIT. Data: subject to the terms of the respective providers (EIA, NOAA).
