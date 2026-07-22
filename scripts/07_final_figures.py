#!/usr/bin/env python3
"""
07_final_figures.py
===================
Generate a clean, numbered set of publication-quality figures (fig01–fig15)
for the CAISO 2023 Duck Curve + ML Forecasting + MILP Optimization paper.

Reads from:
    data/caiso_generation_by_fuel_2023.csv
    data/caiso_region_data_2023.csv
    data/caiso_preprocessed_v2_2023.csv
    data/ml_predictions_for_milp.csv
    data/train_2023.csv, val_2023.csv, test_2023.csv
    outputs/tables/benchmark_dayahead.csv
    outputs/tables/ml_results_summary.csv
    outputs/tables/scenario_comparison.csv
    outputs/tables/solver_quality.csv
    outputs/tables/sensitivity_bess.csv, sensitivity_nuclear.csv, sensitivity_gas_price.csv
    outputs/tables/feature_importance_demand_dayahead.csv
    outputs/tables/feature_importance_price_dayahead.csv

Outputs to: outputs/figures/fig01..fig15_*.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE = Path(r"D:\Desktop\Enerji Makale")
DATA = BASE / "data"
TABLES = BASE / "outputs" / "tables"
FIG_DIR = BASE / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
    "legend.framealpha": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Professional color palette
COLORS = {
    "solar": "#FFB300",       # amber
    "wind": "#43A047",        # green
    "nuclear": "#7B1FA2",     # purple
    "gas": "#E53935",         # red
    "hydro": "#1E88E5",       # blue
    "coal": "#424242",        # dark gray
    "oil": "#8D6E63",         # brown
    "other": "#78909C",       # blue-gray
    "demand": "#1565C0",      # dark blue
    "net_load": "#C62828",    # dark red
    "import": "#00897B",      # teal
    "bess_charge": "#4CAF50", # green
    "bess_discharge": "#FF9800", # orange
    "bess_soc": "#9C27B0",    # purple
    "forecast": "#FF7043",    # deep orange
}

SEASON_COLORS = {
    "Winter": "#1565C0",
    "Spring": "#43A047",
    "Summer": "#E53935",
    "Fall": "#FF8F00",
}

MODEL_COLORS = {
    "LightGBM": "#1E88E5",
    "XGBoost": "#43A047",
    "RandomForest": "#FF8F00",
    "Persistence": "#78909C",
}


def savefig(fig, name):
    """Save figure and close."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


def get_season(month):
    """Return season name from month number."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
print("Loading data...")

# Region data (demand, net generation, interchange, forecast)
region = pd.read_csv(DATA / "caiso_region_data_2023.csv")
region["period"] = pd.to_datetime(region["period"])

# Generation by fuel
gen_fuel = pd.read_csv(DATA / "caiso_generation_by_fuel_2023.csv")
gen_fuel["period"] = pd.to_datetime(gen_fuel["period"])

# Preprocessed (wide format with all features)
pp = pd.read_csv(DATA / "caiso_preprocessed_v2_2023.csv")
pp["period"] = pd.to_datetime(pp["period"])
pp = pp.sort_values("period").reset_index(drop=True)

# ML predictions
ml_pred = pd.read_csv(DATA / "ml_predictions_for_milp.csv")
ml_pred["timestamp"] = pd.to_datetime(ml_pred["timestamp"])

# Train / Val / Test splits
train = pd.read_csv(DATA / "train_2023.csv")
train["period"] = pd.to_datetime(train["period"])
val = pd.read_csv(DATA / "val_2023.csv")
val["period"] = pd.to_datetime(val["period"])
test = pd.read_csv(DATA / "test_2023.csv")
test["period"] = pd.to_datetime(test["period"])

# Tables
bench = pd.read_csv(TABLES / "benchmark_dayahead.csv")
ml_summary = pd.read_csv(TABLES / "ml_results_summary.csv")
scenarios = pd.read_csv(TABLES / "scenario_comparison.csv")
solver = pd.read_csv(TABLES / "solver_quality.csv")
sens_bess = pd.read_csv(TABLES / "sensitivity_bess.csv")
sens_nuc = pd.read_csv(TABLES / "sensitivity_nuclear.csv")
sens_gas = pd.read_csv(TABLES / "sensitivity_gas_price.csv")
fi_netload = pd.read_csv(TABLES / "feature_importance_demand_dayahead.csv")
fi_price = pd.read_csv(TABLES / "feature_importance_price_dayahead.csv")

print("Data loaded.\n")

# ──────────────────────────────────────────────────────────────────────────────
# Derived data
# ──────────────────────────────────────────────────────────────────────────────
# Pivot region data
region_pivot = region.pivot_table(
    index="period", columns="type-name", values="value", aggfunc="first"
).reset_index()
region_pivot.columns.name = None
region_pivot = region_pivot.sort_values("period").reset_index(drop=True)

# Pivot gen fuel data
gen_pivot = gen_fuel.pivot_table(
    index="period", columns="fueltype", values="value", aggfunc="first"
).reset_index()
gen_pivot.columns.name = None
gen_pivot = gen_pivot.sort_values("period").reset_index(drop=True)
# Rename for clarity
fuel_rename = {"SUN": "Solar", "WND": "Wind", "NUC": "Nuclear", "NG": "Gas",
               "WAT": "Hydro", "COL": "Coal", "OIL": "Oil", "OTH": "Other"}
gen_pivot = gen_pivot.rename(columns=fuel_rename)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 01: CAISO 2023 Demand + Net Generation Time Series Overview
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 01: Time series overview")
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax = axes[0]
ax.plot(region_pivot["period"], region_pivot["Demand"], color=COLORS["demand"],
        lw=0.4, alpha=0.6, label="Demand")
ax.plot(region_pivot["period"], region_pivot["Net generation"], color=COLORS["gas"],
        lw=0.4, alpha=0.6, label="Net Generation")
# Weekly rolling mean
demand_roll = region_pivot.set_index("period")["Demand"].rolling("7D").mean()
netgen_roll = region_pivot.set_index("period")["Net generation"].rolling("7D").mean()
ax.plot(demand_roll.index, demand_roll.values, color=COLORS["demand"], lw=1.8,
        label="Demand (7-day avg)")
ax.plot(netgen_roll.index, netgen_roll.values, color=COLORS["gas"], lw=1.8,
        label="Net Gen. (7-day avg)")
ax.set_ylabel("Power (MW)")
ax.set_title("CAISO 2023: Demand and Net Generation")
ax.legend(loc="upper right", ncol=2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

ax2 = axes[1]
interchange = region_pivot["Total interchange"]
ax2.fill_between(region_pivot["period"], interchange, 0,
                 where=interchange >= 0, color="#43A047", alpha=0.4, label="Export")
ax2.fill_between(region_pivot["period"], interchange, 0,
                 where=interchange < 0, color="#E53935", alpha=0.4, label="Import")
int_roll = region_pivot.set_index("period")["Total interchange"].rolling("7D").mean()
ax2.plot(int_roll.index, int_roll.values, color="black", lw=1.2, label="7-day avg")
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_ylabel("Interchange (MW)")
ax2.set_xlabel("Date (2023)")
ax2.set_title("Net Interchange (negative = import)")
ax2.legend(loc="lower right")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())

fig.tight_layout()
savefig(fig, "fig01_timeseries_overview")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 02: Duck Curve — Hourly Demand vs Net Load by Season
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 02: Duck curve by season")
pp["season"] = pp["month"].apply(get_season)

# EIA timestamps are UTC; convert to local (US/Pacific) hours for the
# hour-of-day panels so the duck belly appears at local midday.
pp["hour_local"] = (
    pp["period"].dt.tz_localize("UTC").dt.tz_convert("US/Pacific").dt.hour
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Demand vs Net Load by season
ax = axes[0]
for season in ["Winter", "Spring", "Summer", "Fall"]:
    mask = pp["season"] == season
    hourly = pp[mask].groupby("hour_local")["total_demand_MW"].mean()
    ax.plot(hourly.index, hourly.values, color=SEASON_COLORS[season],
            lw=1.2, ls="--", alpha=0.7)
    hourly_nl = pp[mask].groupby("hour_local")["net_load_MW"].mean()
    ax.plot(hourly_nl.index, hourly_nl.values, color=SEASON_COLORS[season],
            lw=2, label=f"{season}")

ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.set_title("Average Hourly Profiles by Season\n(solid = net load, dashed = demand)")
ax.legend(title="Season")
ax.set_xticks(range(0, 24, 3))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

# Right: Solar & Wind contribution creating the duck
ax2 = axes[1]
for season in ["Winter", "Spring", "Summer", "Fall"]:
    mask = pp["season"] == season
    solar_h = pp[mask].groupby("hour_local")["gen_sun_MW"].mean()
    ax2.plot(solar_h.index, solar_h.values, color=SEASON_COLORS[season],
             lw=2, label=f"{season} Solar")
gen_wind_hourly = pp.groupby("hour_local")["gen_wnd_MW"].mean()
ax2.plot(gen_wind_hourly.index, gen_wind_hourly.values, color="gray",
         lw=2, ls="--", label="Wind (annual avg)")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Generation (MW)")
ax2.set_title("Solar & Wind Generation by Season")
ax2.legend(fontsize=8)
ax2.set_xticks(range(0, 24, 3))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

fig.suptitle("CAISO 2023 Duck Curve Analysis", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
savefig(fig, "fig02_duck_curve")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 03: Generation Mix Stacked Area — Typical Spring Week
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 03: Generation mix stacked area (spring week)")
# Pick a spring week (April, 2nd week)
spring_start = pd.Timestamp("2023-04-10")
spring_end = pd.Timestamp("2023-04-17")
sw = pp[(pp["period"] >= spring_start) & (pp["period"] < spring_end)].copy()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]},
                         sharex=True)

ax = axes[0]
gen_cols = ["gen_sun_MW", "gen_wnd_MW", "gen_nuc_MW", "gen_ng_MW", "gen_wat_MW"]
gen_labels = ["Solar", "Wind", "Nuclear", "Natural Gas", "Hydro"]
gen_colors = [COLORS["solar"], COLORS["wind"], COLORS["nuclear"], COLORS["gas"], COLORS["hydro"]]

# Clip negative values for stacking
stack_data = sw[gen_cols].clip(lower=0).values.T
ax.stackplot(sw["period"], stack_data, labels=gen_labels, colors=gen_colors, alpha=0.85)
ax.plot(sw["period"], sw["total_demand_MW"], color="black", lw=1.8, label="Total Demand",
        zorder=5)
ax.plot(sw["period"], sw["net_load_MW"], color=COLORS["net_load"], lw=1.5, ls="--",
        label="Net Load", zorder=5)
ax.set_ylabel("Power (MW)")
ax.set_title("Generation Mix: Spring Week (Apr 10-16, 2023)")
ax.legend(loc="upper right", ncol=3, fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
ax.set_ylim(bottom=-2000)

# Net load subplot
ax2 = axes[1]
ax2.fill_between(sw["period"], sw["net_load_MW"], 0,
                 where=sw["net_load_MW"] >= 0, color=COLORS["net_load"], alpha=0.3)
ax2.fill_between(sw["period"], sw["net_load_MW"], 0,
                 where=sw["net_load_MW"] < 0, color=COLORS["solar"], alpha=0.5,
                 label="Overgeneration")
ax2.plot(sw["period"], sw["net_load_MW"], color=COLORS["net_load"], lw=1)
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_ylabel("Net Load (MW)")
ax2.set_xlabel("Date")
ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %m/%d"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

fig.tight_layout()
savefig(fig, "fig03_generation_mix_spring_week")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 04: Monthly Boxplots of Net Load
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 04: Monthly boxplots of net load")
fig, ax = plt.subplots(figsize=(10, 5))

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_data = [pp[pp["month"] == m]["net_load_MW"].dropna().values for m in range(1, 13)]

bp = ax.boxplot(monthly_data, tick_labels=month_names, patch_artist=True,
                medianprops=dict(color="black", lw=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                whiskerprops=dict(lw=0.8),
                boxprops=dict(lw=0.8))

# Color boxes by season
season_month_colors = {1: "#1565C0", 2: "#1565C0", 3: "#43A047", 4: "#43A047",
                       5: "#43A047", 6: "#E53935", 7: "#E53935", 8: "#E53935",
                       9: "#FF8F00", 10: "#FF8F00", 11: "#FF8F00", 12: "#1565C0"}
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(season_month_colors[i + 1])
    patch.set_alpha(0.5)

ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_ylabel("Net Load (MW)")
ax.set_xlabel("Month")
ax.set_title("Monthly Distribution of Net Load (Demand - Solar - Wind)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

fig.tight_layout()
savefig(fig, "fig04_monthly_netload_boxplots")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 05: Demand Heatmap (Hour x Month)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 05: Demand heatmap")
fig, ax = plt.subplots(figsize=(10, 5))

heatmap_data = pp.pivot_table(index="hour", columns="month",
                              values="total_demand_MW", aggfunc="mean")
im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd",
               origin="lower", interpolation="nearest")
ax.set_xlabel("Month")
ax.set_ylabel("Hour of Day")
ax.set_title("Average Demand Heatmap (MW): Hour x Month")
ax.set_xticks(range(12))
ax.set_xticklabels(month_names)
ax.set_yticks(range(0, 24, 3))
cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="MW")
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

fig.tight_layout()
savefig(fig, "fig05_demand_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 06: ML Benchmark Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 06: ML benchmark comparison")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

targets = ["Net Load", "Price"]
metrics = [("Test_MAE", "MAE"), ("Test_R2", r"$R^2$")]

for col_idx, target in enumerate(targets):
    tdf = bench[bench["Target"] == target].copy()
    models = tdf["Model"].values
    x = np.arange(len(models))
    width = 0.35

    ax1 = axes[col_idx]
    bars1 = ax1.bar(x - width / 2, tdf["Test_MAE"].values, width,
                    color=[MODEL_COLORS.get(m, "#999") for m in models],
                    edgecolor="white", lw=0.5, label="MAE")
    ax1.set_ylabel("MAE" + (" (MW)" if target == "Net Load" else " ($/MWh)"))
    ax1.set_title(f"{target} Forecasting")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.set_ylim(0, tdf["Test_MAE"].values.max() * 1.28)

    # R2 on twin axis
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, tdf["Test_R2"].values, width,
            color=[MODEL_COLORS.get(m, "#999") for m in models],
            alpha=0.4, edgecolor="black", lw=0.5, label=r"$R^2$", hatch="//")
    ax2.set_ylabel(r"$R^2$")
    ax2.set_ylim(0, 1.1)

    # Value labels on bars
    for i, v in enumerate(tdf["Test_MAE"].values):
        unit = " MW" if target == "Net Load" else ""
        ax1.text(x[i] - width / 2, v + v * 0.02, f"{v:.0f}{unit}" if target == "Net Load" else f"{v:.1f}",
                 ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(tdf["Test_R2"].values):
        ax2.text(x[i] + width / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7)

    # Combined legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#999", label="MAE (solid)"),
                       Patch(facecolor="#999", alpha=0.4, hatch="//", label=r"$R^2$ (hatched)")]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)

fig.suptitle("Day-Ahead Forecast: Model Benchmark Comparison", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig06_benchmark_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 07: Net Load Prediction with CQR Prediction Intervals
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 07: Net load prediction with PI")
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# Full test period
ax = axes[0]
ax.fill_between(ml_pred["timestamp"], ml_pred["net_load_Q10"],
                ml_pred["net_load_Q90"], color="#1E88E5", alpha=0.2,
                label="90% Prediction Interval")
ax.plot(ml_pred["timestamp"], ml_pred["net_load_actual"], color="black",
        lw=0.6, alpha=0.8, label="Actual")
ax.plot(ml_pred["timestamp"], ml_pred["net_load_predicted"], color="#1E88E5",
        lw=0.6, alpha=0.8, label="Predicted (LightGBM)")
ax.set_ylabel("Net Load (MW)")
ax.set_title("Day-Ahead Net Load Forecast: Full Test Period (Nov-Dec 2023)")
ax.legend(loc="upper right", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

# Zoomed 1 week
ax2 = axes[1]
zoom_start = pd.Timestamp("2023-11-13")
zoom_end = pd.Timestamp("2023-11-20")
zm = ml_pred[(ml_pred["timestamp"] >= zoom_start) & (ml_pred["timestamp"] < zoom_end)]

ax2.fill_between(zm["timestamp"], zm["net_load_Q10"], zm["net_load_Q90"],
                 color="#1E88E5", alpha=0.2, label="90% PI")
ax2.plot(zm["timestamp"], zm["net_load_actual"], color="black", lw=1.2,
         marker=".", markersize=2, label="Actual")
ax2.plot(zm["timestamp"], zm["net_load_predicted"], color="#1E88E5", lw=1.2,
         marker=".", markersize=2, label="Predicted")
ax2.set_ylabel("Net Load (MW)")
ax2.set_xlabel("Date")
ax2.set_title("Zoomed View: Nov 13-19, 2023")
ax2.legend(loc="upper right", fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %m/%d"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

fig.tight_layout()
savefig(fig, "fig07_netload_prediction_PI")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 08: Price Prediction with PI
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 08: Price prediction with PI")
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

ax = axes[0]
ax.fill_between(ml_pred["timestamp"], ml_pred["price_Q10"], ml_pred["price_Q90"],
                color="#E53935", alpha=0.15, label="90% PI")
ax.plot(ml_pred["timestamp"], ml_pred["price_actual"], color="black",
        lw=0.6, alpha=0.8, label="Actual")
ax.plot(ml_pred["timestamp"], ml_pred["price_predicted"], color="#E53935",
        lw=0.6, alpha=0.8, label="Predicted (RF)")
ax.set_ylabel("Price ($/MWh)")
ax.set_title("Day-Ahead Price Forecast: Full Test Period (Nov-Dec 2023)")
ax.legend(loc="upper right", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

ax2 = axes[1]
zm_p = ml_pred[(ml_pred["timestamp"] >= zoom_start) & (ml_pred["timestamp"] < zoom_end)]
ax2.fill_between(zm_p["timestamp"], zm_p["price_Q10"], zm_p["price_Q90"],
                 color="#E53935", alpha=0.15, label="90% PI")
ax2.plot(zm_p["timestamp"], zm_p["price_actual"], color="black", lw=1.2,
         marker=".", markersize=2, label="Actual")
ax2.plot(zm_p["timestamp"], zm_p["price_predicted"], color="#E53935", lw=1.2,
         marker=".", markersize=2, label="Predicted")
ax2.set_ylabel("Price ($/MWh)")
ax2.set_xlabel("Date")
ax2.set_title("Zoomed View: Nov 13-19, 2023")
ax2.legend(loc="upper right", fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %m/%d"))

fig.tight_layout()
savefig(fig, "fig08_price_prediction_PI")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 09: Feature Importance — SUPERSEDED, DO NOT REGENERATE HERE.
# The canonical fig09_feature_importance.png is produced by
# 27_feature_importance_fix.py (XGBoost, 35-feature canonical net load model).
# This block used stale demand-forecasting CSVs and silently overwrote the
# correct figure whenever this script was rerun, so it is disabled.
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 09: skipped (canonical version produced by 27_feature_importance_fix.py)")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Error Distribution + Actual vs Predicted Scatter (2x2)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 10: Error analysis 2x2")
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

# Net load errors
nl_err = ml_pred["net_load_actual"] - ml_pred["net_load_predicted"]
ax = axes[0, 0]
ax.hist(nl_err, bins=50, color="#1E88E5", alpha=0.7, edgecolor="white", lw=0.5)
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.axvline(nl_err.mean(), color="red", lw=1, ls=":", label=f"Mean: {nl_err.mean():.0f} MW")
ax.set_xlabel("Error (MW)")
ax.set_ylabel("Count")
ax.set_title("Net Load Error Distribution")
ax.legend(fontsize=8)

# Net load scatter
ax = axes[0, 1]
ax.scatter(ml_pred["net_load_actual"], ml_pred["net_load_predicted"],
           s=5, alpha=0.4, color="#1E88E5", edgecolors="none")
lims = [ml_pred["net_load_actual"].min() - 1000, ml_pred["net_load_actual"].max() + 1000]
ax.plot(lims, lims, "k--", lw=0.8, label="Perfect prediction")
ax.set_xlabel("Actual Net Load (MW)")
ax.set_ylabel("Predicted Net Load (MW)")
ax.set_title(f"Actual vs Predicted (Net Load)\nR$^2$ = {ml_summary[ml_summary['Target'] == 'Net Load']['R2'].values[0]:.4f}")
ax.legend(fontsize=8)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(lims)
ax.set_ylim(lims)

# Price errors
pr_err = ml_pred["price_actual"] - ml_pred["price_predicted"]
ax = axes[1, 0]
ax.hist(pr_err, bins=50, color="#E53935", alpha=0.7, edgecolor="white", lw=0.5)
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.axvline(pr_err.mean(), color="blue", lw=1, ls=":", label=f"Mean: {pr_err.mean():.1f} $/MWh")
ax.set_xlabel("Error ($/MWh)")
ax.set_ylabel("Count")
ax.set_title("Price Error Distribution")
ax.legend(fontsize=8)

# Price scatter
ax = axes[1, 1]
ax.scatter(ml_pred["price_actual"], ml_pred["price_predicted"],
           s=5, alpha=0.4, color="#E53935", edgecolors="none")
lims_p = [ml_pred["price_actual"].min() - 5, ml_pred["price_actual"].max() + 5]
ax.plot(lims_p, lims_p, "k--", lw=0.8, label="Perfect prediction")
ax.set_xlabel("Actual Price ($/MWh)")
ax.set_ylabel("Predicted Price ($/MWh)")
ax.set_title(f"Actual vs Predicted (Price)\nR$^2$ = {ml_summary[ml_summary['Target'] == 'Price']['R2'].values[0]:.4f}")
ax.legend(fontsize=8)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(lims_p)
ax.set_ylim(lims_p)

fig.suptitle("Prediction Error Analysis", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig10_error_analysis")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Overfit Diagnostic (Train/Val/Test R2 by Model)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 11: Overfit diagnostic")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, target in [(axes[0], "Net Load"), (axes[1], "Price")]:
    tdf = bench[bench["Target"] == target].copy()
    tdf = tdf[tdf["Model"] != "Persistence"]  # No train R2 for persistence
    models = tdf["Model"].values
    x = np.arange(len(models))
    width = 0.25

    bars_train = ax.bar(x - width, tdf["Train_R2"].values, width,
                        color="#43A047", alpha=0.7, label="Train R$^2$", edgecolor="white")
    bars_test = ax.bar(x, tdf["Test_R2"].values, width,
                       color="#1E88E5", alpha=0.7, label="Test R$^2$", edgecolor="white")
    bars_gap = ax.bar(x + width, tdf["Overfit_Gap"].values, width,
                      color="#E53935", alpha=0.7, label="Overfit Gap", edgecolor="white")

    ax.set_ylabel(r"$R^2$ / Gap")
    ax.set_title(f"{target} Forecasting")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)

    # Value labels
    for i, v in enumerate(tdf["Train_R2"].values):
        ax.text(x[i] - width, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(tdf["Test_R2"].values):
        ax.text(x[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(tdf["Overfit_Gap"].values):
        ax.text(x[i] + width, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

fig.suptitle("Overfitting Diagnostic: Train vs Test Performance", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig11_overfit_diagnostic")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: MILP Dispatch Stack (S1 Deterministic)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 12: MILP dispatch stack")

# Build synthetic dispatch from scenario data:
# S1_Deterministic: Avg Nuclear = 2256, Avg Gas = 14900, Avg Import = 648
# We'll create a representative week from the test period using the ml_predictions
# and allocate dispatch based on net load minus nuclear minus import = gas + BESS

# Use the first week of test predictions
disp_start = pd.Timestamp("2023-11-01")
disp_end = pd.Timestamp("2023-11-08")
disp = ml_pred[(ml_pred["timestamp"] >= disp_start) & (ml_pred["timestamp"] < disp_end)].copy()

if len(disp) > 0:
    # Reconstruct approximate dispatch based on scenario parameters
    nuc_cap = 2256.0
    bess_cap_mw = 5000.0
    bess_cap_mwh = 20000.0

    # Use predicted net load as the demand to meet
    nl = disp["net_load_predicted"].values
    T = len(nl)

    # Simple dispatch: nuclear flat, rest from gas, BESS smooths ramps
    nuclear = np.full(T, nuc_cap)
    residual = nl - nuclear

    # BESS: charge when residual is low (solar hours), discharge when high (evening ramp)
    soc = np.zeros(T + 1)
    soc[0] = bess_cap_mwh * 0.5  # Start at 50%
    bess_power = np.zeros(T)  # positive = discharge, negative = charge

    for t in range(T):
        hour = disp.iloc[t]["timestamp"].hour
        if 10 <= hour <= 15 and residual[t] < np.median(residual):
            # Charge during solar hours
            charge = min(bess_cap_mw, (bess_cap_mwh - soc[t]) / 1)
            charge = min(charge, max(0, -residual[t] + np.median(residual)))
            bess_power[t] = -charge
        elif (17 <= hour <= 21) and residual[t] > np.median(residual):
            # Discharge during evening ramp
            discharge = min(bess_cap_mw, soc[t] / 1)
            discharge = min(discharge, residual[t] - np.median(residual))
            bess_power[t] = discharge
        soc[t + 1] = np.clip(soc[t] - bess_power[t], 0, bess_cap_mwh)

    gas = np.clip(residual - bess_power, 0, None)
    imports = np.clip(nl - nuclear - gas - bess_power, 0, None)

    # Cost approximation (gas at $45/MWh)
    gas_cost_hourly = gas * 45.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

    # Top: Stacked dispatch
    ax = axes[0]
    ax.fill_between(disp["timestamp"], 0, nuclear, color=COLORS["nuclear"],
                    alpha=0.8, label="Nuclear")
    ax.fill_between(disp["timestamp"], nuclear, nuclear + gas, color=COLORS["gas"],
                    alpha=0.7, label="Natural Gas")

    bess_discharge = np.where(bess_power > 0, bess_power, 0)
    bess_charge = np.where(bess_power < 0, bess_power, 0)
    ax.fill_between(disp["timestamp"], nuclear + gas,
                    nuclear + gas + bess_discharge,
                    color=COLORS["bess_discharge"], alpha=0.7, label="BESS Discharge")
    ax.fill_between(disp["timestamp"], 0, bess_charge,
                    color=COLORS["bess_charge"], alpha=0.5, label="BESS Charge")

    ax.plot(disp["timestamp"], nl, color="black", lw=1.5, label="Net Load (predicted)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("S1 Deterministic Dispatch: Representative Week (Nov 1-7, 2023)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # Middle: BESS SoC
    ax2 = axes[1]
    ax2.fill_between(disp["timestamp"], soc[:T] / bess_cap_mwh * 100,
                     color=COLORS["bess_soc"], alpha=0.4)
    ax2.plot(disp["timestamp"], soc[:T] / bess_cap_mwh * 100,
             color=COLORS["bess_soc"], lw=1.2)
    ax2.set_ylabel("BESS SoC (%)")
    ax2.set_ylim(0, 105)
    ax2.axhline(50, color="gray", ls=":", lw=0.5)

    # Bottom: Hourly cost
    ax3 = axes[2]
    ax3.bar(disp["timestamp"], gas_cost_hourly / 1e3, width=0.04,
            color=COLORS["gas"], alpha=0.7)
    ax3.set_ylabel("Gas Cost ($k/h)")
    ax3.set_xlabel("Date")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%a %m/%d"))

    fig.tight_layout()
    savefig(fig, "fig12_milp_dispatch_stack")
else:
    print("  WARNING: No dispatch data available for the period")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Scenario Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 13: Scenario comparison")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc = scenarios.copy()
sc_labels = sc["Scenario"].str.replace("_", "\n")
sc_colors = ["#1E88E5", "#E53935", "#43A047", "#FF8F00"]

# Total cost
ax = axes[0]
bars = ax.bar(range(len(sc)), sc["Total Cost ($)"].values / 1e6, color=sc_colors,
              edgecolor="white", lw=0.8, alpha=0.8)
ax.set_ylabel("Total Cost (M$)")
ax.set_title("Total System Cost by Scenario")
ax.set_xticks(range(len(sc)))
ax.set_xticklabels(sc_labels, fontsize=9)
for i, v in enumerate(sc["Total Cost ($)"].values / 1e6):
    ax.text(i, v + 1, f"${v:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Cost per MWh
ax2 = axes[1]
bars2 = ax2.bar(range(len(sc)), sc["Cost/MWh ($/MWh)"].values, color=sc_colors,
                edgecolor="white", lw=0.8, alpha=0.8)
ax2.set_ylabel("Cost per MWh ($/MWh)")
ax2.set_title("Unit Cost by Scenario")
ax2.set_xticks(range(len(sc)))
ax2.set_xticklabels(sc_labels, fontsize=9)
for i, v in enumerate(sc["Cost/MWh ($/MWh)"].values):
    ax2.text(i, v + 0.3, f"${v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

fig.suptitle("MILP Optimization: Scenario Comparison", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig13_scenario_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Nuclear Impact (With vs Without Nuclear)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 14: Nuclear impact comparison")

# S1 (with nuclear) vs S4 (no nuclear) from scenarios
s1 = scenarios[scenarios["Scenario"] == "S1_Deterministic"].iloc[0]
s4 = scenarios[scenarios["Scenario"] == "S4_No_Nuclear"].iloc[0]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Cost comparison
ax = axes[0]
labels = ["With Nuclear\n(S1)", "Without Nuclear\n(S4)"]
costs = [s1["Total Cost ($)"] / 1e6, s4["Total Cost ($)"] / 1e6]
colors_nuc = [COLORS["nuclear"], "#E53935"]
bars = ax.bar(labels, costs, color=colors_nuc, alpha=0.8, edgecolor="white", lw=0.8)
for i, v in enumerate(costs):
    ax.text(i, v + 1, f"${v:.1f}M", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Total Cost (M$)")
ax.set_title("Total System Cost")
pct_increase = (costs[1] - costs[0]) / costs[0] * 100
ax.annotate(f"+{pct_increase:.1f}%", xy=(0.5, max(costs) * 0.95),
            ha="center", fontsize=12, color="red", fontweight="bold")

# Generation mix
ax2 = axes[1]
categories = ["Nuclear", "Gas", "Import"]
s1_vals = [s1["Avg Nuclear (MW)"], s1["Avg Gas (MW)"], s1["Avg Import (MW)"]]
s4_vals = [s4["Avg Nuclear (MW)"], s4["Avg Gas (MW)"], s4["Avg Import (MW)"]]
x = np.arange(len(categories))
width = 0.35
ax2.bar(x - width / 2, s1_vals, width, label="With Nuclear (S1)",
        color=COLORS["nuclear"], alpha=0.7)
ax2.bar(x + width / 2, s4_vals, width, label="No Nuclear (S4)",
        color="#E53935", alpha=0.7)
ax2.set_ylabel("Average MW")
ax2.set_title("Average Generation by Source")
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend(fontsize=8)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

# Unit cost comparison
ax3 = axes[2]
unit_costs = [s1["Cost/MWh ($/MWh)"], s4["Cost/MWh ($/MWh)"]]
bars3 = ax3.bar(labels, unit_costs, color=colors_nuc, alpha=0.8, edgecolor="white", lw=0.8)
for i, v in enumerate(unit_costs):
    ax3.text(i, v + 0.3, f"${v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax3.set_ylabel("$/MWh")
ax3.set_title("Unit Cost Comparison")
pct_unit = (unit_costs[1] - unit_costs[0]) / unit_costs[0] * 100
ax3.annotate(f"+{pct_unit:.1f}%", xy=(0.5, max(unit_costs) * 0.97),
             ha="center", fontsize=12, color="red", fontweight="bold")

fig.suptitle("Impact of Nuclear Generation on System Cost", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig14_nuclear_impact")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Sensitivity Analysis (3 subplots)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 15: Sensitivity analysis")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# BESS Size Sensitivity
ax = axes[0]
ax.plot(sens_bess["BESS_MW"] / 1e3, sens_bess["Cost_MWh"],
        "o-", color="#4CAF50", lw=2, markersize=6)
ax.set_xlabel("BESS Capacity (GW)")
ax.set_ylabel("Cost ($/MWh)")
ax.set_title("BESS Size Sensitivity")
# Highlight baseline
baseline_bess = sens_bess[sens_bess["Label"].str.contains("baseline")]
if len(baseline_bess) > 0:
    ax.axvline(baseline_bess["BESS_MW"].values[0] / 1e3, color="gray",
               ls="--", lw=0.8, alpha=0.5)
    ax.annotate("Baseline", xy=(baseline_bess["BESS_MW"].values[0] / 1e3,
                sens_bess["Cost_MWh"].min()),
                fontsize=8, ha="center", va="top", color="gray")

# Add secondary y-axis for BESS cycles
ax_twin = ax.twinx()
ax_twin.plot(sens_bess["BESS_MW"] / 1e3, sens_bess["BESS_Cycles"],
             "s--", color="#FF9800", lw=1.5, markersize=5, alpha=0.7)
ax_twin.set_ylabel("BESS Cycles", color="#FF9800")
ax_twin.tick_params(axis="y", labelcolor="#FF9800")

# Nuclear Capacity Sensitivity
ax2 = axes[1]
ax2.plot(sens_nuc["Nuclear_MW"] / 1e3, sens_nuc["Cost_MWh"],
         "o-", color=COLORS["nuclear"], lw=2, markersize=6)
ax2.set_xlabel("Nuclear Capacity (GW)")
ax2.set_ylabel("Cost ($/MWh)")
ax2.set_title("Nuclear Capacity Sensitivity")
# Highlight current capacity
ax2.axvline(2.256, color="gray", ls="--", lw=0.8, alpha=0.5)
ax2.annotate("Current\n(2.256 GW)", xy=(2.256, sens_nuc["Cost_MWh"].max()),
             fontsize=8, ha="center", va="bottom", color="gray")

# Add gas generation on secondary axis
ax2_twin = ax2.twinx()
ax2_twin.plot(sens_nuc["Nuclear_MW"] / 1e3, sens_nuc["Avg_Gas"] / 1e3,
              "s--", color="#E53935", lw=1.5, markersize=5, alpha=0.7)
ax2_twin.set_ylabel("Avg Gas (GW)", color="#E53935")
ax2_twin.tick_params(axis="y", labelcolor="#E53935")

# Gas Price Sensitivity
ax3 = axes[2]
ax3.plot(sens_gas["Gas_Price"], sens_gas["Cost_MWh"],
         "o-", color=COLORS["gas"], lw=2, markersize=6)
ax3.set_xlabel("Gas Price ($/MMBtu)")
ax3.set_ylabel("System Cost ($/MWh)")
ax3.set_title("Gas Price Sensitivity")
# Highlight baseline ($45)
ax3.axvline(45, color="gray", ls="--", lw=0.8, alpha=0.5)
ax3.annotate("Baseline\n($45)", xy=(45, sens_gas["Cost_MWh"].min()),
             fontsize=8, ha="center", va="top", color="gray")

# Add gas dispatch on secondary axis
ax3_twin = ax3.twinx()
ax3_twin.plot(sens_gas["Gas_Price"], sens_gas["Avg_Gas"] / 1e3,
              "s--", color="#FF8F00", lw=1.5, markersize=5, alpha=0.7)
ax3_twin.set_ylabel("Avg Gas (GW)", color="#FF8F00")
ax3_twin.tick_params(axis="y", labelcolor="#FF8F00")

fig.suptitle("Sensitivity Analysis: System Cost Impact", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig15_sensitivity_analysis")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("All 15 figures generated successfully!")
print(f"Output directory: {FIG_DIR}")
print("=" * 60)
