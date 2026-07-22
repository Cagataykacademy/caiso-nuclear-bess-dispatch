"""
=============================================================================
 STEP 1: DATA ACQUISITION - CAISO Grid Data via gridstatus
=============================================================================
 Project : Data-Driven Nuclear Baseload & BESS Optimization 
           under "Duck Curve" Uncertainty
 Author  : Research Team
 Target  : Q1 Journal (Applied Energy / EJOR)
 
 Description:
   Downloads real-world CAISO (California ISO) hourly grid data using the 
   `gridstatus` Python library. We fetch:
     - Demand (Total Load & Net Load)
     - Supply (Solar, Wind, and other generation by fuel type)
     - Day-Ahead & Real-Time Locational Marginal Prices (LMP)
   
   The data is saved to CSV for reproducibility.
=============================================================================
"""

import os
import sys
import io
import warnings
import datetime
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# -- Configuration ------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Date range: We target a full recent year for robust modeling
# Using 2023 as the primary year (most complete recent data available)
START_DATE = "2023-01-01"
END_DATE   = "2023-12-31"

print("=" * 78)
print("  CAISO DATA ACQUISITION PIPELINE")
print("=" * 78)
print(f"  Target period : {START_DATE} -> {END_DATE}")
print(f"  Output dir    : {DATA_DIR}")
print("=" * 78)

# ── Step 1A: Fetch CAISO Demand Data (Load & Net Load) ───────────────────────
def fetch_caiso_demand():
    """Fetch hourly demand data including total load and net load."""
    import gridstatus
    
    print("\n[1/3] Fetching CAISO Demand Data (Load & Net Load)...")
    caiso = gridstatus.CAISO()
    
    # Fetch demand data in monthly chunks to avoid timeout issues
    all_demand = []
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    
    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(months=1), end)
        print(f"       Downloading demand: {current.date()} → {chunk_end.date()}...")
        try:
            df = caiso.get_demand(
                start=current.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
            )
            all_demand.append(df)
        except Exception as e:
            print(f"       ⚠ Warning: Failed chunk {current.date()}: {e}")
        current = chunk_end
    
    if all_demand:
        demand_df = pd.concat(all_demand, ignore_index=True)
        print(f"       ✓ Demand data: {len(demand_df)} records fetched")
        return demand_df
    else:
        print("       ✗ No demand data fetched")
        return pd.DataFrame()

# ── Step 1B: Fetch CAISO Supply Data (Generation by Fuel Type) ───────────────
def fetch_caiso_supply():
    """Fetch hourly supply/generation breakdown by fuel type (solar, wind, etc.)."""
    import gridstatus
    
    print("\n[2/3] Fetching CAISO Supply Data (Generation by Fuel Type)...")
    caiso = gridstatus.CAISO()
    
    all_supply = []
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    
    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(months=1), end)
        print(f"       Downloading supply: {current.date()} → {chunk_end.date()}...")
        try:
            df = caiso.get_supply(
                start=current.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
            )
            all_supply.append(df)
        except Exception as e:
            print(f"       ⚠ Warning: Failed chunk {current.date()}: {e}")
        current = chunk_end
    
    if all_supply:
        supply_df = pd.concat(all_supply, ignore_index=True)
        print(f"       ✓ Supply data: {len(supply_df)} records fetched")
        return supply_df
    else:
        print("       ✗ No supply data fetched")
        return pd.DataFrame()

# ── Step 1C: Fetch CAISO Day-Ahead LMP Prices ───────────────────────────────
def fetch_caiso_prices():
    """Fetch Day-Ahead Locational Marginal Prices (LMP) for the system-wide hub."""
    import gridstatus
    
    print("\n[3/3] Fetching CAISO Day-Ahead LMP Prices...")
    caiso = gridstatus.CAISO()
    
    all_prices = []
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    
    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(days=7), end)
        print(f"       Downloading prices: {current.date()} → {chunk_end.date()}...")
        try:
            df = caiso.get_lmp(
                start=current.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                market="DAY_AHEAD_HOURLY",
                locations=["TH_SP15_GEN-APND"],  # SP15 hub (Southern California)
            )
            all_prices.append(df)
        except Exception as e:
            print(f"       ⚠ Warning: Failed chunk {current.date()}: {e}")
        current = chunk_end
    
    if all_prices:
        prices_df = pd.concat(all_prices, ignore_index=True)
        print(f"       ✓ Price data: {len(prices_df)} records fetched")
        return prices_df
    else:
        print("       ✗ No price data fetched")
        return pd.DataFrame()


# ── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    success_count = 0
    
    # --- Demand ---
    try:
        demand_df = fetch_caiso_demand()
        if not demand_df.empty:
            demand_path = os.path.join(DATA_DIR, "caiso_demand_2023.csv")
            demand_df.to_csv(demand_path, index=False)
            print(f"       → Saved to: {demand_path}")
            success_count += 1
        else:
            demand_df = None
    except Exception as e:
        print(f"  ✗ Demand fetch failed: {e}")
        demand_df = None

    # --- Supply ---
    try:
        supply_df = fetch_caiso_supply()
        if not supply_df.empty:
            supply_path = os.path.join(DATA_DIR, "caiso_supply_2023.csv")
            supply_df.to_csv(supply_path, index=False)
            print(f"       → Saved to: {supply_path}")
            success_count += 1
        else:
            supply_df = None
    except Exception as e:
        print(f"  ✗ Supply fetch failed: {e}")
        supply_df = None

    # --- Prices ---
    try:
        prices_df = fetch_caiso_prices()
        if not prices_df.empty:
            prices_path = os.path.join(DATA_DIR, "caiso_lmp_2023.csv")
            prices_df.to_csv(prices_path, index=False)
            print(f"       → Saved to: {prices_path}")
            success_count += 1
        else:
            prices_df = None
    except Exception as e:
        print(f"  ✗ Price fetch failed: {e}")
        prices_df = None

    # --- Summary ---
    print("\n" + "=" * 78)
    print(f"  DATA ACQUISITION COMPLETE: {success_count}/3 datasets saved")
    print("=" * 78)
    
    if success_count > 0:
        print("\n  Files in data directory:")
        for f in os.listdir(DATA_DIR):
            fpath = os.path.join(DATA_DIR, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"    • {f} ({size_mb:.2f} MB)")
    
    if success_count == 0:
        print("\n  ⚠ All fetches failed. Please check network/API connectivity.")
        print("    Falling back to alternative data source...")
        sys.exit(1)
