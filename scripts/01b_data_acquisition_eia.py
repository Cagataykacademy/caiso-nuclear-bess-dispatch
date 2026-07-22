"""
=============================================================================
 STEP 1: DATA ACQUISITION - CAISO Grid Data via EIA Open Data API
=============================================================================
 Project : Data-Driven Nuclear Baseload & BESS Optimization
           under "Duck Curve" Uncertainty
 
 Fallback Strategy:
   Since CAISO OASIS is not accessible from this network, we use the
   U.S. Energy Information Administration (EIA) Open Data API v2.
   
   EIA provides hourly electricity data for all US ISOs including CAISO:
     - Hourly demand (total and net)
     - Hourly generation by fuel type (solar, wind, nuclear, natural gas, etc.)
     - Hourly interchange data
   
   Reference: https://api.eia.gov/v2/
   Dataset: Electricity -> Electric Power Operations (Hourly)
=============================================================================
"""

import os
import sys
import io
import json
import warnings
import datetime
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# -- Configuration ------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# EIA API v2 - Free registration key
# Register at: https://www.eia.gov/opendata/register.php
# For demonstration, we use the demo key (limited rate)
EIA_API_KEY = "DEMO_KEY"  # Replace with your own key for production

# CAISO region identifier in EIA
RESPONDENT = "CISO"  # California ISO

# Date range
START_DATE = "2023-01-01"
END_DATE   = "2023-12-31"

BASE_URL = "https://api.eia.gov/v2"

print("=" * 78)
print("  EIA OPEN DATA API - CAISO DATA ACQUISITION")
print("=" * 78)
print(f"  Target region : CISO (California ISO)")
print(f"  Target period : {START_DATE} -> {END_DATE}")
print(f"  Output dir    : {DATA_DIR}")
print("=" * 78)


def fetch_eia_data(route, params, description="data"):
    """Fetch data from EIA API v2 with pagination support."""
    import urllib.request
    import urllib.parse
    import urllib.error
    import ssl
    
    # Allow unverified SSL for corporate proxies
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    all_records = []
    offset = 0
    page_size = 5000  # EIA max per request
    
    base_params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "start": START_DATE,
        "end": END_DATE,
        "offset": offset,
        "length": page_size,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }
    base_params.update(params)
    
    while True:
        base_params["offset"] = offset
        query_string = urllib.parse.urlencode(base_params, doseq=True)
        url = f"{BASE_URL}/{route}?{query_string}"
        
        print(f"       Fetching {description} (offset={offset})...")
        
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Research Project)')
            response = urllib.request.urlopen(req, timeout=60, context=ctx)
            data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            print(f"       HTTP Error {e.code}: {error_body[:200]}")
            break
        except Exception as e:
            print(f"       Error: {e}")
            break
        
        if "response" not in data or "data" not in data["response"]:
            print(f"       Unexpected response structure: {list(data.keys())}")
            if "error" in data:
                print(f"       API Error: {data['error']}")
            break
        
        records = data["response"]["data"]
        if not records:
            break
        
        all_records.extend(records)
        total = int(data["response"].get("total", len(all_records)))
        print(f"       -> Got {len(records)} records (total so far: {len(all_records)}/{total})")
        
        if len(all_records) >= total:
            break
        
        offset += page_size
    
    if all_records:
        df = pd.DataFrame(all_records)
        print(f"       [OK] Total {description}: {len(df)} records")
        return df
    else:
        print(f"       [FAIL] No {description} retrieved")
        return pd.DataFrame()


# =============================================================================
#  FETCH 1: Hourly Demand Data (Total & Net Load)
# =============================================================================
def fetch_demand():
    """Fetch CAISO hourly demand data from EIA."""
    print("\n[1/3] Fetching CAISO Hourly Demand...")
    
    params = {
        "facets[respondent][]": RESPONDENT,
        "data[]": "value",
    }
    
    df = fetch_eia_data(
        "electricity/rto/region-data/data",
        params,
        description="CAISO demand"
    )
    return df


# =============================================================================
#  FETCH 2: Hourly Generation by Fuel Type
# =============================================================================
def fetch_generation_by_fuel():
    """Fetch CAISO hourly generation by fuel type."""
    print("\n[2/3] Fetching CAISO Hourly Generation by Fuel Type...")
    
    params = {
        "facets[respondent][]": RESPONDENT,
        "data[]": "value",
    }
    
    df = fetch_eia_data(
        "electricity/rto/fuel-type-data/data",
        params,
        description="CAISO generation by fuel"
    )
    return df


# =============================================================================
#  FETCH 3: Hourly Interchange Data
# =============================================================================
def fetch_interchange():
    """Fetch CAISO hourly interchange data."""
    print("\n[3/3] Fetching CAISO Hourly Interchange...")
    
    params = {
        "facets[fromba][]": RESPONDENT,
        "data[]": "value",
    }
    
    df = fetch_eia_data(
        "electricity/rto/interchange-data/data",
        params,
        description="CAISO interchange"
    )
    return df


# =============================================================================
#  POST-PROCESSING: Transform EIA data into analysis-ready format
# =============================================================================
def process_demand_data(df):
    """Process raw EIA demand data into clean hourly format."""
    if df.empty:
        return pd.DataFrame()
    
    print("\n  Processing demand data...")
    print(f"    Raw columns: {list(df.columns)}")
    print(f"    Unique type-name values: {df.get('type-name', pd.Series()).unique()}")
    
    # Pivot: each demand type becomes a column
    if 'type-name' in df.columns and 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        pivoted = df.pivot_table(
            index='period', columns='type-name', values='value', aggfunc='mean'
        )
    elif 'type' in df.columns and 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        pivoted = df.pivot_table(
            index='period', columns='type', values='value', aggfunc='mean'
        )
    else:
        print("    Unexpected column structure, saving raw")
        return df
    
    pivoted.index = pd.to_datetime(pivoted.index, utc=True)
    pivoted = pivoted.sort_index()
    
    # Rename columns for clarity
    col_rename = {}
    for col in pivoted.columns:
        cl = str(col).lower()
        if 'demand' in cl and 'net' not in cl and 'forecast' not in cl:
            col_rename[col] = 'total_demand_MW'
        elif 'net' in cl and ('demand' in cl or 'gen' in cl or 'load' in cl):
            col_rename[col] = 'net_demand_MW'
        elif 'forecast' in cl and 'demand' in cl and 'day' in cl:
            col_rename[col] = 'day_ahead_demand_forecast_MW'
        elif 'forecast' in cl and 'demand' in cl:
            col_rename[col] = 'demand_forecast_MW'
        elif 'net' in cl and 'forecast' in cl:
            col_rename[col] = 'net_demand_forecast_MW'
    
    if col_rename:
        pivoted = pivoted.rename(columns=col_rename)
    
    print(f"    Processed columns: {list(pivoted.columns)}")
    print(f"    Shape: {pivoted.shape}")
    print(f"    Date range: {pivoted.index.min()} -> {pivoted.index.max()}")
    
    return pivoted


def process_generation_data(df):
    """Process raw EIA generation-by-fuel data into clean hourly format."""
    if df.empty:
        return pd.DataFrame()
    
    print("\n  Processing generation data...")
    print(f"    Raw columns: {list(df.columns)}")
    
    fuel_col = 'fueltype' if 'fueltype' in df.columns else 'type-name'
    if fuel_col not in df.columns:
        for c in df.columns:
            if 'fuel' in c.lower() or 'type' in c.lower():
                fuel_col = c
                break
    
    print(f"    Unique fuel types: {df.get(fuel_col, pd.Series()).unique()}")
    
    if fuel_col in df.columns and 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        pivoted = df.pivot_table(
            index='period', columns=fuel_col, values='value', aggfunc='mean'
        )
    else:
        print("    Unexpected column structure, saving raw")
        return df
    
    pivoted.index = pd.to_datetime(pivoted.index, utc=True)
    pivoted = pivoted.sort_index()
    
    # Standardize column names
    col_rename = {}
    for col in pivoted.columns:
        cl = str(col).lower()
        if 'solar' in cl or cl == 'sun':
            col_rename[col] = 'solar_gen_MW'
        elif 'wind' in cl or cl == 'wnd':
            col_rename[col] = 'wind_gen_MW'
        elif 'nuclear' in cl or cl == 'nuc':
            col_rename[col] = 'nuclear_gen_MW'
        elif 'natural' in cl or 'gas' in cl or cl == 'ng':
            col_rename[col] = 'nat_gas_gen_MW'
        elif 'coal' in cl or cl == 'col':
            col_rename[col] = 'coal_gen_MW'
        elif 'hydro' in cl or cl == 'wat':
            col_rename[col] = 'hydro_gen_MW'
        elif 'other' in cl or cl == 'oth':
            col_rename[col] = 'other_gen_MW'
        elif 'petroleum' in cl or 'oil' in cl or cl == 'oil':
            col_rename[col] = 'petroleum_gen_MW'
        elif 'all' in cl and ('solar' not in cl):
            col_rename[col] = 'total_gen_MW'
        else:
            col_rename[col] = f'{col}_gen_MW'
    
    if col_rename:
        pivoted = pivoted.rename(columns=col_rename)
    
    print(f"    Processed columns: {list(pivoted.columns)}")
    print(f"    Shape: {pivoted.shape}")
    
    return pivoted


# =============================================================================
#  MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    success_count = 0
    
    # --- Fetch Demand ---
    try:
        demand_raw = fetch_demand()
        if not demand_raw.empty:
            raw_path = os.path.join(DATA_DIR, "eia_caiso_demand_raw_2023.csv")
            demand_raw.to_csv(raw_path, index=False)
            
            demand_df = process_demand_data(demand_raw)
            if not demand_df.empty:
                demand_path = os.path.join(DATA_DIR, "caiso_demand_2023.csv")
                demand_df.to_csv(demand_path)
                print(f"    -> Saved processed: {demand_path}")
                success_count += 1
    except Exception as e:
        print(f"  [FAIL] Demand: {e}")
        import traceback
        traceback.print_exc()
        demand_df = pd.DataFrame()

    # --- Fetch Generation ---
    try:
        gen_raw = fetch_generation_by_fuel()
        if not gen_raw.empty:
            raw_path = os.path.join(DATA_DIR, "eia_caiso_generation_raw_2023.csv")
            gen_raw.to_csv(raw_path, index=False)
            
            gen_df = process_generation_data(gen_raw)
            if not gen_df.empty:
                gen_path = os.path.join(DATA_DIR, "caiso_supply_2023.csv")
                gen_df.to_csv(gen_path)
                print(f"    -> Saved processed: {gen_path}")
                success_count += 1
    except Exception as e:
        print(f"  [FAIL] Generation: {e}")
        import traceback
        traceback.print_exc()
        gen_df = pd.DataFrame()

    # --- Fetch Interchange (optional) ---
    try:
        int_raw = fetch_interchange()
        if not int_raw.empty:
            raw_path = os.path.join(DATA_DIR, "eia_caiso_interchange_raw_2023.csv")
            int_raw.to_csv(raw_path, index=False)
            success_count += 1
    except Exception as e:
        print(f"  [FAIL] Interchange: {e}")
        int_raw = pd.DataFrame()

    # --- Merge into unified dataset ---
    print("\n" + "=" * 78)
    print(f"  DATA ACQUISITION SUMMARY: {success_count}/3 datasets retrieved")
    print("=" * 78)
    
    # Try to merge demand + generation
    try:
        dfs_to_merge = []
        if 'demand_df' in dir() and not demand_df.empty:
            dfs_to_merge.append(demand_df)
        if 'gen_df' in dir() and not gen_df.empty:
            dfs_to_merge.append(gen_df)
        
        if dfs_to_merge:
            merged = pd.concat(dfs_to_merge, axis=1, join='outer')
            merged = merged.loc[:, ~merged.columns.duplicated()]
            
            # Compute net load if not present
            if 'total_demand_MW' in merged.columns and 'solar_gen_MW' in merged.columns:
                solar = merged.get('solar_gen_MW', 0)
                wind = merged.get('wind_gen_MW', 0)
                if isinstance(solar, (int, float)):
                    solar = pd.Series(0, index=merged.index)
                if isinstance(wind, (int, float)):
                    wind = pd.Series(0, index=merged.index)
                merged['net_load_MW'] = merged['total_demand_MW'] - solar - wind
                print("  Computed: net_load_MW = total_demand - solar - wind")
            
            merged_path = os.path.join(DATA_DIR, "caiso_unified_2023.csv")
            merged.to_csv(merged_path)
            print(f"\n  [OK] Unified dataset saved: {merged_path}")
            print(f"       Shape: {merged.shape}")
            print(f"       Columns: {list(merged.columns)}")
        else:
            print("\n  [WARN] No datasets to merge")
            
    except Exception as e:
        print(f"  [FAIL] Merge: {e}")
        import traceback
        traceback.print_exc()
    
    # List output files
    print("\n  Output files:")
    if os.path.exists(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            fpath = os.path.join(DATA_DIR, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"    * {f} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 78)
    print("  DATA ACQUISITION COMPLETE")
    print("=" * 78)
