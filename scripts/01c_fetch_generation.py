"""
Fetch CAISO generation-by-fuel and interchange data from EIA API.
This is a follow-up script to get the remaining data after rate limiting.
"""

import os
import sys
import io
import json
import time
import warnings
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EIA_API_KEY = "DEMO_KEY"
RESPONDENT = "CISO"
START_DATE = "2023-01-01"
END_DATE   = "2023-12-31"
BASE_URL = "https://api.eia.gov/v2"


def fetch_eia_data(route, params, description="data"):
    import urllib.request, urllib.parse, urllib.error, ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    all_records = []
    offset = 0
    page_size = 5000
    
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
        
        print(f"  Fetching {description} (offset={offset})...")
        
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Research Project)')
            response = urllib.request.urlopen(req, timeout=120, context=ctx)
            data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"  Rate limited! Waiting 60 seconds...")
                time.sleep(60)
                try:
                    req = urllib.request.Request(url)
                    req.add_header('User-Agent', 'Mozilla/5.0 (Research Project)')
                    response = urllib.request.urlopen(req, timeout=120, context=ctx)
                    data = json.loads(response.read().decode('utf-8'))
                except Exception as e2:
                    print(f"  Retry also failed: {e2}")
                    break
            else:
                print(f"  HTTP Error {e.code}")
                break
        except Exception as e:
            print(f"  Error: {e}")
            break
        
        if "response" not in data or "data" not in data["response"]:
            if "error" in data:
                print(f"  API Error: {data['error']}")
                print("  Waiting 60 seconds before retry...")
                time.sleep(60)
                continue
            break
        
        records = data["response"]["data"]
        if not records:
            break
        
        all_records.extend(records)
        total = int(data["response"].get("total", len(all_records)))
        print(f"    -> {len(all_records)}/{total} records")
        
        if len(all_records) >= total:
            break
        
        offset += page_size
        # Rate limiting: pause between requests
        time.sleep(2)
    
    if all_records:
        return pd.DataFrame(all_records)
    return pd.DataFrame()


print("=" * 78)
print("  EIA API - FETCHING GENERATION BY FUEL TYPE")
print("=" * 78)
print("  Waiting 10 seconds for rate limit reset...")
time.sleep(10)

# --- Generation by fuel type ---
gen_raw = fetch_eia_data(
    "electricity/rto/fuel-type-data/data",
    {"facets[respondent][]": RESPONDENT, "data[]": "value"},
    description="CAISO generation by fuel"
)

if not gen_raw.empty:
    raw_path = os.path.join(DATA_DIR, "eia_caiso_generation_raw_2023.csv")
    gen_raw.to_csv(raw_path, index=False)
    print(f"\n  Raw generation data saved: {raw_path}")
    print(f"  Shape: {gen_raw.shape}")
    print(f"  Columns: {list(gen_raw.columns)}")
    
    # Identify fuel type column
    fuel_col = None
    for c in ['fueltype', 'type-name', 'fuelTypeDescription']:
        if c in gen_raw.columns:
            fuel_col = c
            break
    if fuel_col is None:
        for c in gen_raw.columns:
            if 'fuel' in c.lower() or 'type' in c.lower():
                fuel_col = c
                break
    
    if fuel_col:
        print(f"  Fuel types found: {gen_raw[fuel_col].unique()}")
        gen_raw['value'] = pd.to_numeric(gen_raw['value'], errors='coerce')
        
        pivoted = gen_raw.pivot_table(
            index='period', columns=fuel_col, values='value', aggfunc='mean'
        )
        pivoted.index = pd.to_datetime(pivoted.index, utc=True)
        pivoted = pivoted.sort_index()
        
        # Rename columns
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
            elif 'hydro' in cl or 'water' in cl or cl == 'wat':
                col_rename[col] = 'hydro_gen_MW'
            elif 'other' in cl or cl == 'oth':
                col_rename[col] = 'other_gen_MW'
            elif 'petroleum' in cl or 'oil' in cl:
                col_rename[col] = 'petroleum_gen_MW'
            else:
                col_rename[col] = f'{col}_gen_MW'
        
        pivoted = pivoted.rename(columns=col_rename)
        gen_path = os.path.join(DATA_DIR, "caiso_supply_2023.csv")
        pivoted.to_csv(gen_path)
        print(f"  Processed generation saved: {gen_path}")
        print(f"  Shape: {pivoted.shape}")
        print(f"  Columns: {list(pivoted.columns)}")
        
        # Now merge with existing demand data
        demand_path = os.path.join(DATA_DIR, "caiso_demand_2023.csv")
        if os.path.exists(demand_path):
            demand = pd.read_csv(demand_path, index_col=0, parse_dates=True)
            merged = pd.concat([demand, pivoted], axis=1, join='outer')
            merged = merged.loc[:, ~merged.columns.duplicated()]
            
            # Compute net load
            if 'total_demand_MW' in merged.columns and 'solar_gen_MW' in merged.columns:
                solar = merged.get('solar_gen_MW', 0).fillna(0)
                wind = merged.get('wind_gen_MW', 0).fillna(0)
                merged['net_load_MW'] = merged['total_demand_MW'] - solar - wind
                print("  Computed: net_load_MW = total_demand - solar - wind")
            
            unified_path = os.path.join(DATA_DIR, "caiso_unified_2023.csv")
            merged.to_csv(unified_path)
            print(f"\n  UNIFIED DATASET: {unified_path}")
            print(f"  Shape: {merged.shape}")
            print(f"  Columns: {list(merged.columns)}")
    else:
        print("  Could not identify fuel type column")
else:
    print("  FAILED to fetch generation data")

# List output files
print("\n  Output files:")
for f in sorted(os.listdir(DATA_DIR)):
    fpath = os.path.join(DATA_DIR, f)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    * {f} ({size_mb:.2f} MB)")

print("\n  DONE")
