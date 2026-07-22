"""
Retry fetch of CAISO generation-by-fuel data with extended rate limit wait.
The DEMO_KEY allows ~100 requests/hour. We need ~14 pages of 5000 records.
"""
import os, sys, io, json, time, warnings
import pandas as pd, numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EIA_API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")
BASE_URL = "https://api.eia.gov/v2"

import urllib.request, urllib.parse, urllib.error, ssl

def fetch_with_retry(url, max_retries=5, base_wait=30):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Academic Research)')
            response = urllib.request.urlopen(req, timeout=120, context=ctx)
            return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_wait * (2 ** attempt)  # exponential backoff
                print(f"    Rate limited (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP {e.code} error")
                return None
        except Exception as e:
            print(f"    Error: {e}")
            return None
    return None

print("=" * 78)
print("  RETRY: Fetching CAISO Generation by Fuel Type")
print("=" * 78)
print(f"  Initial rate limit cooldown: 120 seconds...")
time.sleep(120)  # Wait 2 minutes for rate limit to fully reset

all_records = []
offset = 0
page_size = 5000
total_expected = 70034

while True:
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "start": "2023-01-01",
        "end": "2023-12-31",
        "offset": offset,
        "length": page_size,
        "facets[respondent][]": "CISO",
        "data[]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }
    
    query_string = urllib.parse.urlencode(params, doseq=True)
    url = f"{BASE_URL}/electricity/rto/fuel-type-data/data?{query_string}"
    
    print(f"  Fetching offset={offset}...")
    data = fetch_with_retry(url, max_retries=5, base_wait=30)
    
    if data is None:
        print("  FAILED after all retries")
        break
    
    if "response" not in data or "data" not in data["response"]:
        if "error" in data:
            print(f"  API error: {data['error']}")
            print("  Waiting 120 seconds...")
            time.sleep(120)
            continue
        break
    
    records = data["response"]["data"]
    if not records:
        break
    
    all_records.extend(records)
    total = int(data["response"].get("total", total_expected))
    print(f"    -> {len(all_records)}/{total} records")
    
    if len(all_records) >= total:
        break
    
    offset += page_size
    time.sleep(5)  # 5s between pages to respect rate limits

if all_records:
    gen_raw = pd.DataFrame(all_records)
    raw_path = os.path.join(DATA_DIR, "eia_caiso_generation_raw_2023.csv")
    gen_raw.to_csv(raw_path, index=False)
    print(f"\n  Raw data saved: {raw_path} ({len(gen_raw)} records)")
    
    # Detect fuel type column
    fuel_col = None
    for c in ['fueltype', 'type-name', 'fuelTypeDescription']:
        if c in gen_raw.columns:
            fuel_col = c
            break
    
    if fuel_col:
        print(f"  Fuel types: {gen_raw[fuel_col].unique()}")
        gen_raw['value'] = pd.to_numeric(gen_raw['value'], errors='coerce')
        
        pivoted = gen_raw.pivot_table(index='period', columns=fuel_col, values='value', aggfunc='mean')
        pivoted.index = pd.to_datetime(pivoted.index, utc=True)
        pivoted = pivoted.sort_index()
        
        # Rename
        col_rename = {}
        for col in pivoted.columns:
            cl = str(col).lower()
            if 'solar' in cl: col_rename[col] = 'solar_gen_MW'
            elif 'wind' in cl: col_rename[col] = 'wind_gen_MW'
            elif 'nuclear' in cl: col_rename[col] = 'nuclear_gen_MW'
            elif 'natural' in cl or 'gas' in cl: col_rename[col] = 'nat_gas_gen_MW'
            elif 'coal' in cl: col_rename[col] = 'coal_gen_MW'
            elif 'hydro' in cl or 'water' in cl: col_rename[col] = 'hydro_gen_MW'
            elif 'other' in cl: col_rename[col] = 'other_gen_MW'
            elif 'petroleum' in cl or 'oil' in cl: col_rename[col] = 'petroleum_gen_MW'
            else: col_rename[col] = f'{col}_gen_MW'
        
        pivoted = pivoted.rename(columns=col_rename)
        gen_path = os.path.join(DATA_DIR, "caiso_supply_2023.csv")
        pivoted.to_csv(gen_path)
        print(f"  Processed: {gen_path} (shape: {pivoted.shape})")
        print(f"  Columns: {list(pivoted.columns)}")
        
        # Merge with demand
        demand_path = os.path.join(DATA_DIR, "caiso_demand_2023.csv")
        if os.path.exists(demand_path):
            demand = pd.read_csv(demand_path, index_col=0, parse_dates=True)
            merged = pd.concat([demand, pivoted], axis=1, join='outer')
            merged = merged.loc[:, ~merged.columns.duplicated()]
            
            if 'total_demand_MW' in merged.columns and 'solar_gen_MW' in merged.columns:
                solar = merged.get('solar_gen_MW', 0).fillna(0)
                wind = merged.get('wind_gen_MW', 0).fillna(0)
                merged['net_load_MW'] = merged['total_demand_MW'] - solar - wind
            
            unified_path = os.path.join(DATA_DIR, "caiso_unified_2023.csv")
            merged.to_csv(unified_path)
            print(f"\n  UNIFIED: {unified_path} (shape: {merged.shape})")
            print(f"  Columns: {list(merged.columns)}")
else:
    print("\n  No records retrieved. The DEMO_KEY rate limit may need more time.")
    print("  You can register for a free API key at: https://www.eia.gov/opendata/register.php")

print("\n  Files:")
for f in sorted(os.listdir(DATA_DIR)):
    fpath = os.path.join(DATA_DIR, f)
    if os.path.isfile(fpath):
        print(f"    * {f} ({os.path.getsize(fpath)/1024:.0f} KB)")

print("\n  DONE")
