"""
Fast EIA data fetcher with real API key (1000 req/hr).
Downloads CAISO wholesale prices + generation by fuel type.
"""
import os, sys, io, time, json, requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")  # set your own key: https://www.eia.gov/opendata/register.php
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DELAY = 4  # 4s between requests (safe for 1000/hr limit)

def eia_get(endpoint, facets, start_str, end_str):
    url = f'https://api.eia.gov/v2/{endpoint}'
    params = {
        'api_key': API_KEY, 'frequency': 'hourly', 'data[0]': 'value',
        'start': start_str, 'end': end_str,
        'sort[0][column]': 'period', 'sort[0][direction]': 'asc',
        'length': 5000,
    }
    params.update(facets)
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    [429] Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if 'response' in data and 'data' in data['response']:
                return data['response']['data']
            return []
        except Exception as e:
            print(f"    Error: {e}", flush=True)
            time.sleep(10)
    return []

def fetch_dataset(endpoint, facets, out_file, desc, chunk_days):
    print(f"\n{'='*60}", flush=True)
    print(f"  {desc}", flush=True)

    if os.path.exists(out_file):
        existing = pd.read_csv(out_file)
        if len(existing) > 5000:
            print(f"  Already have {len(existing)} records, skipping.", flush=True)
            return existing

    all_recs = []
    current = pd.Timestamp('2023-01-01')
    end = pd.Timestamp('2023-12-31')
    chunk = 0

    while current < end:
        chunk_end = min(current + pd.DateOffset(days=chunk_days), end)
        chunk += 1
        recs = eia_get(endpoint, facets, current.strftime('%Y-%m-%dT00'), chunk_end.strftime('%Y-%m-%dT00'))
        all_recs.extend(recs)
        print(f"  #{chunk:>3}: {current.date()}->{chunk_end.date()} +{len(recs):>4} = {len(all_recs):>6}", flush=True)
        current = chunk_end
        time.sleep(DELAY)

    if all_recs:
        df = pd.DataFrame(all_recs)
        df.to_csv(out_file, index=False)
        print(f"  DONE: {len(df)} records -> {os.path.basename(out_file)}", flush=True)
        return df
    print("  FAILED: no records", flush=True)
    return None

print("="*60, flush=True)
print("  FAST EIA FETCHER (real API key)", flush=True)
print("="*60, flush=True)

# 1. Wholesale prices (~26 chunks x 4s = ~2 min)
prices = fetch_dataset(
    'electricity/rto/wholesale-prices/data/',
    {'facets[respondent][]': 'CISO'},
    os.path.join(DATA_DIR, 'caiso_wholesale_prices_2023.csv'),
    'CAISO Wholesale Prices (hourly)', chunk_days=14
)
if prices is not None and len(prices) > 0:
    print(f"  Columns: {list(prices.columns)}", flush=True)
    if 'value' in prices.columns:
        v = pd.to_numeric(prices['value'], errors='coerce')
        print(f"  Price stats: mean=${v.mean():.2f}, min=${v.min():.2f}, max=${v.max():.2f}", flush=True)

# 2. Generation by fuel type (~73 chunks x 4s = ~5 min)
gen = fetch_dataset(
    'electricity/rto/fuel-type-data/data/',
    {'facets[respondent][]': 'CISO'},
    os.path.join(DATA_DIR, 'caiso_generation_by_fuel_2023.csv'),
    'CAISO Generation by Fuel Type (hourly)', chunk_days=5
)
if gen is not None and len(gen) > 0:
    print(f"  Columns: {list(gen.columns)}", flush=True)
    for col in ['fueltype', 'type-name', 'fueltypeid']:
        if col in gen.columns:
            print(f"\n  Fuel types:", flush=True)
            for ft, cnt in gen[col].value_counts().items():
                print(f"    {ft}: {cnt}", flush=True)
            break

print("\n" + "="*60, flush=True)
print("  ALL DOWNLOADS COMPLETE", flush=True)
print("="*60, flush=True)
