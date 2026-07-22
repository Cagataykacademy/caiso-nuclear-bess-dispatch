"""
Patient EIA data fetcher — waits 5 min for rate limit to fully reset,
then uses 2-min gaps between requests. Downloads both prices and generation.
"""
import os, sys, io, time, json, requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

API_KEY = "DEMO_KEY"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DELAY = 120  # 2 min between requests (DEMO_KEY: 30/hr = 1 every 2 min)

def eia_get(endpoint, facets, start_str, end_str):
    url = f'https://api.eia.gov/v2/{endpoint}'
    params = {
        'api_key': API_KEY, 'frequency': 'hourly', 'data[0]': 'value',
        'start': start_str, 'end': end_str,
        'sort[0][column]': 'period', 'sort[0][direction]': 'asc',
        'length': 5000,
    }
    params.update(facets)
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                wait = 180 * (attempt + 1)
                print(f"    [429] Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if 'response' in data and 'data' in data['response']:
                return data['response']['data']
            return []
        except Exception as e:
            print(f"    Error: {e}, retry in 60s", flush=True)
            time.sleep(60)
    return []

def fetch_dataset(endpoint, facets, out_file, desc, chunk_days):
    print(f"\n{'='*60}", flush=True)
    print(f"  {desc}", flush=True)

    if os.path.exists(out_file):
        existing = pd.read_csv(out_file)
        if len(existing) > 5000:
            print(f"  Already have {len(existing)} records, skipping.", flush=True)
            return True

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

        if chunk % 5 == 0 and all_recs:
            pd.DataFrame(all_recs).to_csv(out_file, index=False)

        time.sleep(DELAY)

    if all_recs:
        df = pd.DataFrame(all_recs)
        df.to_csv(out_file, index=False)
        print(f"  DONE: {len(df)} records saved", flush=True)
        return True
    return False

# ---- MAIN ----
print("="*60, flush=True)
print("  PATIENT EIA FETCHER (2-min intervals)", flush=True)
print(f"  Waiting 5 minutes for rate limit full reset...", flush=True)
time.sleep(300)
print("  Rate limit should be clear now. Starting downloads.", flush=True)

# 1. Wholesale prices (26 chunks at 14-day intervals, ~52 min)
fetch_dataset(
    'electricity/rto/wholesale-prices/data/',
    {'facets[respondent][]': 'CISO'},
    os.path.join(DATA_DIR, 'caiso_wholesale_prices_2023.csv'),
    'Wholesale Prices', chunk_days=14
)

# 2. Generation by fuel (73 chunks at 5-day intervals, ~146 min)
fetch_dataset(
    'electricity/rto/fuel-type-data/data/',
    {'facets[respondent][]': 'CISO'},
    os.path.join(DATA_DIR, 'caiso_generation_by_fuel_2023.csv'),
    'Generation by Fuel Type', chunk_days=5
)

print("\n" + "="*60, flush=True)
print("  ALL DONE", flush=True)
