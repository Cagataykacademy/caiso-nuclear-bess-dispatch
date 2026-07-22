"""
Fetch CAISO wholesale prices + generation by fuel from EIA API.
Handles rate limiting with built-in delays.
"""
import os, sys, io, time, json
import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

API_KEY = "DEMO_KEY"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def fetch_eia(endpoint, facets, out_file, description, chunk_days=7):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")

    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        if len(df) > 5000:
            print(f"  Already have {len(df)} records. Skipping.")
            return df

    url = f'https://api.eia.gov/v2/{endpoint}'
    all_recs = []
    current = pd.Timestamp('2023-01-01')
    end = pd.Timestamp('2023-12-31')
    chunk_num = 0

    while current < end:
        chunk_end = min(current + pd.DateOffset(days=chunk_days), end)
        chunk_num += 1

        params = {
            'api_key': API_KEY,
            'frequency': 'hourly',
            'data[0]': 'value',
            'start': current.strftime('%Y-%m-%dT00'),
            'end': chunk_end.strftime('%Y-%m-%dT00'),
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'length': 5000,
        }
        params.update(facets)

        success = False
        for attempt in range(6):
            try:
                r = requests.get(url, params=params, timeout=60)
                if r.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"  [429] Rate limited. Waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                if 'response' in data and 'data' in data['response']:
                    recs = data['response']['data']
                    all_recs.extend(recs)
                    print(f"  Chunk {chunk_num:>3}: {current.date()}->{chunk_end.date()} "
                          f"| +{len(recs):>4} | Total: {len(all_recs):>6}", flush=True)
                    success = True
                    break
                else:
                    print(f"  Chunk {chunk_num}: empty response", flush=True)
                    success = True
                    break
            except requests.exceptions.ConnectionError as e:
                print(f"  Chunk {chunk_num}: connection error, retry in 30s", flush=True)
                time.sleep(30)
            except Exception as e:
                print(f"  Chunk {chunk_num}: {e}", flush=True)
                time.sleep(15)

        if not success:
            print(f"  Chunk {chunk_num}: FAILED after 6 attempts, skipping", flush=True)

        current = chunk_end
        time.sleep(12)

        if chunk_num % 10 == 0 and all_recs:
            pd.DataFrame(all_recs).to_csv(out_file, index=False)
            print(f"  [Saved progress: {len(all_recs)} records]", flush=True)

    if all_recs:
        df = pd.DataFrame(all_recs)
        df.to_csv(out_file, index=False)
        print(f"\n  DONE: {len(df)} records -> {os.path.basename(out_file)}", flush=True)
        return df
    else:
        print(f"\n  FAILED: no records", flush=True)
        return None

# Wait for rate limit to clear
print("Waiting 65s for rate limit to clear...", flush=True)
time.sleep(65)

# 1. Wholesale prices
prices_df = fetch_eia(
    endpoint='electricity/rto/wholesale-prices/data/',
    facets={'facets[respondent][]': 'CISO'},
    out_file=os.path.join(DATA_DIR, 'caiso_wholesale_prices_2023.csv'),
    description='CAISO Wholesale Prices (hourly LMP)',
    chunk_days=14,
)

if prices_df is not None:
    print(f"\n  Price columns: {list(prices_df.columns)}")
    if 'value' in prices_df.columns:
        v = pd.to_numeric(prices_df['value'], errors='coerce')
        print(f"  Stats: mean=${v.mean():.2f}, min=${v.min():.2f}, max=${v.max():.2f}")

# 2. Generation by fuel type
gen_df = fetch_eia(
    endpoint='electricity/rto/fuel-type-data/data/',
    facets={'facets[respondent][]': 'CISO'},
    out_file=os.path.join(DATA_DIR, 'caiso_generation_by_fuel_2023.csv'),
    description='CAISO Generation by Fuel Type',
    chunk_days=5,
)

if gen_df is not None:
    print(f"\n  Generation columns: {list(gen_df.columns)}")
    for col in ['fueltype', 'type-name', 'fueltypeid']:
        if col in gen_df.columns:
            print(f"\n  Fuel types ({col}):")
            for ft, cnt in gen_df[col].value_counts().items():
                print(f"    {ft}: {cnt}")
            break

print("\n" + "="*60)
print("  ALL DOWNLOADS COMPLETE")
print("="*60)
