"""
=============================================================================
 Fetch CAISO Hourly Wholesale Electricity Prices from EIA API
 Endpoint: electricity/rto/wholesale-prices/data
 Rate limit handling: 10s delay between requests (DEMO_KEY = 30 req/hr)
=============================================================================
"""

import os
import sys
import io
import time
import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

EIA_API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_FILE = os.path.join(DATA_DIR, "caiso_wholesale_prices_2023.csv")

print("=" * 78)
print("  CAISO WHOLESALE PRICE DATA ACQUISITION (EIA API)")
print("=" * 78)

# Check if already downloaded
if os.path.exists(OUT_FILE):
    existing = pd.read_csv(OUT_FILE)
    print(f"  Existing file found: {len(existing)} records")
    if len(existing) >= 8000:
        print("  Already have enough data. Skipping download.")
        sys.exit(0)

url = 'https://api.eia.gov/v2/electricity/rto/wholesale-prices/data/'

all_records = []
# Fetch in weekly chunks to stay within API limits
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')

current = start_date
chunk_num = 0

print(f"\n  Downloading hourly wholesale prices for CAISO (2023)...")
print(f"  Using 12s delay between requests to respect rate limits\n")

# Initial wait to clear rate limit
print("  Waiting 15s for rate limit to reset...")
time.sleep(15)

while current < end_date:
    chunk_end = min(current + pd.DateOffset(days=7), end_date)
    chunk_num += 1

    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': 'CISO',
        'start': current.strftime('%Y-%m-%dT00'),
        'end': chunk_end.strftime('%Y-%m-%dT00'),
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'length': 5000,
    }

    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()

            if 'response' in data and 'data' in data['response']:
                records = data['response']['data']
                all_records.extend(records)
                print(f"  Chunk {chunk_num:>3}: {current.date()} -> {chunk_end.date()} | "
                      f"{len(records):>4} records | Total: {len(all_records):>6}")
                break
            else:
                print(f"  Chunk {chunk_num}: No data in response")
                break
        except Exception as e:
            print(f"  Chunk {chunk_num} attempt {attempt+1} failed: {e}")
            time.sleep(15)

    current = chunk_end
    time.sleep(12)

    # Save progress every 10 chunks
    if chunk_num % 10 == 0 and all_records:
        temp_df = pd.DataFrame(all_records)
        temp_df.to_csv(OUT_FILE, index=False)
        print(f"  [Progress saved: {len(all_records)} records]")

# Final save
if all_records:
    df = pd.DataFrame(all_records)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n  DONE: {len(df)} records saved to {OUT_FILE}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['period'].min()} -> {df['period'].max()}")
    if 'value' in df.columns:
        vals = pd.to_numeric(df['value'], errors='coerce')
        print(f"  Price stats: mean=${vals.mean():.2f}, min=${vals.min():.2f}, max=${vals.max():.2f}")
else:
    print("\n  ERROR: No records downloaded")
    sys.exit(1)
