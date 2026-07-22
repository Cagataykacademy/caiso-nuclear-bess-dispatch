"""
=============================================================================
 Fetch CAISO Generation by Fuel Type from EIA API
 Endpoint: electricity/rto/fuel-type-data/data
 Fuels: solar, wind, nuclear, natural gas, hydro, etc.
=============================================================================
"""

import os
import sys
import io
import time
import json
import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

EIA_API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_FILE = os.path.join(DATA_DIR, "caiso_generation_by_fuel_2023.csv")

print("=" * 78)
print("  CAISO GENERATION BY FUEL TYPE (EIA API)")
print("=" * 78)

if os.path.exists(OUT_FILE):
    existing = pd.read_csv(OUT_FILE)
    print(f"  Existing file: {len(existing)} records")
    if len(existing) >= 50000:
        print("  Already have enough data. Skipping.")
        sys.exit(0)

url = 'https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/'

all_records = []
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')

current = start_date
chunk_num = 0

print(f"\n  Target: Hourly generation by fuel type for CAISO 2023")
print(f"  Expected fuels: SUN (solar), WND (wind), NUC (nuclear), NG (gas), WAT (hydro)")
print(f"  Using 12s delay between requests\n")

print("  Waiting 20s for rate limit to clear...")
time.sleep(20)

while current < end_date:
    chunk_end = min(current + pd.DateOffset(days=5), end_date)
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
                print(f"  Chunk {chunk_num}: No data")
                break
        except Exception as e:
            print(f"  Chunk {chunk_num} attempt {attempt+1}: {e}")
            time.sleep(15)

    current = chunk_end
    time.sleep(12)

    if chunk_num % 10 == 0 and all_records:
        pd.DataFrame(all_records).to_csv(OUT_FILE, index=False)
        print(f"  [Progress: {len(all_records)} records saved]")

if all_records:
    df = pd.DataFrame(all_records)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n  DONE: {len(df)} records -> {OUT_FILE}")
    print(f"  Columns: {list(df.columns)}")

    if 'fueltype' in df.columns:
        print(f"\n  Fuel types found:")
        for ft, count in df['fueltype'].value_counts().items():
            print(f"    {ft}: {count} records")
    elif 'type-name' in df.columns:
        print(f"\n  Types found:")
        for ft, count in df['type-name'].value_counts().items():
            print(f"    {ft}: {count} records")
else:
    print("\n  ERROR: No records downloaded")
    sys.exit(1)
