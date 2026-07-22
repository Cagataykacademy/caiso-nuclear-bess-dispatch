"""
Fetch 2022 CAISO data for out-of-sample validation.
Same pipeline as 2023: generation by fuel + region data.
"""
import os, sys, io, time, requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")  # set your own key: https://www.eia.gov/opendata/register.php
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DELAY = 3

def fetch(endpoint, facets, out_file, desc, chunk_days, year='2022'):
    print(f"\n  {desc}", flush=True)
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        if len(df) > 5000:
            print(f"  Already have {len(df)} records, skipping.", flush=True)
            return df

    all_recs = []
    current = pd.Timestamp(f'{year}-01-01')
    end = pd.Timestamp(f'{int(year)+1}-01-01')
    chunk = 0
    while current < end:
        chunk_end = min(current + pd.DateOffset(days=chunk_days), end)
        chunk += 1
        try:
            r = requests.get(f'https://api.eia.gov/v2/{endpoint}', params={
                'api_key': API_KEY, 'frequency': 'hourly', 'data[0]': 'value',
                'start': current.strftime('%Y-%m-%dT00'), 'end': chunk_end.strftime('%Y-%m-%dT00'),
                'sort[0][column]': 'period', 'sort[0][direction]': 'asc', 'length': 5000,
                **facets}, timeout=60)
            if r.status_code == 200:
                recs = r.json().get('response', {}).get('data', [])
                all_recs.extend(recs)
                if chunk % 10 == 0:
                    print(f"    chunk {chunk}: {len(all_recs)} records", flush=True)
        except Exception as e:
            print(f"    chunk {chunk} error: {e}", flush=True)
        current = chunk_end
        time.sleep(DELAY)

    if all_recs:
        df = pd.DataFrame(all_recs)
        df.to_csv(out_file, index=False)
        print(f"  DONE: {len(df)} records -> {os.path.basename(out_file)}", flush=True)
        return df
    return None

print("="*60, flush=True)
print("  FETCHING 2022 CAISO DATA", flush=True)
print("="*60, flush=True)

fetch('electricity/rto/fuel-type-data/data/', {'facets[respondent][]': 'CISO'},
      os.path.join(DATA, 'caiso_generation_by_fuel_2022.csv'),
      'Generation by fuel 2022', chunk_days=7)

fetch('electricity/rto/region-data/data/', {'facets[respondent][]': 'CISO'},
      os.path.join(DATA, 'caiso_region_data_2022.csv'),
      'Region data 2022', chunk_days=14)

print("\n  ALL 2022 DATA FETCHED", flush=True)
