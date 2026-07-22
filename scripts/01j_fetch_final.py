"""
Fast download: CAISO generation by fuel type (EIA API with real key).
Also re-downloads region data for completeness.
"""
import os, sys, io, time, json, requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")  # set your own key: https://www.eia.gov/opendata/register.php
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DELAY = 3

def eia_fetch_all(endpoint, facets, out_file, desc, chunk_days=10):
    print(f"\n{'='*60}", flush=True)
    print(f"  {desc}", flush=True)

    if os.path.exists(out_file):
        existing = pd.read_csv(out_file)
        if len(existing) > 5000:
            print(f"  Already have {len(existing)} records, skipping.", flush=True)
            return existing

    all_recs = []
    current = pd.Timestamp('2023-01-01')
    end = pd.Timestamp('2024-01-01')
    chunk = 0

    while current < end:
        chunk_end = min(current + pd.DateOffset(days=chunk_days), end)
        chunk += 1
        url = f'https://api.eia.gov/v2/{endpoint}'
        params = {
            'api_key': API_KEY, 'frequency': 'hourly', 'data[0]': 'value',
            'start': current.strftime('%Y-%m-%dT00'),
            'end': chunk_end.strftime('%Y-%m-%dT00'),
            'sort[0][column]': 'period', 'sort[0][direction]': 'asc',
            'length': 5000,
        }
        params.update(facets)

        ok = False
        for att in range(3):
            try:
                r = requests.get(url, params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(30*(att+1))
                    continue
                r.raise_for_status()
                d = r.json()
                recs = d.get('response', {}).get('data', [])
                all_recs.extend(recs)
                print(f"  #{chunk:>3}: {current.date()}->{chunk_end.date()} +{len(recs):>4} = {len(all_recs):>6}", flush=True)
                ok = True
                break
            except Exception as e:
                print(f"  #{chunk} err: {e}", flush=True)
                time.sleep(10)
        if not ok:
            print(f"  #{chunk} FAILED", flush=True)

        current = chunk_end
        time.sleep(DELAY)

    if all_recs:
        df = pd.DataFrame(all_recs)
        df.to_csv(out_file, index=False)
        print(f"\n  SAVED: {len(df)} records -> {os.path.basename(out_file)}", flush=True)
        return df
    return None

print("="*60, flush=True)
print("  FAST EIA DATA DOWNLOAD", flush=True)
print("="*60, flush=True)

# 1. Generation by fuel type (SUN, WND, NUC, NG, WAT, etc.)
gen = eia_fetch_all(
    'electricity/rto/fuel-type-data/data/',
    {'facets[respondent][]': 'CISO'},
    os.path.join(DATA_DIR, 'caiso_generation_by_fuel_2023.csv'),
    'CAISO Generation by Fuel Type', chunk_days=7
)
if gen is not None:
    if 'type-name' in gen.columns:
        print("\n  Fuel breakdown:", flush=True)
        for ft, cnt in gen['type-name'].value_counts().items():
            print(f"    {ft}: {cnt} records", flush=True)

# 2. Region data (demand + generation + interchange) - refresh with better key
gen2 = eia_fetch_all(
    'electricity/rto/region-data/data/',
    {'facets[respondent][]': 'CISO'},
    os.path.join(DATA_DIR, 'caiso_region_data_2023.csv'),
    'CAISO Region Data (demand/gen/interchange)', chunk_days=14
)
if gen2 is not None:
    if 'type-name' in gen2.columns:
        print("\n  Data types:", flush=True)
        for ft, cnt in gen2['type-name'].value_counts().items():
            print(f"    {ft}: {cnt} records", flush=True)

print("\n" + "="*60, flush=True)
print("  ALL DONE", flush=True)
print("="*60, flush=True)
