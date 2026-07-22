"""
Fetch Henry Hub natural gas spot price (daily) from EIA API.
Used to build a physically-grounded electricity price model.
"""
import os, sys, io, time, json, requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

API_KEY = os.environ.get("EIA_API_KEY", "DEMO_KEY")  # set your own key: https://www.eia.gov/opendata/register.php
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

print("Fetching Henry Hub natural gas spot price 2023...", flush=True)

all_recs = []
current = pd.Timestamp("2023-01-01")
end = pd.Timestamp("2024-01-01")

while current < end:
    chunk_end = min(current + pd.DateOffset(months=3), end)
    r = requests.get("https://api.eia.gov/v2/natural-gas/pri/fut/data/", params={
        "api_key": API_KEY, "frequency": "daily", "data[0]": "value",
        "facets[series][]": "RNGWHHD",
        "start": current.strftime("%Y-%m-%d"),
        "end": chunk_end.strftime("%Y-%m-%d"),
        "sort[0][column]": "period", "sort[0][direction]": "asc",
        "length": 5000,
    }, timeout=30)
    if r.status_code == 200:
        recs = r.json().get("response", {}).get("data", [])
        all_recs.extend(recs)
        print(f"  {current.date()}->{chunk_end.date()}: +{len(recs)} = {len(all_recs)}", flush=True)
    current = chunk_end
    time.sleep(3)

df = pd.DataFrame(all_recs)
out = os.path.join(DATA_DIR, "henry_hub_gas_price_2023.csv")
df.to_csv(out, index=False)
print(f"\nDONE: {len(df)} records -> {out}", flush=True)
if "value" in df.columns:
    v = pd.to_numeric(df["value"], errors="coerce")
    print(f"Price range: ${v.min():.2f} - ${v.max():.2f} / MMBtu", flush=True)
    print(f"Mean: ${v.mean():.2f} / MMBtu", flush=True)
