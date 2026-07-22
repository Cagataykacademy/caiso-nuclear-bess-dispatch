"""
Fetch hourly temperature data for CAISO region from NOAA ISD Lite (AWS).
Station: Los Angeles International Airport (LAX) — representative of SP15 region.
USAF: 722950, WBAN: 23174
"""
import os, sys, io, gzip
import requests
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ISD Lite: free, no API key, direct download from NOAA
# Format: fixed-width, columns: year, month, day, hour, air_temp(*10), dewpoint, pressure, wind_dir, wind_speed, sky_cover, precip_1h, precip_6h
# Multiple California stations for better coverage
stations = {
    'LAX': ('722950', '23174'),  # Los Angeles (SP15)
    'SFO': ('724940', '23234'),  # San Francisco (NP15)
    'SAC': ('724839', '23232'),  # Sacramento (NP15)
    'FAT': ('723890', '93193'),  # Fresno (central valley)
}

all_dfs = []

for name, (usaf, wban) in stations.items():
    url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/2023/{usaf}-{wban}-2023.gz"
    print(f"  Fetching {name} ({usaf}-{wban})...", end=' ', flush=True)

    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            continue

        lines = gzip.decompress(r.content).decode('utf-8').strip().split('\n')
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                yr, mo, dy, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                temp_raw = int(parts[4])
                temp_c = temp_raw / 10.0 if temp_raw != -9999 else np.nan
                dew_raw = int(parts[5]) if len(parts) > 5 else -9999
                dew_c = dew_raw / 10.0 if dew_raw != -9999 else np.nan
                wind_raw = int(parts[8]) if len(parts) > 8 else -9999
                wind = wind_raw / 10.0 if wind_raw != -9999 else np.nan
                records.append({
                    'timestamp': pd.Timestamp(yr, mo, dy, hr),
                    f'temp_{name}_C': temp_c,
                    f'dewpoint_{name}_C': dew_c,
                    f'wind_{name}_ms': wind,
                })

        df = pd.DataFrame(records).set_index('timestamp')
        all_dfs.append(df)
        valid = df[f'temp_{name}_C'].notna().sum()
        print(f"{len(df)} records, {valid} valid temps, "
              f"range: {df[f'temp_{name}_C'].min():.1f} - {df[f'temp_{name}_C'].max():.1f} C")

    except Exception as e:
        print(f"Error: {e}")

if all_dfs:
    weather = pd.concat(all_dfs, axis=1)
    weather = weather.sort_index()

    # Compute CAISO-wide averages (population-weighted proxy)
    temp_cols = [c for c in weather.columns if c.startswith('temp_')]
    weather['temp_avg_C'] = weather[temp_cols].mean(axis=1)
    weather['temp_max_C'] = weather[temp_cols].max(axis=1)
    weather['temp_min_C'] = weather[temp_cols].min(axis=1)

    # Cooling/heating degree hours (base 18.3C = 65F)
    weather['CDH'] = (weather['temp_avg_C'] - 18.3).clip(lower=0)
    weather['HDH'] = (18.3 - weather['temp_avg_C']).clip(lower=0)

    # Fill small gaps
    weather = weather.interpolate(method='time', limit=3)

    out = os.path.join(DATA_DIR, "caiso_weather_2023.csv")
    weather.to_csv(out)

    print(f"\nSaved: {len(weather)} hourly records -> {out}")
    print(f"Columns: {list(weather.columns)}")
    print(f"Avg temp range: {weather['temp_avg_C'].min():.1f} - {weather['temp_avg_C'].max():.1f} C")
    print(f"CDH range: {weather['CDH'].min():.1f} - {weather['CDH'].max():.1f}")
else:
    print("\nERROR: No weather data downloaded")
