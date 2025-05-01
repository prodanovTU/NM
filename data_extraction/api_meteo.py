import requests
import pandas as pd
import time
from requests.exceptions import HTTPError

meteo_base_url = "https://archive-api.open-meteo.com"

endpoints = [
    "/meteo_history"
]

meteo_params = {
    "latitude": 42.70,  # София, България
    "longitude": 23.32,
    "hourly": "temperature_2m,shortwave_radiation,wind_speed_10m,wind_direction_10m,cloud_cover",
    
    "start_date": "2022-12-28",
    "end_date": "2024-12-31"
}

def fetch_data(endpoint, params):
    for attempt in range(3):
        try:
            if endpoint == "/meteo_history":
                response = requests.get(f"{meteo_base_url}/v1/archive", params=params)
                response.raise_for_status()
                data = response.json()
                
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                temperature = hourly.get("temperature_2m", [])
                radiation = hourly.get("shortwave_radiation", [])
                wind_speed = hourly.get("wind_speed_10m", [])
                wind_direction = hourly.get("wind_direction_10m", [])
                cloud_cover = hourly.get("cloud_cover", [])
                
                print(f"{endpoint} - Length of times: {len(times)}, temperature: {len(temperature)}, radiation: {len(radiation)}, wind_speed: {len(wind_speed)}, wind_direction: {len(wind_direction)}, cloud_cover: {len(cloud_cover)}")
                
                if not times or not any([temperature, radiation, wind_speed, wind_direction, cloud_cover]):
                    print(f"No valid data for {endpoint}")
                    return pd.DataFrame()
                
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(times),
                    "meteo_temperature_2m": temperature,
                    "meteo_shortwave_radiation": radiation,
                    "meteo_wind_speed_10m": wind_speed,
                    "meteo_wind_direction_10m": wind_direction,
                    "meteo_cloud_cover": cloud_cover
                })
                
                return df

        except HTTPError as e:
            print(f"Грешка при извличане на {endpoint} (опит {attempt + 1}/3): {e}")
            try:
                print(f"Response content: {response.text}")
            except:
                pass
            if attempt < 2:
                print(f"Повторен опит след {5 * (2 ** attempt)} секунди...")
                time.sleep(5 * (2 ** attempt))
            else:
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"Грешка при извличане на {endpoint} (опит {attempt + 1}/3): {e}")
            try:
                print(f"Response content: {response.text}")
            except:
                pass
            if attempt < 2:
                print(f"Повторен опит след {5 * (2 ** attempt)} секунди...")
                time.sleep(5 * (2 ** attempt))
            else:
                return pd.DataFrame()

# Извличане на данни
dataframes = {}
for endpoint in endpoints:
    df = fetch_data(endpoint, meteo_params)
    if not df.empty:
        dataframes[endpoint] = df
        print(f"Извлечени данни от {endpoint}: {len(df)} записа")
    time.sleep(1)

# Обединяване на данните
if dataframes:
    merged_df = dataframes["/meteo_history"].copy()
    
    merged_df = merged_df.sort_values("timestamp")
    
    merged_df = merged_df[merged_df['timestamp'] <= '2024-12-31 23:59:59']
    
    merged_df.to_csv("combined_meteo_data.csv", index=False)
    print(f"Обединените данни са запазени в combined_meteo_data.csv")
    print(f"Период: {merged_df['timestamp'].min()} до {merged_df['timestamp'].max()}")
    print(f"Колони: {list(merged_df.columns)}")
else:
    print("Няма извлечени данни.")