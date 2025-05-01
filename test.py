import requests
import pandas as pd
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pytz # Make sure pytz is installed: pip install pytz

# --- Parameters ---
API_KEY = 'YOUR_ACTUAL_API_KEY_HERE'  # <<< --- REPLACE THIS !!!
COUNTRY_CODE = '10Y1001A1001A82H'       # Germany-Luxembourg Bidding Zone EIC
TARGET_TIMEZONE = 'Europe/Berlin'       # Timezone for Germany

# Fetch data for April 26th, 2025
start_date = datetime(2025, 4, 26, 0, 0)
end_date = datetime(2025, 4, 27, 0, 0)

# Check if API key is set
if not API_KEY or API_KEY == 'YOUR_ACTUAL_API_KEY_HERE':
    print("ERROR: Please replace 'YOUR_ACTUAL_API_KEY_HERE' with your real ENTSO-E API key.")
    exit()

# --- Format Dates for API ---
start_str = start_date.strftime('%Y%m%d%H%M')
end_str = end_date.strftime('%Y%m%d%H%M')

# --- Construct API URL ---
url = (
    f"https://web-api.tp.entsoe.eu/api?securityToken={API_KEY}"
    f"&documentType=A44&in_Domain={COUNTRY_CODE}&out_Domain={COUNTRY_CODE}"
    f"&periodStart={start_str}&periodEnd={end_str}"
)

print(f"--- Step 1: Preparing Request ---")
print(f"Requesting Day-Ahead Prices (A44) for {COUNTRY_CODE} (Germany-Luxembourg)")
print(f"Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
print(f"API URL (Key Masked): {url.replace(API_KEY, '***')}")

try:
    # --- Step 2: Fetch Data ---
    print("\n--- Step 2: Sending Request to ENTSO-E API ---")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # --- Step 3: Print Raw Response ---
    print("\n--- Step 3: Received Response - Printing Raw Body ---")
    print("-----------------------------------------------------")
    raw_xml = response.text
    print(raw_xml)
    print("-----------------------------------------------------")

    # --- Step 4: Parse XML ---
    print("\n--- Step 4: Attempting to Parse XML ---")
    ns = {'ts': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3'} # Correct Namespace
    root = ET.fromstring(response.content)

    points_data = []
    processed_timeseries = 0
    found_hourly_timeseries = False

    for timeseries in root.findall('.//ts:TimeSeries', ns):
        processed_timeseries += 1
        period = timeseries.find('.//ts:Period', ns)
        if period is None:
            print("   - Skipping TimeSeries: No Period found.")
            continue

        resolution_str = period.find('.//ts:resolution', ns).text
        if resolution_str != 'PT60M':
            print(f"   - Skipping TimeSeries: Found resolution {resolution_str}, expecting PT60M.")
            continue
        found_hourly_timeseries = True
        print(f"   - Processing TimeSeries with resolution {resolution_str}")

        time_interval_node = period.find('.//ts:timeInterval', ns)
        start_dt_str = time_interval_node.find('.//ts:start', ns).text

        # *** FIX FOR TIMEZONE LOCALIZATION ERROR IS HERE ***
        try:
            # Parse ISO 8601 format (handles 'Z' for UTC automatically)
            period_start_dt_utc = pd.to_datetime(start_dt_str)

            # Add robustness check for timezone
            if period_start_dt_utc.tzinfo is None:
                # If pd.to_datetime didn't recognize TZ, assume and localize to UTC
                print(f"   - Warning: Parsed start time '{start_dt_str}' as naive, localizing to UTC.")
                period_start_dt_utc = period_start_dt_utc.tz_localize('UTC')
            elif str(period_start_dt_utc.tzinfo) != 'UTC':
                # If it was parsed with a different TZ, convert it to UTC
                print(f"   - Warning: Parsed start time '{start_dt_str}' with non-UTC timezone ({period_start_dt_utc.tzinfo}), converting to UTC.")
                period_start_dt_utc = period_start_dt_utc.tz_convert('UTC')
            # Now period_start_dt_utc is reliably UTC-aware

        except Exception as e:
             print(f"   - Warning: Could not parse period start time '{start_dt_str}'. Skipping Period. Error: {e}")
             continue
        # *** END OF FIX ***

        interval_timedelta = timedelta(hours=1) # We know it's hourly now

        # Extract points
        for point in period.findall('.//ts:Point', ns):
            try:
                position = int(point.find('ts:position', ns).text)
                price = float(point.find('ts:price.amount', ns).text)
                point_start_time_utc = period_start_dt_utc + (position - 1) * interval_timedelta
                points_data.append({'Datetime_UTC': point_start_time_utc, 'Price': price})
            except Exception as e:
                print(f"   - Warning: Could not process a Point element. Error: {e}")

    # --- Step 5: Process Results ---
    print("\n--- Step 5: Processing Parsed Data ---")
    if not found_hourly_timeseries:
         print(f"   - RESULT: Failed. Found {processed_timeseries} TimeSeries element(s), but NONE had PT60M (hourly) resolution.")
         reason_code = root.find('.//{*}Reason/{*}code')
         reason_text = root.find('.//{*}Reason/{*}text')
         if reason_code is not None and reason_text is not None:
              print(f"   - API also returned explicit reason: Code='{reason_code.text}', Text='{reason_text.text}'")

    elif not points_data:
        print("   - RESULT: Partial Success? Found hourly TimeSeries/Period structure but no price points were extracted (check parsing warnings above).")

    else:
        df_prices = pd.DataFrame(points_data)
        df_prices.rename(columns={'Datetime_UTC': 'Datetime'}, inplace=True)
        df_prices.sort_values('Datetime', inplace=True)
        df_prices['Datetime'] = df_prices['Datetime'].dt.tz_convert(TARGET_TIMEZONE)
        df_prices['Datetime'] = df_prices['Datetime'].dt.tz_localize(None)
        df_prices.drop_duplicates(subset=['Datetime'], keep='first', inplace=True)

        df_prices.set_index('Datetime', inplace=True)
        if not df_prices.empty:
             print("   - Note: Reindexing data to ensure complete hourly frequency.")
             full_range = pd.date_range(start=df_prices.index.min(), end=df_prices.index.max(), freq='H')
             df_prices = df_prices.reindex(full_range)
             df_prices['Price'] = df_prices['Price'].interpolate(method='linear')
             original_count = len(df_prices)
             df_prices.dropna(subset=['Price'], inplace=True)
             if len(df_prices) < original_count:
                  print(f"   - Note: Dropped {original_count - len(df_prices)} rows with NaN prices after reindexing/interpolation.")
        df_prices.reset_index(inplace=True)

        print(f"   - RESULT: Success! Parsed {len(df_prices)} HOURLY price points for Germany (DE-LU).")
        print("\n--- Final DataFrame ---")
        print(df_prices.to_string())


# --- Error Handling ---
except requests.exceptions.Timeout:
    print(f"\nERROR: Request timed out.")
except requests.exceptions.RequestException as e:
    print(f"\nERROR: Request failed for Germany: {e}")
    if e.response is not None:
         print("\n--- Error Response Body ---")
         try: print(e.response.text)
         except: print(e.response.content)
         print("-------------------------")
except ET.ParseError as e:
    print(f"\nERROR: Failed to parse XML response for Germany: {e}. Check raw body above.")
except Exception as e:
    print(f"\nERROR: An unexpected error occurred for Germany: {e}")