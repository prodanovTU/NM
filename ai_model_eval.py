import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pytz # Make sure pytz is installed: pip install pytz
import time # For optional delays
import os # For model caching path
import tensorflow as tf # For model loading

# Consider StandardScaler as an alternative if outliers are an issue
from sklearn.preprocessing import MinMaxScaler #, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Added r2_score
# TensorFlow / Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
# Try different backends if plotting fails
try:
    # Using 'Agg' for non-interactive environments first, suitable for servers
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Agg backend not available, trying default.")
    matplotlib.use(None) # Use default backend
    import matplotlib.pyplot as plt


# Disable overly specific TF warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# =============================================================================
# TASK 1: Research available free data, which can be extracted via API
# =============================================================================

# ========== 1A. FETCH ELECTRICITY PRICE DATA FROM ENTSO-E (A44) ==========
def fetch_entsoe_prices(api_key: str,
                        start_date: datetime,
                        end_date: datetime,
                        country_code: str, # Now passed explicitly
                        target_timezone: str # Now passed explicitly
                        ) -> pd.DataFrame:
    """
    Fetches, parses, and converts HOURLY (PT60M) Day-Ahead (A44) electricity
    price data from ENTSO-E API.

    Args:
        api_key: Your personal ENTSO-E API security token.
        start_date: The start datetime for the data fetch period.
        end_date: The end datetime for the data fetch period (exclusive).
        country_code: The ENTSO-E bidding zone EIC code.
        target_timezone: The desired local timezone for the output DataFrame.

    Returns:
        A pandas DataFrame with 'Datetime' and 'Price' columns, or an empty
        DataFrame if fetching/parsing fails or no HOURLY data is found.
        'Datetime' column is timezone-naive, representing the local time.
    """
    # 1. --- Input Validation ---
    if not api_key or api_key == 'YOUR_ENTSOE_API_KEY':
        logging.error("ENTSO-E API Key is missing or is the placeholder value.")
        raise ValueError("ENTSO-E API Key is missing or is the placeholder value.")
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
         logging.error("start_date and end_date must be datetime objects.")
         raise TypeError("start_date and end_date must be datetime objects.")
    if end_date <= start_date:
         logging.warning(f"end_date ({end_date}) is not after start_date ({start_date}), request might yield no data.")

    # 2. --- Prepare API Request ---
    start_str = start_date.strftime('%Y%m%d%H%M')
    end_str = end_date.strftime('%Y%m%d%H%M')
    url = (
        f"https://web-api.tp.entsoe.eu/api?securityToken={api_key}"
        f"&documentType=A44&in_Domain={country_code}&out_Domain={country_code}"
        f"&periodStart={start_str}&periodEnd={end_str}"
    )
    logging.info(f"Attempting ENTSO-E fetch: Area='{country_code}', Period='{start_date.date()}':'{end_date.date()}'")
    logging.debug(f"Request URL: {url.replace(api_key, '***')}")

    # Namespace confirmed from successful German response
    ns = {'ts': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3'}

    # 3. --- Fetch Data ---
    try:
        response = requests.get(url, timeout=60)
        logging.debug(f"API Response Status Code: {response.status_code}")
        response.raise_for_status()

        # --- Optional: Print Raw Response (Uncomment for deep debugging) ---
        # print("\n-----------------------------------------")
        # print("--- RAW ENTSO-E API Response Body Start ---")
        # print("-----------------------------------------")
        # try:
        #     print(response.text) # Print decoded text
        # except Exception as decode_err:
        #     print(f"(Could not decode response.text: {decode_err}), printing bytes:")
        #     print(response.content) # Print raw bytes if decoding fails
        # print("---------------------------------------")
        # print("--- RAW ENTSO-E API Response Body End ---")
        # print("---------------------------------------\n")
        # --- End Optional Print ---


        # 4. --- Parse XML Response ---
        root = ET.fromstring(response.content)
        points_data = []
        processed_timeseries = 0
        found_hourly_timeseries = False

        for timeseries in root.findall('.//ts:TimeSeries', ns):
            processed_timeseries += 1
            period = timeseries.find('.//ts:Period', ns)
            if period is None:
                logging.debug("Skipping TimeSeries: No Period found.")
                continue

            resolution_str = period.find('.//ts:resolution', ns).text
            # --- Filter for Hourly Data ---
            if resolution_str != 'PT60M':
                logging.debug(f"Skipping TimeSeries: Found resolution {resolution_str}, expecting PT60M.")
                continue
            # --- End Filter ---

            found_hourly_timeseries = True # Mark that we are processing an hourly series
            logging.debug(f"Processing TimeSeries with resolution {resolution_str}")

            time_interval_node = period.find('.//ts:timeInterval', ns)
            if time_interval_node is None:
                 logging.warning("Found Period block without timeInterval. Skipping Period.")
                 continue
            start_dt_str = time_interval_node.find('.//ts:start', ns).text

            # Parse start time, ensure it's UTC aware
            try:
                period_start_dt_utc = pd.to_datetime(start_dt_str) # Handles 'Z' automatically
                if period_start_dt_utc.tzinfo is None:
                    logging.warning(f"Parsed start time '{start_dt_str}' as naive, localizing to UTC.")
                    period_start_dt_utc = period_start_dt_utc.tz_localize('UTC')
                elif str(period_start_dt_utc.tzinfo) != 'UTC':
                    logging.warning(f"Parsed start time '{start_dt_str}' with non-UTC timezone ({period_start_dt_utc.tzinfo}), converting to UTC.")
                    period_start_dt_utc = period_start_dt_utc.tz_convert('UTC')
            except Exception as e:
                 logging.warning(f"Could not parse period start time '{start_dt_str}'. Skipping Period. Error: {e}")
                 continue

            interval_timedelta = timedelta(hours=1) # We filtered for PT60M

            # Extract price points
            points_in_period = 0
            for point in period.findall('.//ts:Point', ns):
                try:
                    position = int(point.find('ts:position', ns).text)
                    price = float(point.find('ts:price.amount', ns).text)
                    point_start_time_utc = period_start_dt_utc + (position - 1) * interval_timedelta
                    points_data.append({'Datetime_UTC': point_start_time_utc, 'Price': price})
                    points_in_period += 1
                except (AttributeError, ValueError, TypeError, ET.ParseError) as e: # Added ParseError
                    logging.warning(f"Could not parse Point data (Pos/Price). Skipping point. Error: {e}")
                    continue # Skip this malformed point
            logging.debug(f"Extracted {points_in_period} points from this Period.")

        # --- Handle Cases Where No Data Was Found ---
        if not found_hourly_timeseries:
             reason_code = root.find('.//{*}Reason/{*}code') # Wildcard NS search
             reason_text = root.find('.//{*}Reason/{*}text')
             if reason_code is not None and reason_text is not None:
                  logging.warning(f"ENTSO-E API Reason Code='{reason_code.text}', Text='{reason_text.text}' (Likely no data matching request).")
             else:
                  # Check if any TimeSeries were found at all (e.g., only PT15M)
                  if processed_timeseries > 0:
                       logging.warning(f"Found {processed_timeseries} TimeSeries block(s), but none had PT60M (hourly) resolution.")
                  else:
                       logging.warning("No TimeSeries data blocks found at all in the ENTSO-E response XML.")
             return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})

        if not points_data:
            logging.warning("Hourly TimeSeries/Period blocks found, but failed to extract any valid Price Points.")
            return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})

        # 5. --- Convert to DataFrame and Process ---
        logging.info(f"Successfully extracted {len(points_data)} raw hourly data points from API response.")
        df_prices = pd.DataFrame(points_data)
        df_prices.rename(columns={'Datetime_UTC': 'Datetime'}, inplace=True)
        df_prices.sort_values('Datetime', inplace=True)

        print(df_prices.columns.values)

        try:
            df_prices['Datetime'] = df_prices['Datetime'].dt.tz_convert(target_timezone)
        except pytz.UnknownTimeZoneError:
            logging.error(f"Unknown target timezone: '{target_timezone}'. Using UTC.")
        except Exception as e:
             logging.error(f"Could not convert timezone to {target_timezone}. Error: {e}. Keeping UTC.")

        df_prices['Datetime'] = df_prices['Datetime'].dt.tz_localize(None)
        df_prices.drop_duplicates(subset=['Datetime'], keep='first', inplace=True)
        df_prices.set_index('Datetime', inplace=True)

        # Optional Reindexing/Interpolation
        if not df_prices.empty:
             try:
                 logging.debug("Attempting reindexing to ensure complete hourly frequency.")
                 full_range = pd.date_range(start=df_prices.index.min(), end=df_prices.index.max(), freq='H')
                 df_prices = df_prices.reindex(full_range)
                 df_prices['Price'] = df_prices['Price'].interpolate(method='linear')
                 original_count = len(df_prices)
                 df_prices.dropna(subset=['Price'], inplace=True)
                 if len(df_prices) < original_count:
                      logging.debug(f"Dropped {original_count - len(df_prices)} rows with NaN prices after reindexing/interpolation.")
             except Exception as e:
                  logging.warning(f"Could not reindex/interpolate data. Error: {e}. Returning potentially gappy data.")

        df_prices = df_prices.reset_index()  # Moves old index ('Datetime') to column 'index'
        df_prices.rename(columns={'index': 'Datetime'}, inplace=True)  # Rename the 'index' column to 'Datetime'
        if df_prices.empty:
             logging.warning("No price points remaining after processing.")
        else:
             logging.info(f"Finished processing. Returning DataFrame with {len(df_prices)} rows.")
        return df_prices

    # 6. --- Exception Handling ---
    except requests.exceptions.Timeout:
        logging.error(f"ENTSO-E API request timed out for URL: {url.replace(api_key, '***')}")
        return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})
    except requests.exceptions.RequestException as e:
        logging.error(f"ENTSO-E API request error: {e}")
        if e.response is not None:
             logging.error(f"Error Response Status Code: {e.response.status_code}")
             try: logging.error(f"Error Response Body: {e.response.text}")
             except Exception: logging.error("Error Response Body could not be decoded as text.")
        return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})
    except ET.ParseError as e:
        logging.error(f"ENTSO-E XML parsing error: {e}. Invalid XML received.")
        return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})
    except Exception as e:
        logging.error(f"An unexpected error occurred in fetch_entsoe_prices: {e}", exc_info=True)
        return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})
# ========== 1B. FETCH GENERATION DATA FROM ENTSO-E (A75 - Placeholder) ==========
def fetch_entsoe_generation(api_key: str, start_date: datetime, end_date: datetime, country_code: str, target_timezone: str) -> pd.DataFrame:
    logging.debug("Skipping Generation fetch (Placeholder).")
    return pd.DataFrame({'Datetime': pd.to_datetime([])})

# ========== 1C. FETCH LOAD DATA FROM ENTSO-E (A65 - Placeholder) ==========
def fetch_entsoe_load(api_key: str, start_date: datetime, end_date: datetime, country_code: str, target_timezone: str) -> pd.DataFrame:
    logging.debug("Skipping Load fetch (Placeholder).")
    return pd.DataFrame({'Datetime': pd.to_datetime([])})

# ========== 1D. FETCH WEATHER DATA FROM OPEN-METEO (HISTORICAL) ==========
def fetch_openmeteo_data(start_date, end_date, latitude, longitude, timezone):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    start_str = start_date.strftime('%Y-%m-%d')
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if end_date >= today_start:
        effective_end_date_str = (today_start - timedelta(days=1)).strftime('%Y-%m-%d')
        logging.debug(f"Requested end date {end_date.date()} >= today. Capping Open-Meteo history request to {effective_end_date_str}.")
    else:
        effective_end_date_str = end_date.strftime('%Y-%m-%d')
    logging.info(f"Fetching Open-Meteo HISTORY: Lat={latitude:.2f}, Lon={longitude:.2f}, Period='{start_str}':'{effective_end_date_str}'")
    url = (
        f"{base_url}?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_str}&end_date={effective_end_date_str}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,cloudcover"
        f"&timezone={timezone.replace('/', '%2F')}"
    )
    logging.debug(f"Open-Meteo History URL: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'hourly' not in data or not data['hourly'].get('time'): return pd.DataFrame()
        hourly_data = data['hourly']
        timestamps = pd.to_datetime(hourly_data['time']).tz_localize(None)
        df_weather = pd.DataFrame({
            'Datetime': timestamps, 'Temperature': hourly_data['temperature_2m'],
            'Humidity': hourly_data['relativehumidity_2m'], 'Wind Speed': hourly_data['windspeed_10m'],
            'Cloudiness': hourly_data['cloudcover']
        })
        df_weather = df_weather[(df_weather['Datetime'] >= start_date) & (df_weather['Datetime'] < end_date)].copy()
        df_weather.dropna(inplace=True)
        logging.info(f"Successfully processed {len(df_weather)} historical weather points.")
        return df_weather
    except requests.exceptions.RequestException as e: logging.error(f"Open-Meteo History API request error: {e}")
    except Exception as e: logging.error(f"Error processing Open-Meteo history data: {e}", exc_info=True)
    return pd.DataFrame()


# ========== 1E. FETCH WEATHER FORECAST FROM OPEN-METEO (By Date Range) ==========
def fetch_openmeteo_forecast(latitude, longitude, timezone, start_date_forecast, end_date_forecast):
    base_url = "https://api.open-meteo.com/v1/forecast"
    start_str = start_date_forecast.strftime('%Y-%m-%d')
    end_str = end_date_forecast.strftime('%Y-%m-%d')
    logging.info(f"Fetching Open-Meteo FORECAST: Lat={latitude:.2f}, Lon={longitude:.2f}, Period='{start_str}' to '{end_str}'")
    url = (
        f"{base_url}?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,cloudcover"
        f"&start_date={start_str}&end_date={end_str}"
        f"&timezone={timezone.replace('/', '%2F')}"
    )
    logging.debug(f"Open-Meteo Forecast URL: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'hourly' not in data or not data['hourly'].get('time'): return pd.DataFrame()
        hourly_data = data['hourly']
        timestamps = pd.to_datetime(hourly_data['time']).tz_localize(None)
        df_forecast = pd.DataFrame({
            'Datetime': timestamps, 'Temperature': hourly_data['temperature_2m'],
            'Humidity': hourly_data['relativehumidity_2m'], 'Wind Speed': hourly_data['windspeed_10m'],
            'Cloudiness': hourly_data['cloudcover']
        })
        logging.info(f"Successfully fetched {len(df_forecast)} weather forecast points.")
        return df_forecast
    except requests.exceptions.RequestException as e: logging.error(f"Open-Meteo Forecast API request error: {e}")
    except Exception as e: logging.error(f"Error processing Open-Meteo forecast data: {e}", exc_info=True)
    return pd.DataFrame()

# ========== 2. PREPROCESSING AND FEATURE ENGINEERING ==========
def preprocess_data(df, scaler=None, fit_scaler=False):
    df_processed = df.copy()
    if 'Datetime' not in df_processed.columns: raise ValueError("'Datetime' column missing.")
    df_processed['Datetime'] = pd.to_datetime(df_processed['Datetime'])
    df_processed.set_index('Datetime', inplace=True)
    df_processed.sort_index(inplace=True)
    logging.info(f"Preprocessing data from {df_processed.index.min()} to {df_processed.index.max()}")

    # Feature Engineering
    df_processed['Hour'] = df_processed.index.hour
    df_processed['DayOfWeek'] = df_processed.index.dayofweek
    df_processed['Month'] = df_processed.index.month
    df_processed['IsWeekend'] = df_processed['DayOfWeek'].isin([5, 6]).astype(int)

    # Define potential columns dynamically
    price_col_original = 'Price'; scaled_price_col_name = 'ScaledPrice'
    base_feature_cols = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend']
    weather_cols = ['Temperature', 'Humidity', 'Wind Speed', 'Cloudiness']
    generation_cols = ['Solar_Gen', 'Wind_Gen', 'Hydro_Gen', 'Nuclear_Gen', 'Other_Gen']
    load_cols = ['TotalLoad', 'LoadForecast']

    present_weather_cols = [col for col in weather_cols if col in df_processed.columns]
    present_generation_cols = [col for col in generation_cols if col in df_processed.columns]
    present_load_cols = [col for col in load_cols if col in df_processed.columns]

    price_col_current = []
    if price_col_original in df_processed.columns:
        df_processed.rename(columns={price_col_original: scaled_price_col_name}, inplace=True)
        price_col_current = [scaled_price_col_name]
    else: scaled_price_col_name = None

    feature_cols = present_weather_cols + present_generation_cols + present_load_cols + base_feature_cols
    columns_to_scale = price_col_current + feature_cols
    columns_to_scale = [col for col in columns_to_scale if col in df_processed.columns]
    feature_cols = [col for col in feature_cols if col in df_processed.columns]

    logging.info(f"Columns identified for scaling: {columns_to_scale}")
    logging.info(f"Feature columns identified: {feature_cols}")

    if not columns_to_scale: return df_processed, scaler, [], scaled_price_col_name, feature_cols

    # Handle Missing Values
    nan_counts = df_processed[columns_to_scale].isnull().sum()
    if nan_counts.sum() > 0:
         logging.warning(f"NaNs detected before scaling:\n{nan_counts[nan_counts > 0]}")
         logging.info("Filling NaNs using ffill/bfill/0.")
         df_processed.fillna(method='ffill', inplace=True); df_processed.fillna(method='bfill', inplace=True); df_processed.fillna(0, inplace=True)

    # Scaling
    if not price_col_current: return df_processed, scaler, [], None, feature_cols
    try:
        if fit_scaler:
            if scaler is None: scaler = MinMaxScaler()
            logging.info(f"Fitting scaler on {len(columns_to_scale)} columns.")
            df_processed[columns_to_scale] = scaler.fit_transform(df_processed[columns_to_scale])
        elif scaler is not None:
            if len(columns_to_scale) != scaler.n_features_in_: raise ValueError(f"Scaler mismatch: {len(columns_to_scale)} vs {scaler.n_features_in_}")
            logging.info(f"Transforming data using existing scaler.")
            df_processed[columns_to_scale] = scaler.transform(df_processed[columns_to_scale])
        else: logging.warning("Scaler not provided; fit_scaler=False. Data not scaled.")
    except Exception as e:
         logging.error(f"Error during scaling: {e}.", exc_info=True)
         return df_processed, None, [], scaled_price_col_name, feature_cols

    scaled_feature_cols = feature_cols
    logging.info("Preprocessing finished.")
    return df_processed, scaler, columns_to_scale, scaled_price_col_name, scaled_feature_cols

# ========== Function to Create Sequences ==========
def create_sequences(data, price_col_name, feature_col_names, seq_length):
    """ Creates sequences for LSTM input (price + features) and target (price). """
    X_price_list, X_features_list, y_list = [], [], []
    if price_col_name not in data.columns: raise ValueError(f"Price column '{price_col_name}' not found.")
    present_feature_cols = [col for col in feature_col_names if col in data.columns]
    if len(present_feature_cols) != len(feature_col_names): logging.warning(f"Using features: {present_feature_cols}")
    price_values = data[price_col_name].values
    feature_values = data[present_feature_cols].values if present_feature_cols else None
    use_features = feature_values is not None and len(present_feature_cols) > 0
    logging.info(f"Creating sequences with length {seq_length}...")
    if len(data) <= seq_length: return np.array([]), None, np.array([])
    for i in range(len(data) - seq_length):
        X_price_list.append(price_values[i : i + seq_length])
        y_list.append(price_values[i + seq_length])
        if use_features: X_features_list.append(feature_values[i : i + seq_length])
    X_price = np.array(X_price_list); y = np.array(y_list)
    X_features = np.array(X_features_list) if use_features and X_features_list else None
    if X_price.ndim == 2: X_price = X_price.reshape((*X_price.shape, 1))
    logging.info(f"Sequence shapes: X_price={X_price.shape}, X_features={X_features.shape if use_features else 'None'}, y={y.shape}")
    return X_price, X_features, y

# =============================================================================
# TASK 2: Develop model(s) for price forecasting using Neural Networks
# =============================================================================

# ========== 3. BUILD LSTM MODEL ==========
def build_model(input_shape_price, input_shape_features):
    """ Builds the multi-input LSTM model, adapting to feature presence. """
    input_price = Input(shape=input_shape_price, name='price_input')
    lstm_price_1 = LSTM(units=64, return_sequences=True, name='lstm_price_1')(input_price)
    dropout_price_1 = Dropout(0.2, name='dropout_price_1')(lstm_price_1)
    lstm_price_2 = LSTM(units=32, name='lstm_price_2')(dropout_price_1)
    dropout_price_2 = Dropout(0.2, name='dropout_price_2')(lstm_price_2)
    price_branch_output = dropout_price_2
    inputs = [input_price]
    features_exist = isinstance(input_shape_features, (list, tuple)) and len(input_shape_features) > 1 and input_shape_features[1] > 0
    if features_exist:
        logging.info(f"Building model with Feature input branch. Shape: {input_shape_features}")
        input_features = Input(shape=input_shape_features, name='features_input')
        feature_lstm_units_1 = max(32, input_shape_features[1] * 2)
        feature_lstm_units_2 = max(16, input_shape_features[1])
        lstm_features_1 = LSTM(units=feature_lstm_units_1, return_sequences=True, name='lstm_features_1')(input_features)
        dropout_features_1 = Dropout(0.2, name='dropout_features_1')(lstm_features_1)
        lstm_features_2 = LSTM(units=feature_lstm_units_2, name='lstm_features_2')(dropout_features_1)
        dropout_features_2 = Dropout(0.2, name='dropout_features_2')(lstm_features_2)
        features_branch_output = dropout_features_2
        merged = Concatenate(name='concatenate_branches')([price_branch_output, features_branch_output])
        inputs.append(input_features)
    else: merged = price_branch_output; logging.info("Building model with Price input ONLY.")
    dense_merged = Dense(units=32, activation='relu', name='dense_merged')(merged)
    dropout_merged = Dropout(0.2, name='dropout_merged')(dense_merged)
    dense_output = Dense(units=1, name='output_dense')(dropout_merged)
    model = Model(inputs=inputs, outputs=dense_output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("\nModel Summary:"); model.summary(print_fn=logging.info)
    return model

# =============================================================================
# TASK 3: Visualization of results
# =============================================================================

# ========== 4A. PLOT HISTORICAL EVALUATION RESULTS ==========
def plot_predictions(true_index, true_values, predicted_values, title='Electricity Price Prediction'):
    if len(true_index) != len(true_values) or len(true_index) != len(predicted_values): return
    plt.figure(figsize=(15, 7)); plt.plot(true_index, true_values, 'b-', label='Actual Price', alpha=0.8, linewidth=1.5)
    plt.plot(true_index, predicted_values, 'r--', label='Predicted Price', alpha=0.8, linewidth=1.2)
    plt.title(f"{title} (Test Set Evaluation)"); plt.xlabel('Time'); plt.ylabel('Price (EUR/MWh)')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
    plot_filename = "historical_evaluation_plot.png"
    try: plt.savefig(plot_filename); logging.info(f"Historical plot saved: {plot_filename}")
    except Exception as e: logging.error(f"Could not save historical plot: {e}")
    finally: plt.close()

# ========== 4B. PLOT FUTURE PREDICTION RESULTS ==========
def plot_future_predictions(df_future, title='Future Electricity Price Prediction'):
    if df_future.empty or 'PredictedPrice' not in df_future.columns or 'Datetime' not in df_future.columns: return
    plt.figure(figsize=(15, 7))
    plt.plot(df_future['Datetime'], df_future['PredictedPrice'], 'orange', marker='o', markersize=4, linestyle='-', label='Future Predicted Price')
    plt.title(title); plt.xlabel('Time'); plt.ylabel('Predicted Price (EUR/MWh)')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
    plot_filename = "future_prediction_plot.png"
    try: plt.savefig(plot_filename); logging.info(f"Future prediction plot saved: {plot_filename}")
    except Exception as e: logging.error(f"Could not save future plot: {e}")
    finally: plt.close()

# =============================================================================
# TASK 4: Evaluate accuracy / Calculate error (Refactored Functions)
# =============================================================================

# ========== 5A. TRAIN MODEL ========== ## REFACTORED ##
def train_model(model, X_train_price, X_train_features, y_train, X_val_price, X_val_features, y_val, epochs=100, batch_size=32):
    """ Trains the LSTM model with early stopping. """
    train_inputs = [X_train_price]
    val_inputs = [X_val_price]
    features_exist = X_train_features is not None and X_train_features.ndim == 3 and X_train_features.shape[2] > 0
    if features_exist:
        train_inputs.append(X_train_features); val_inputs.append(X_val_features)
        logging.info("Training model with Price + Features.")
    else: logging.info("Training model with Price ONLY.")

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    logging.info("--- Starting Model Training ---")
    history = model.fit(train_inputs, y_train, epochs=epochs, batch_size=batch_size, validation_data=(val_inputs, y_val), callbacks=[early_stopping], verbose=1)
    logging.info(f"--- Model Training Finished (Epochs: {len(history.history['loss'])}) ---")
    return model

# ========== 5B. EVALUATE MODEL ========== ## REFACTORED ##
def evaluate_model(model, scaler,
                   X_test_price, X_test_features, y_test_scaled,
                   test_data_index, scaled_columns_list,
                   accuracy_tolerance_eur=15.0):
    """ Evaluates the trained model using MAE, RMSE, R2, Acc w/ Tolerance. """
    test_inputs = [X_test_price]
    features_exist = X_test_features is not None and X_test_features.ndim == 3 and X_test_features.shape[2] > 0
    if features_exist:
        test_inputs.append(X_test_features)
        logging.info("Evaluating model with Price + Features on test set.")
    else: logging.info("Evaluating model with Price ONLY on test set.")

    logging.info("\n--- Evaluating on Test Set ---")
    try: predictions_scaled = model.predict(test_inputs)
    except Exception as e: logging.error(f"Model prediction failed during evaluation: {e}", exc_info=True); return None, None, None, None

    # Inverse Scaling
    if scaler is None or not scaled_columns_list: return None, None, None, None
    price_col_name_scaled = 'ScaledPrice'
    try: price_col_index = scaled_columns_list.index(price_col_name_scaled)
    except ValueError: logging.error(f"'{price_col_name_scaled}' not in scaled_columns_list."); return None, None, None, None
    num_scaled_features = len(scaled_columns_list)
    if num_scaled_features != scaler.n_features_in_: logging.error(f"Scaler mismatch"); return None, None, None, None

    try: # Rescale
        dummy_predictions = np.zeros((len(predictions_scaled), num_scaled_features)); dummy_predictions[:, price_col_index] = predictions_scaled.flatten()
        predictions_rescaled = scaler.inverse_transform(dummy_predictions)[:, price_col_index]
        dummy_y_test = np.zeros((len(y_test_scaled), num_scaled_features)); dummy_y_test[:, price_col_index] = y_test_scaled.flatten()
        y_test_rescaled = scaler.inverse_transform(dummy_y_test)[:, price_col_index]
    except Exception as e: logging.error(f"Inverse scaling failed during evaluation: {e}", exc_info=True); return None, None, None, None

    # Calculate Metrics
    logging.info("\n--- Test Set Evaluation Metrics ---")
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled); logging.info(f"Average Price Delta (MAE):  {mae:.3f} (EUR/MWh)")
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled)); logging.info(f"RMSE:                       {rmse:.3f} (EUR/MWh)")
    r2 = r2_score(y_test_rescaled, predictions_rescaled); logging.info(f"R-squared (R²):             {r2:.3f}")
    absolute_errors = np.abs(y_test_rescaled - predictions_rescaled)
    within_tolerance_count = np.sum(absolute_errors <= accuracy_tolerance_eur)
    accuracy_within_tolerance = (within_tolerance_count / len(y_test_rescaled)) * 100 if len(y_test_rescaled) > 0 else 0
    logging.info(f"Accuracy (within +/- {accuracy_tolerance_eur:.1f} EUR): {accuracy_within_tolerance:.2f}%")

    # Plotting (Task 3 for evaluation)
    if len(test_data_index) == len(y_test_rescaled): plot_predictions(test_data_index, y_test_rescaled, predictions_rescaled, title='Historical Price Prediction')
    else: logging.warning(f"Length mismatch plotting test data.")

    return rmse, mae, r2, accuracy_within_tolerance

# ========== 5C. MAKE FUTURE PREDICTIONS ==========
def predict_future_prices(model,
                          scaler,
                          last_known_sequence_scaled_df,
                          weather_forecast_df,
                          n_future_hours,
                          sequence_length,
                          price_col_name_scaled,
                          all_feature_cols,
                          scaled_cols_list):
    """ Predicts future prices iteratively using weather forecasts. """
    logging.info(f"Starting future prediction for {n_future_hours} hours...")
    # Input Validation
    if not isinstance(last_known_sequence_scaled_df, pd.DataFrame) or len(last_known_sequence_scaled_df) != sequence_length: return pd.DataFrame()
    if price_col_name_scaled not in last_known_sequence_scaled_df.columns: return pd.DataFrame()
    if scaler is None or not scaled_cols_list: return pd.DataFrame()

    # Prepare Weather Forecast
    weather_forecast_df = weather_forecast_df.copy()
    weather_forecast_df['Datetime'] = pd.to_datetime(weather_forecast_df['Datetime'])
    weather_forecast_df.set_index('Datetime', inplace=True)

    # Generate Future Timestamps and Time Features
    last_historical_dt = last_known_sequence_scaled_df.index[-1]
    future_datetimes = pd.date_range(start=last_historical_dt + timedelta(hours=1), periods=n_future_hours, freq='h')
    future_df = pd.DataFrame(index=future_datetimes)
    future_df['Hour'] = future_df.index.hour; future_df['DayOfWeek'] = future_df.index.dayofweek
    future_df['Month'] = future_df.index.month; future_df['IsWeekend'] = future_df['DayOfWeek'].isin([5, 6]).astype(int)

    # Merge and Scale Future Features
    weather_cols_needed = [col for col in ['Temperature', 'Humidity', 'Wind Speed', 'Cloudiness'] if col in all_feature_cols]
    present_feature_cols = weather_cols_needed + [col for col in ['Hour','DayOfWeek','Month','IsWeekend'] if col in all_feature_cols]
    future_df = future_df.merge(weather_forecast_df[weather_cols_needed], left_index=True, right_index=True, how='left')

    if future_df[weather_cols_needed].isnull().any().any():
         logging.warning("Missing weather forecast data after merge. Filling with ffill/bfill/0.")
         future_df[weather_cols_needed] = future_df[weather_cols_needed].ffill().bfill().fillna(0)

    temp_scaling_df = pd.DataFrame(0.0, index=future_df.index, columns=scaled_cols_list)
    available_future_features = [col for col in present_feature_cols if col in future_df.columns]
    for col in available_future_features: temp_scaling_df[col] = future_df[col]
    try:
        scaled_future_vals_array = scaler.transform(temp_scaling_df)
        scaled_future_features_df = pd.DataFrame(scaled_future_vals_array, index=future_df.index, columns=scaled_cols_list)
        scaled_future_features_df = scaled_future_features_df[present_feature_cols]
    except Exception as e: logging.error(f"Error scaling future features: {e}.", exc_info=True); return pd.DataFrame()

    # Iterative Prediction Loop
    initial_sequence_cols = [price_col_name_scaled] + present_feature_cols
    if not all(col in last_known_sequence_scaled_df.columns for col in initial_sequence_cols):
         logging.error(f"Mismatch: required cols {initial_sequence_cols} vs history {last_known_sequence_scaled_df.columns}")
         return pd.DataFrame()
    current_sequence_scaled_array = last_known_sequence_scaled_df[initial_sequence_cols].values
    future_predictions_scaled = []
    model_uses_features = len(model.inputs) > 1

    for i in range(n_future_hours):
        current_input_price = current_sequence_scaled_array[:, 0].reshape(1, sequence_length, 1)
        model_input = [current_input_price]
        if model_uses_features:
            num_features_in_array = current_sequence_scaled_array.shape[1] - 1
            if num_features_in_array > 0:
                 current_input_features = current_sequence_scaled_array[:, 1:].reshape(1, sequence_length, num_features_in_array)
                 model_input.append(current_input_features)
            elif len(present_feature_cols) > 0: return pd.DataFrame()

        predicted_scaled_price = model.predict(model_input, verbose=0)[0, 0]
        future_predictions_scaled.append(predicted_scaled_price)

        next_dt = future_datetimes[i]
        if next_dt not in scaled_future_features_df.index: break

        if model_uses_features:
             next_scaled_features = scaled_future_features_df.loc[next_dt].values
             new_row_scaled = np.concatenate(([predicted_scaled_price], next_scaled_features))
        else: new_row_scaled = np.array([predicted_scaled_price])

        if new_row_scaled.shape[0] != current_sequence_scaled_array.shape[1]: break

        current_sequence_scaled_array = np.vstack([current_sequence_scaled_array[1:], new_row_scaled])

    # Inverse Transform Predictions
    if not future_predictions_scaled: return pd.DataFrame()
    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
    try:
        price_col_index = scaled_cols_list.index(price_col_name_scaled)
        dummy_future_predictions = np.zeros((len(future_predictions_scaled), len(scaled_cols_list)))
        dummy_future_predictions[:, price_col_index] = future_predictions_scaled.flatten()
        future_predictions_rescaled = scaler.inverse_transform(dummy_future_predictions)[:, price_col_index]
    except Exception as e: logging.error(f"Inverse scaling future predictions failed: {e}", exc_info=True); return pd.DataFrame()

    actual_future_datetimes = future_datetimes[:len(future_predictions_rescaled)]
    df_future_predictions = pd.DataFrame({'Datetime': actual_future_datetimes, 'PredictedPrice': future_predictions_rescaled})
    logging.info(f"Finished future prediction. Generated {len(df_future_predictions)} predictions.")
    return df_future_predictions


# ========== 6. MAIN EXECUTION ==========
# ========== 6. MAIN EXECUTION ==========
if __name__ == '__main__':
    # --- Configuration ---
    API_KEY_ENTSOE = 'a5298d45-1477-4ecf-8335-dd4d99fa969f'
    MODEL_CACHE_DIR = "saved_models"
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    LOAD_CACHED_MODEL = True # Set True to load if exists, False to force retrain

    # --- TARGET CONFIGURATION ---
    # Option 1: Bulgaria
    # TARGET_COUNTRY_CODE = '10Y10YBG-CEEG-0C'
    # TARGET_COUNTRY_TZ = 'Europe/Sofia'
    # TARGET_WEATHER_LAT = 42.6977
    # TARGET_WEATHER_LON = 23.3219

    # Option 2: Germany
    TARGET_COUNTRY_CODE = '10YCA-BULGARIA-R'    # Germany-Luxembourg Bidding Zone EIC
    TARGET_COUNTRY_TZ = 'Europe/Sofia'         # Germany Timezone
    TARGET_WEATHER_LAT = 42.6977  # Sofia Latitude (approx.)
    TARGET_WEATHER_LON = 23.3219  # Sofia Longitude (approx.)
    # --- END TARGET ---

    if API_KEY_ENTSOE == 'YOUR_ENTSOE_API_KEY' or not API_KEY_ENTSOE: exit()

    # --- Define Date Range & Fetch --- ### MODIFIED SECTION ###
    # Define overall range using UTC
    end_date_fetch_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0) # Start of today in UTC
    start_date_fetch_utc = end_date_fetch_utc - timedelta(days=735) # Fetch ~2 years back from today (UTC based)

    # Log the approximate local range for user understanding
    start_date_fetch_local_approx = start_date_fetch_utc.astimezone(pytz.timezone(TARGET_COUNTRY_TZ))
    end_date_fetch_local_approx = end_date_fetch_utc.astimezone(pytz.timezone(TARGET_COUNTRY_TZ))
    logging.info(f"--- Fetching historical data from approx {start_date_fetch_local_approx.date()} to {end_date_fetch_local_approx.date()} ({TARGET_COUNTRY_TZ}) ---")

    all_price_chunks = []
    all_weather_chunks = []
    all_generation_chunks = []
    all_load_chunks = []

    # Generate chunk start dates in UTC directly
    chunk_start_dates_utc = pd.date_range(start=start_date_fetch_utc, end=end_date_fetch_utc, freq='MS', tz='UTC')
    # Ensure the very first fetch date is included if it wasn't a month start
    if start_date_fetch_utc < chunk_start_dates_utc[0]:
         chunk_start_dates_utc = chunk_start_dates_utc.insert(0, start_date_fetch_utc)

    # --- Start Fetching Loop ---
    for i, chunk_start_utc in enumerate(chunk_start_dates_utc):
        # Determine UTC end date for the chunk
        chunk_end_utc = chunk_start_dates_utc[i+1] if i + 1 < len(chunk_start_dates_utc) else end_date_fetch_utc
        if chunk_end_utc <= chunk_start_utc: continue

        # Convert chunk boundaries to local naive time JUST for Open-Meteo history call and logging
        chunk_start_local_naive = chunk_start_utc.astimezone(pytz.timezone(TARGET_COUNTRY_TZ)).replace(tzinfo=None)
        chunk_end_local_naive = chunk_end_utc.astimezone(pytz.timezone(TARGET_COUNTRY_TZ)).replace(tzinfo=None)

        logging.info(f"\nFetching Chunk {i+1}/{len(chunk_start_dates_utc)}: Local {chunk_start_local_naive.date()} to {chunk_end_local_naive.date()} for {TARGET_COUNTRY_CODE}")

        # Fetch Prices (Pass UTC aware datetimes)
        df_price_chunk = fetch_entsoe_prices(API_KEY_ENTSOE, chunk_start_utc, chunk_end_utc, TARGET_COUNTRY_CODE, TARGET_COUNTRY_TZ)
        if not df_price_chunk.empty:
            # Print full chunk if needed (uncomment below)
            # print(f"\n--- Price Chunk {i+1} Data ---\n{df_price_chunk.to_string()}\n--- End Chunk ---")
            all_price_chunks.append(df_price_chunk)
        else: logging.warning(f"No price data returned for chunk.")
        time.sleep(0.1) # Be polite to API

        # Fetch Historical Weather (Pass local naive datetimes)
        df_weather_chunk = fetch_openmeteo_data(chunk_start_local_naive, chunk_end_local_naive, TARGET_WEATHER_LAT, TARGET_WEATHER_LON, TARGET_COUNTRY_TZ)
        if not df_weather_chunk.empty: all_weather_chunks.append(df_weather_chunk)
        else: logging.warning(f"No historical weather data returned for chunk.")
        time.sleep(0.1)

        # Fetch Generation (Placeholder - ENTSO-E needs UTC aware)
        # df_gen_chunk = fetch_entsoe_generation(API_KEY_ENTSOE, chunk_start_utc, chunk_end_utc, TARGET_COUNTRY_CODE, TARGET_COUNTRY_TZ)
        # if not df_gen_chunk.empty: all_generation_chunks.append(df_gen_chunk)
        # time.sleep(0.1)

        # Fetch Load (Placeholder - ENTSO-E needs UTC aware)
        # df_load_chunk = fetch_entsoe_load(API_KEY_ENTSOE, chunk_start_utc, chunk_end_utc, TARGET_COUNTRY_CODE, TARGET_COUNTRY_TZ)
        # if not df_load_chunk.empty: all_load_chunks.append(df_load_chunk)
        # time.sleep(0.1)

    # --- Combine and Merge Fetched Data ---
    # ... (rest of the script remains the same) ...

        df_gen_chunk = fetch_entsoe_generation(API_KEY_ENTSOE, chunk_start_utc, chunk_end_utc, TARGET_COUNTRY_CODE, TARGET_COUNTRY_TZ)
        if not df_gen_chunk.empty: all_generation_chunks.append(df_gen_chunk)
        time.sleep(0.1)

        df_load_chunk = fetch_entsoe_load(API_KEY_ENTSOE, chunk_start_utc, chunk_end_utc, TARGET_COUNTRY_CODE, TARGET_COUNTRY_TZ)
        if not df_load_chunk.empty: all_load_chunks.append(df_load_chunk)
        time.sleep(0.1)

    # --- Combine and Merge Fetched Data ---
    if not all_price_chunks: logging.critical("FATAL: No price data fetched."); exit()
    df_prices_full = pd.concat(all_price_chunks, ignore_index=True).drop_duplicates('Datetime').sort_values('Datetime')
    df_full = df_prices_full; logging.info(f"Combined {len(df_full)} price points.")
    data_types_present = ['Price']
    weather_data_fetched = False
    if all_weather_chunks:
        df_weather_full = pd.concat(all_weather_chunks, ignore_index=True).drop_duplicates('Datetime').sort_values('Datetime')
        if not df_weather_full.empty:
             df_full['Datetime'] = pd.to_datetime(df_full['Datetime'])
             df_weather_full['Datetime'] = pd.to_datetime(df_weather_full['Datetime'])
             df_full = pd.merge(df_full, df_weather_full, on='Datetime', how='left')
             data_types_present.append('Weather'); weather_data_fetched = True
             logging.info(f"Merged Weather. Total rows after merge: {len(df_full)}")
    if not weather_data_fetched: logging.warning("No weather data merged.")
    if all_generation_chunks: # Placeholder logic
        df_gen_full = pd.concat(all_generation_chunks,ignore_index=True).drop_duplicates('Datetime').sort_values('Datetime')
        if not df_gen_full.empty: df_full=pd.merge(df_full,df_gen_full,on='Datetime',how='left'); data_types_present.append('Gen')
    if all_load_chunks: # Placeholder logic
        df_load_full = pd.concat(all_load_chunks,ignore_index=True).drop_duplicates('Datetime').sort_values('Datetime')
        if not df_load_full.empty: df_full=pd.merge(df_full,df_load_full,on='Datetime',how='left'); data_types_present.append('Load')

    if df_full.empty or 'Price' not in df_full.columns or df_full['Price'].isnull().all(): exit()
    if 'Datetime' not in df_full.columns: exit()
    df_full['Datetime'] = pd.to_datetime(df_full['Datetime']); df_full.sort_values('Datetime', inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    logging.info(f"Final combined DataFrame columns: {df_full.columns.tolist()}")
    logging.info(f"Final Data range: {df_full['Datetime'].min()} to {df_full['Datetime'].max()}")
    logging.info(f"Total rows before split: {len(df_full)}")

    # --- Chronological Train/Validation/Test Split --- ## MODIFIED ##
    logging.info("\n--- Splitting data by date for realistic evaluation ---")
    last_data_point_dt = df_full['Datetime'].iloc[-1]
    test_start_date = (last_data_point_dt - pd.offsets.MonthBegin(n=1, normalize=True))
    logging.info(f"Test set intended to start on: {test_start_date.date()}")
    val_start_date = (test_start_date - pd.offsets.MonthBegin(n=1, normalize=True))
    logging.info(f"Validation set intended to start on: {val_start_date.date()}")
    first_data_point_dt = df_full['Datetime'].iloc[0]
    if val_start_date <= first_data_point_dt + timedelta(days=30): logging.critical(f"Not enough training data before {val_start_date.date()}."); exit()
    if test_start_date <= val_start_date: logging.critical(f"Validation start date not before test start date."); exit()
    logging.info(f"Slicing data: Train < {val_start_date.date()}, Validation < {test_start_date.date()}, Test >= {test_start_date.date()}")
    df_train = df_full[df_full['Datetime'] < val_start_date].copy()
    df_val = df_full[(df_full['Datetime'] >= val_start_date) & (df_full['Datetime'] < test_start_date)].copy()
    df_test = df_full[df_full['Datetime'] >= test_start_date].copy()
    if df_train.empty or df_val.empty or df_test.empty: logging.critical(f"Splitting resulted in empty sets."); exit()
    logging.info(f"\nData Split Results (Date-Based):")
    logging.info(f"Train: {len(df_train)} points ({df_train['Datetime'].min().date()} to {df_train['Datetime'].max().date()})")
    logging.info(f"Val:   {len(df_val)} points ({df_val['Datetime'].min().date()} to {df_val['Datetime'].max().date()})")
    logging.info(f"Test:  {len(df_test)} points ({df_test['Datetime'].min().date()} to {df_test['Datetime'].max().date()})")
    # --- End Date-Based Split ---

    # --- Preprocess Data ---
    logging.info("\n--- Preprocessing Train Data (Fitting Scaler) ---")
    df_train_processed, scaler, scaled_cols_list, scaled_price_col_name, feature_cols_list = preprocess_data(df_train, fit_scaler=True)
    if scaler is None or scaled_price_col_name is None: logging.critical("Preprocessing failed."); exit()
    logging.info("\n--- Preprocessing Validation Data (Transforming Only) ---")
    df_val_processed, _, _, _, _ = preprocess_data(df_val, scaler=scaler, fit_scaler=False)
    logging.info("\n--- Preprocessing Test Data (Transforming Only) ---")
    df_test_processed, _, _, _, _ = preprocess_data(df_test, scaler=scaler, fit_scaler=False)

    # --- Create Sequences ---
    sequence_length = 24 * 3 # Shortened sequence length
    logging.info(f"\n--- Creating Sequences (Length={sequence_length}) ---")
    X_train_price, X_train_features, y_train = create_sequences(df_train_processed, scaled_price_col_name, feature_cols_list, sequence_length)
    X_val_price, X_val_features, y_val = create_sequences(df_val_processed, scaled_price_col_name, feature_cols_list, sequence_length)
    X_test_price, X_test_features, y_test_scaled = create_sequences(df_test_processed, scaled_price_col_name, feature_cols_list, sequence_length)
    if y_train.size == 0 or y_val.size == 0 or y_test_scaled.size == 0: logging.critical("Sequence creation failed."); exit()
    test_data_index = df_test_processed.index[sequence_length:] if len(df_test_processed) > sequence_length else pd.Index([])

    # --- Build or Load Model --- ## NEW CACHING LOGIC ##
    model = None
    model_loaded_from_cache = False
    model_filename = f"lstm_model_{TARGET_COUNTRY_CODE}_seq{sequence_length}.keras"
    model_path = os.path.join(MODEL_CACHE_DIR, model_filename)

    logging.info("\n--- Checking for Cached Model ---")
    if LOAD_CACHED_MODEL and os.path.exists(model_path):
        logging.info(f"Found cached model: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path)
            logging.info("Successfully loaded model from cache.")
            model_loaded_from_cache = True
        except Exception as e:
            logging.warning(f"Failed to load cached model: {e}. Will retrain.")
            model = None
    else:
        logging.info("No cached model found or LOAD_CACHED_MODEL=False. Building and training a new model.")

    if model is None:
        logging.info("\n--- Building New Model ---")
        price_input_shape = (sequence_length, X_train_price.shape[2])
        feature_input_shape = (sequence_length, X_train_features.shape[2]) if X_train_features is not None and X_train_features.ndim == 3 and X_train_features.shape[2] > 0 else None
        model = build_model(price_input_shape, feature_input_shape)

    # --- Train Model (if not loaded from cache) --- ## MODIFIED ##
    if not model_loaded_from_cache:
        if model is None: logging.critical("Model is None, cannot train."); exit()
        model = train_model(
            model,
            X_train_price, X_train_features, y_train,
            X_val_price, X_val_features, y_val,
            epochs=100, batch_size=64 # Example training params
        )
        try: # Save the newly trained model
            model.save(model_path)
            logging.info(f"Trained model saved to: {model_path}")
        except Exception as e: logging.error(f"Failed to save trained model: {e}")
    else:
         logging.info("Skipping training as model was loaded from cache.")

    # --- Evaluate Model --- ## MODIFIED ##
    logging.info("\n--- Evaluating Final Model ---")
    if model is None: logging.critical("Model is None, cannot evaluate."); exit()
    rmse_test, mae_test, r2_test, acc_within_tol_test = evaluate_model(
        model, scaler,
        X_test_price, X_test_features, y_test_scaled,
        test_data_index, scaled_cols_list
    )

    # =============================================================================
    # === FUTURE PREDICTION SECTION ===
    # =============================================================================
    if model is not None: # Proceed only if model exists (loaded or trained)
        logging.info("\n--- Starting Future Prediction Workflow ---")
        N_FUTURE_HOURS = 24

        min_len_for_future = sequence_length
        if len(df_test_processed) >= min_len_for_future:
            last_historical_scaled_df = df_test_processed.iloc[-sequence_length:]

            current_time_local = datetime.now(pytz.timezone(TARGET_COUNTRY_TZ))
            forecast_start_dt_local = (current_time_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            forecast_end_dt_local = forecast_start_dt_local
            logging.info(f"Requesting weather forecast for date: {forecast_start_dt_local.date()} ({TARGET_COUNTRY_TZ})")

            df_weather_forecast = fetch_openmeteo_forecast(
                latitude=TARGET_WEATHER_LAT,
                longitude=TARGET_WEATHER_LON,
                timezone=TARGET_COUNTRY_TZ,
                start_date_forecast=forecast_start_dt_local.replace(tzinfo=None),
                end_date_forecast=forecast_end_dt_local.replace(tzinfo=None)
            )

            if not df_weather_forecast.empty:
                forecast_start_naive = forecast_start_dt_local.replace(tzinfo=None)
                forecast_end_naive_exclusive = forecast_start_naive + timedelta(hours=N_FUTURE_HOURS)
                df_weather_forecast['Datetime'] = pd.to_datetime(df_weather_forecast['Datetime'])
                df_weather_forecast_filtered = df_weather_forecast[
                    (df_weather_forecast['Datetime'] >= forecast_start_naive) &
                    (df_weather_forecast['Datetime'] < forecast_end_naive_exclusive)
                ].copy()

                if len(df_weather_forecast_filtered) == N_FUTURE_HOURS:
                     logging.info(f"Successfully filtered weather forecast to {len(df_weather_forecast_filtered)} hours for prediction.")
                     present_feature_cols = feature_cols_list # Use features from preprocessing

                     df_future = predict_future_prices(
                         model=model,
                         scaler=scaler,
                         last_known_sequence_scaled_df=last_historical_scaled_df,
                         weather_forecast_df=df_weather_forecast_filtered,
                         n_future_hours=N_FUTURE_HOURS,
                         sequence_length=sequence_length,
                         price_col_name_scaled=scaled_price_col_name,
                         all_feature_cols=present_feature_cols,
                         scaled_cols_list=scaled_cols_list
                     )

                     if not df_future.empty:
                         logging.info(f"\n--- Future Predictions ({forecast_start_dt_local.date()}) ---")
                         print(df_future.to_string())
                         plot_future_predictions(df_future, title=f'Price Prediction for {forecast_start_dt_local.date()}')
                     else: logging.error("Future price prediction function returned empty DataFrame.")
                else: logging.error(f"Filtered weather forecast has {len(df_weather_forecast_filtered)} hours, expected {N_FUTURE_HOURS}.")
            else: logging.error("Could not fetch weather forecast for the next day.")
        else: logging.error(f"Not enough processed test data ({len(df_test_processed)}) for sequence length ({sequence_length}).")
    else: logging.error("No model available for future prediction.")

    # --- Final Log Output ---
    logging.info("\n--- Final Test Set Evaluation Results ---")
    if mae_test is not None: # Check if MAE was calculated (i.e., evaluation ran)
        logging.info(f"Final Test RMSE: {rmse_test:.3f} EUR/MWh")
        logging.info(f"Final Test MAE (Avg Price Delta): {mae_test:.3f} EUR/MWh")
        logging.info(f"Final Test R-squared (R²): {r2_test:.3f}")
        tolerance = 15.0
        logging.info(f"Final Test Accuracy (within +/- {tolerance:.1f} EUR): {acc_within_tol_test:.2f}%")
    elif model is not None and model_loaded_from_cache:
        logging.info("Model loaded from cache. Evaluation metrics not calculated in this run.")
    else:
        logging.error("Evaluation metrics could not be calculated (model train/load failed?).")

    logging.info("\n--- Script Finished ---")