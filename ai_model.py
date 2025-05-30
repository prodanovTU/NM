"""
Electricity Price Forecasting Software

Overview:
This software is designed to fetch, preprocess, and forecast electricity prices using historical data from the ENTSO-E
API and weather data from Open-Meteo. It employs a Long Short-Term Memory (LSTM) neural network to predict future
electricity prices based on historical price trends and correlated features such as weather conditions, time-based
features, and potentially generation/load data. The software supports data retrieval, preprocessing, model training,
evaluation, and future price prediction for a specified region.

Key Components:
1. **Data Fetching**:
   - `fetch_entsoe_prices`: Retrieves hourly electricity price data from the ENTSO-E Transparency Platform.
   - `fetch_openmeteo_data` and `fetch_openmeteo_forecast`: Fetch historical and forecasted weather data from Open-Meteo.
   - Placeholder functions (`fetch_entsoe_generation`, `fetch_entsoe_load`) for potential future integration of generation
     and load data.

2. **Data Preprocessing**:
   - `preprocess_data`: Normalizes data using MinMaxScaler, adds time-based features (hour, day of week, etc.), and handles
     missing values.
   - `create_sequences`: Prepares data for LSTM by creating time-series sequences of specified length.

3. **Model Building and Training**:
   - `build_model`: Constructs a dual-branch LSTM model that processes price sequences and additional features (e.g., weather).
   - `train_model`: Trains the model with early stopping to prevent overfitting.
   - `evaluate_model`: Assesses model performance using RMSE, MAE, R², and accuracy within a tolerance threshold.

4. **Prediction and Visualization**:
   - `predict_future_prices`: Generates future price predictions using the trained model and forecasted weather data.
   - `plot_predictions` and `plot_future_predictions`: Visualize historical and forecasted price predictions.

Usage:
The script is executed as a standalone program (`if __name__ == '__main__':`) and performs the following workflow:
- Fetches historical price and weather data for a specified region (e.g., Bulgaria).
- Splits data into training, validation, and test sets.
- Preprocesses and sequences data for LSTM input.
- Trains or loads a cached LSTM model.
- Evaluates the model on the test set and generates future price predictions for the next 24 hours.
- Saves plots of historical and future predictions.

Configuration:
- API key for ENTSO-E is required (`API_KEY_ENTSOE`).
- Region-specific parameters (country code, timezone, latitude, longitude) are set for data fetching.
- Model parameters (sequence length, epochs, batch size) can be adjusted for experimentation.

Output:
- Logs detailed execution steps and metrics.
- Saves trained model and prediction plots.
- Prints future price predictions for the specified forecast period.

Dependencies:
- Python libraries: requests, pandas, numpy, tensorflow, sklearn, matplotlib, xml.etree.ElementTree, pytz.
- External APIs: ENTSO-E Transparency Platform, Open-Meteo.

Limitations:
- Requires a valid ENTSO-E API key.
- Generation and load data fetching are placeholders and not implemented.
- Weather forecast data is limited to Open-Meteo’s capabilities.
- Model performance depends on data quality and feature availability.

Future Enhancements:
- Implement generation and load data fetching.
- Add support for multiple regions or cross-regional predictions.
- Incorporate additional features (e.g., economic indicators, demand forecasts).
- Optimize model architecture for better accuracy or faster training.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pytz
import time
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib

try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Agg backend not available, trying default.")
    matplotlib.use(None)
    import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def fetch_entsoe_prices(api_key: str,
                        start_date: datetime,
                        end_date: datetime,
                        country_code: str,
                        target_timezone: str
                        ) -> pd.DataFrame:
    """
    Fetches hourly electricity price data from the ENTSO-E Transparency Platform for a specified region and time period.
    Converts timestamps to the target timezone and ensures a complete hourly time series through interpolation.
    """
    if not api_key or api_key == 'YOUR_ENTSOE_API_KEY':
        logging.error("ENTSO-E API Key is missing or is the placeholder value.")
        raise ValueError("ENTSO-E API Key is missing or is the placeholder value.")
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
         logging.error("start_date and end_date must be datetime objects.")
         raise TypeError("start_date and end_date must be datetime objects.")
    if end_date <= start_date:
         logging.warning(f"end_date ({end_date}) is not after start_date ({start_date}), request might yield no data.")

    start_str = start_date.strftime('%Y%m%d%H%M')
    end_str = end_date.strftime('%Y%m%d%H%M')
    url = (
        f"https://web-api.tp.entsoe.eu/api?securityToken={api_key}"
        f"&documentType=A44&in_Domain={country_code}&out_Domain={country_code}"
        f"&periodStart={start_str}&periodEnd={end_str}"
    )
    logging.info(f"Attempting ENTSO-E fetch: Area='{country_code}', Period='{start_date.date()}':'{end_date.date()}'")
    logging.debug(f"Request URL: {url.replace(api_key, '***')}")

    ns = {'ts': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3'}

    try:
        response = requests.get(url, timeout=60)
        logging.debug(f"API Response Status Code: {response.status_code}")
        response.raise_for_status()

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
            if resolution_str != 'PT60M':
                logging.debug(f"Skipping TimeSeries: Found resolution {resolution_str}, expecting PT60M.")
                continue

            found_hourly_timeseries = True
            logging.debug(f"Processing TimeSeries with resolution {resolution_str}")

            time_interval_node = period.find('.//ts:timeInterval', ns)
            if time_interval_node is None:
                 logging.warning("Found Period block without timeInterval. Skipping Period.")
                 continue
            start_dt_str = time_interval_node.find('.//ts:start', ns).text

            try:
                period_start_dt_utc = pd.to_datetime(start_dt_str)
                if period_start_dt_utc.tzinfo is None:
                    logging.warning(f"Parsed start time '{start_dt_str}' as naive, localizing to UTC.")
                    period_start_dt_utc = period_start_dt_utc.tz_localize('UTC')
                elif str(period_start_dt_utc.tzinfo) != 'UTC':
                    logging.warning(f"Parsed start time '{start_dt_str}' with non-UTC timezone ({period_start_dt_utc.tzinfo}), converting to UTC.")
                    period_start_dt_utc = period_start_dt_utc.tz_convert('UTC')
            except Exception as e:
                 logging.warning(f"Could not parse period start time '{start_dt_str}'. Skipping Period. Error: {e}")
                 continue

            interval_timedelta = timedelta(hours=1)

            points_in_period = 0
            for point in period.findall('.//ts:Point', ns):
                try:
                    position = int(point.find('ts:position', ns).text)
                    price = float(point.find('ts:price.amount', ns).text)
                    point_start_time_utc = period_start_dt_utc + (position - 1) * interval_timedelta
                    points_data.append({'Datetime_UTC': point_start_time_utc, 'Price': price})
                    points_in_period += 1
                except (AttributeError, ValueError, TypeError, ET.ParseError) as e:
                    logging.warning(f"Could not parse Point data (Pos/Price). Skipping point. Error: {e}")
                    continue
            logging.debug(f"Extracted {points_in_period} points from this Period.")

        if not found_hourly_timeseries:
             reason_code = root.find('.//{*}Reason/{*}code')
             reason_text = root.find('.//{*}Reason/{*}text')
             if reason_code is not None and reason_text is not None:
                  logging.warning(f"ENTSO-E API Reason Code='{reason_code.text}', Text='{reason_text.text}' (Likely no data matching request).")
             else:
                  if processed_timeseries > 0:
                       logging.warning(f"Found {processed_timeseries} TimeSeries block(s), but none had PT60M (hourly) resolution.")
                  else:
                       logging.warning("No TimeSeries data blocks found at all in the ENTSO-E response XML.")
             return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})

        if not points_data:
            logging.warning("Hourly TimeSeries/Period blocks found, but failed to extract any valid Price Points.")
            return pd.DataFrame({'Datetime': pd.to_datetime([]), 'Price': []})

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

        df_prices = df_prices.reset_index()
        df_prices.rename(columns={'index': 'Datetime'}, inplace=True)
        if df_prices.empty:
             logging.warning("No price points remaining after processing.")
        else:
             logging.info(f"Finished processing. Returning DataFrame with {len(df_prices)} rows.")
        return df_prices

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

def fetch_entsoe_generation(api_key: str, start_date: datetime, end_date: datetime, country_code: str, target_timezone: str) -> pd.DataFrame:
    """
    Placeholder function for fetching electricity generation data from ENTSO-E.
    Currently returns an empty DataFrame as implementation is not provided.
    """
    logging.debug("Skipping Generation fetch (Placeholder).")
    return pd.DataFrame({'Datetime': pd.to_datetime([])})

def fetch_entsoe_load(api_key: str, start_date: datetime, end_date: datetime, country_code: str, target_timezone: str) -> pd.DataFrame:
    """
    Placeholder function for fetching electricity load data from ENTSO-E.
    Currently returns an empty DataFrame as implementation is not provided.
    """
    logging.debug("Skipping Load fetch (Placeholder).")
    return pd.DataFrame({'Datetime': pd.to_datetime([])})

def fetch_openmeteo_data(start_date, end_date, latitude, longitude, timezone):
    """
    Fetches historical weather data (temperature, humidity, wind speed, cloudiness) from Open-Meteo for the specified
    coordinates and time period. Ensures data aligns with the requested timezone.
    """
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

def fetch_openmeteo_forecast(latitude, longitude, timezone, start_date_forecast, end_date_forecast):
    """
    Fetches weather forecast data from Open-Meteo for the specified coordinates and time period.
    Used to provide input features for future price predictions.
    """
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

def preprocess_data(df, scaler=None, fit_scaler=False):
    """
    Preprocesses input data by adding time-based features, scaling numerical columns, and handling missing values.
    Returns the processed DataFrame, scaler, and metadata about scaled columns and features.
    """
    df_processed = df.copy()
    if 'Datetime' not in df_processed.columns: raise ValueError("'Datetime' column missing.")
    df_processed['Datetime'] = pd.to_datetime(df_processed['Datetime'])
    df_processed.set_index('Datetime', inplace=True)
    df_processed.sort_index(inplace=True)
    logging.info(f"Preprocessing data from {df_processed.index.min()} to {df_processed.index.max()}")

    df_processed['Hour'] = df_processed.index.hour
    df_processed['DayOfWeek'] = df_processed.index.dayofweek
    df_processed['Month'] = df_processed.index.month
    df_processed['IsWeekend'] = df_processed['DayOfWeek'].isin([5, 6]).astype(int)

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

    nan_counts = df_processed[columns_to_scale].isnull().sum()
    if nan_counts.sum() > 0:
         logging.warning(f"NaNs detected before scaling:\n{nan_counts[nan_counts > 0]}")
         logging.info("Filling NaNs using ffill/bfill/0.")
         df_processed.fillna(method='ffill', inplace=True); df_processed.fillna(method='bfill', inplace=True); df_processed.fillna(0, inplace=True)

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

def create_sequences(data, price_col_name, feature_col_names, seq_length):
    """
    Creates time-series sequences for LSTM input, including price and feature data.
    Each sequence represents a window of historical data used to predict the next price.
    """
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

def build_model(input_shape_price, input_shape_features):
    """
    Builds a dual-branch LSTM model:
    - Price branch: Processes historical price sequences.
    - Features branch (optional): Processes additional features like weather or time-based data.
    The branches are merged to produce a single price prediction.
    """
    input_price = Input(shape=input_shape_price, name='price_input')
    lstm_price_1 = LSTM(units=64, return_sequences=True, name='lstm_price_1')(input_price)
    dropout_price_1 = Dropout(0.2, name='dropout_price_1')(lstm_price_1)
    lstm_price_2 = LSTM(units=32, name='lstm_price_2')(dropout_price_1)
    dropout  _price_2 = Dropout(0.2, name='dropout_price_2')(lstm_price_2)
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

def plot_predictions(true_index, true_values, predicted_values, title='Electricity Price Prediction'):
    """
    Plots actual vs. predicted electricity prices for historical test data.
    Saves the plot as a PNG file for analysis.
    """
    if len(true_index) != len(true_values) or len(true_index) != len(predicted_values): return
    plt.figure(figsize=(15, 7)); plt.plot(true_index, true_values, 'b-', label='Actual Price', alpha=0.8, linewidth=1.5)
    plt.plot(true_index, predicted_values, 'r--', label='Predicted Price', alpha=0.8, linewidth=1.2)
    plt.title(f"{title} (Test Set Evaluation)"); plt.xlabel('Time'); plt.ylabel('Price (EUR/MWh)')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
    plot_filename = "historical_evaluation_plot.png"
    try: plt.savefig(plot_filename); logging.info(f"Historical plot saved: {plot_filename}")
    except Exception as e: logging.error(f"Could not save historical plot: {e}")
    finally: plt.close()

def plot_future_predictions(df_future, title='Future Electricity Price Prediction'):
    """
    Plots predicted electricity prices for the forecast period.
    Saves the plot as a PNG file for visualization.
    """
    if df_future.empty or 'PredictedPrice' not in df_future.columns or 'Datetime' not in df_future.columns: return
    plt.figure(figsize=(15, 7))
    plt.plot(df_future['Datetime'], df_future['PredictedPrice'], 'orange', marker='o', markersize=4, linestyle='-', label='Future Predicted Price')
    plt.title(title); plt.xlabel('Time'); plt.ylabel('Predicted Price (EUR/MWh)')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
    plot_filename = "future_prediction_plot.png"
    try: plt.savefig(plot_filename); logging.info(f"Future prediction plot saved: {plot_filename}")
    except Exception as e: logging.error(f"Could not save future plot: {e}")
    finally: plt.close()

def train_model(model, X_train_price, X_train_features, y_train, X_val_price, X_val_features, y_val, epochs=100, batch_size=32):
    """
    Trains the LSTM model using training and validation data.
    Implements early stopping to prevent overfitting and logs training progress.
    """
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

def evaluate_model(model, scaler,
                   X_test_price, X_test_features, y_test_scaled,
                   test_data_index, scaled_columns_list,
                   accuracy_tolerance_eur=15.0):
    """
    Evaluates the trained model on the test set, computing metrics like RMSE, MAE, R², and accuracy within a specified
    price tolerance. Generates a plot comparing actual and predicted prices.
    """
    test_inputs = [X_test_price]
    features_exist = X_test_features is not None and X_test_features.ndim == 3 and X_test_features.shape[2] > 0
    if features_exist:
        test_inputs.append(X_test_features)
        logging.info("Evaluating model with Price + Features on test set.")
    else: logging.info("Evaluating model with Price ONLY on test set.")

    logging.info("\n--- Evaluating on Test Set ---")
    try: predictions_scaled = model.predict(test_inputs)
    except Exception as e: logging.error(f"Model prediction failed during evaluation: {e}", exc_info=True); return None, None, None, None

    if scaler is None or not scaled_columns_list: return None, None, None, None
    price_col_name_scaled = 'ScaledPrice'
    try: price_col_index = scaled_columns_list.index(price_col_name_scaled)
    except ValueError: logging.error(f"'{price_col_name_scaled}' not in scaled_columns_list."); return None, None, None, None
    num_scaled_features = len(scaled_columns_list)
    if num_scaled_features != scaler.n_features_in_: logging.error(f"Scaler mismatch"); return None, None, None, None

    try:
        dummy_predictions = np.zeros((len(predictions_scaled), num_scaled_features)); dummy_predictions[:, price_col_index] = predictions_scaled.flatten()
        predictions_rescaled = scaler.inverse_transform(dummy_predictions)[:, price_col_index]
        dummy_y_test = np.zeros((len(y_test_scaled), num_scaled_features)); dummy_y_test[:, price_col_index] = y_test_scaled.flatten()
        y_test_rescaled = scaler.inverse_transform(dummy_y_test)[:, price_col_index]
    except Exception as e: logging.error(f"Inverse scaling failed during evaluation: {e}", exc_info=True); return None, None, None, None

    logging.info("\n--- Test Set Evaluation Metrics ---")
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled); logging.info(f"Average Price Delta (MAE):  {mae:.3f} (EUR/MWh)")
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled)); logging.info(f"RMSE:                       {rmse:.3f} (EUR/MWh)")
    r2 = r2_score(y_test_rescaled, predictions_rescaled); logging.info(f"R-squared (R²):             {r2:.3f}")
    absolute_errors = np.abs(y_test_rescaled - predictions_rescaled)
    within_tolerance_count = np.sum(absolute_errors <= accuracy_tolerance_eur)
    accuracy_within_tolerance = (within_tolerance_count / len(y_test_rescaled)) * 100 if len(y_test_rescaled) > 0 else 0
    logging.info(f"Accuracy (within +/- {accuracy_tolerance_eur:.1f} EUR): {accuracy_within_tolerance:.2f}%")

    if len(test_data_index) == len(y_test_rescaled): plot_predictions(test_data_index, y_test_rescaled, predictions_rescaled, title='Historical Price Prediction')
    else: logging.warning(f"Length mismatch plotting test data.")

    return rmse, mae, r2, accuracy_within_tolerance

def predict_future_prices(model,
                          scaler,
                          last_known_sequence_scaled_df,
                          weather_forecast_df,
                          n_future_hours,
                          sequence_length,
                          price_col_name_scaled,
                          all_feature_cols,
                          scaled_cols_list):
    """
    Predicts future electricity prices for the specified number of hours using the trained model and forecasted weather
    data. Iteratively updates the input sequence with new predictions and feature data.
    """
    logging.info(f"Starting future prediction for {n_future_hours} hours...")
    if not isinstance(last_known_sequence_scaled_df, pd.DataFrame) or len(last_known_sequence_scaled_df) != sequence_length: return pd.DataFrame()
    if price_col_name_scaled not in last_known_sequence_scaled_df.columns: return pd.DataFrame()
    if scaler is None or not scaled_cols_list: return pd.DataFrame()

    weather_forecast_df = weather_forecast_df.copy()
    weather_forecast_df['Datetime'] = pd.to_datetime(weather_forecast_df['Datetime'])
    weather_forecast_df.set_index('Datetime', inplace=True)

    last_historical_dt = last_known_sequence_scaled_df.index[-1]
    future_datetimes = pd.date_range(start=last_historical_dt + timedelta(hours=1), periods=n_future_hours, freq='h')
    future_df = pd.DataFrame(index=future_datetimes)
    future_df['Hour'] = future_df.index.hour; future_df['DayOfWeek'] = future_df.index.dayofweek
    future_df['Month'] = future_df.index.month w; future_df['IsWeekend'] = future_df['DayOfWeek'].isin([5, 6]).astype(int)

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

if __name__ == '__main__':
    """
    Main execution workflow:
    1. Configures parameters (API key, country code, timezone, coordinates).
    2. Fetches historical data in chunks to manage API limitations.
    3. Merges and preprocesses data, splitting into train/validation/test sets.
    4. Trains or loads an LSTM model, evaluates it, and predicts future prices.
    5. Logs results and saves visualizations.
    """
    API_KEY_ENTSOE = 'a5298d45-1477-4ecf-8335-dd4d99fa969f'
    MODEL_CACHE_DIR = "saved_models"
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    LOAD_CACHED_MODEL = True

    TARGET_COUNTRY_CODE = '10YCA-BULGARIA-R'
    TARGET_COUNTRY_TZ = 'Europe/Sofia'
    TARGET_WEATHER_LAT = 42.6977
    TARGET_WEATHER_LON = 23.3219

    if API_KEY_ENTSOE == 'YOUR_ENTSOE_API_KEY' or not API_KEY_ENTSOE: exit    exit()

    end_date_fetch_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date_fetch_utc = end_date_fetch_utc - timedelta(days=735)

    start_date_fetch_local_approx = start_date_fetch_utc.astimezone(pytz.timezone(TARGET_COUNTRY_TZ))
    end_date_fetch_local_approx = end_date_fetch_utc.astimezone(pytz.timezone(TARGET_COUNTRY_TZ))
    logging.info(f"--- Fetching historical data from approx {start_date_fetch_local_approx.date()} to {end_date_fetch_local_approx.date()} ({TARGET_COUNTRY_TZ}) ---")

    all_price_chunks = []
    all_weather_chunks = []
    all_generation_chunks = []
    all_load_chunks = []

    chunk_start_dates_utc = pd.date_range(start=start_date_fetch_utc, end=end_date_fetch_utc, freq='MS', tz='UTC')
    if start_date_fetch_utc < chunk_start_dates_utc[0]:
         chunk_start_dates_utc = chunk_start_dates_utc.insert(0, start_date_fetch_utc)

    for i, chunk_start_utc in enumerate(chunk_start_dates_utc):
        chunk

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
             df_full = pd.merge(df_full,ედ

    if df_full.empty or 'Price' not in df_full.columns or df_full['Price'].isnull().all(): exit()
    if 'Datetime' not in df_full.columns: exit()
    df_full['Datetime'] = pd.to_datetime(df_full['Datetime']); df_full.sort_values('Datetime', inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    logging.info(f"Final combined DataFrame columns: {df_full.columns.tolist()}")
    logging.info(f"Final Data range: {df_full['Datetime'].min()} to {df_full['Datetime'].max()}")
    logging.info(f"Total rows before split: {len(df_full)}")

    total_rows = len(df_full)
    if total_rows < 24 * 90:
        logging.critical(f"Not enough total data points ({total_rows}) for split.")
        exit()

    test_ratio = 0.10
    val_ratio = 0.10

    test_split_idx = int(total_rows * (1 - test_ratio))
    val_split_idx = int(total_rows * (1 - test_ratio - val_ratio))

    df_train = df_full.iloc[:val_split_idx].copy()
    df_val = df_full.iloc[val_split_idx:test_split_idx].copy()
    df_test = df_full.iloc[test_split_idx:].copy()

    if df_train.empty or df_val.empty or df_test.empty:
        logging.critical(
            f"Splitting resulted in empty sets using ratios (Total: {total_rows}, Train End: {val_split_idx}, Test Start: {test_split_idx}). Check ratios and data length.")
        exit()

    logging.info(f"\nData Split Results (Ratio-Based):")
    logging.info(f"Train: {len(df_train)} points ({df_train['Datetime'].min().date()} to {df_train['Datetime'].max().date()})")
    logging.info(f"Val:   {len(df_val)} points ({df_val['Datetime'].min().date()} to {df_val['Datetime'].max().date()})")
    logging.info(f"Test:  {len(df_test)} points ({df_test['Datetime'].min().date()} to {df_test['Datetime'].max().date()})")

    logging.info("\n--- Preprocessing Train Data (Fitting Scaler) ---")
    df_train_processed, scaler, scaled_cols_list, scaled_price_col_name, feature_cols_list = preprocess_data(df_train, fit_scaler=True)
    if scaler is None or scaled_price_col_name is None: logging.critical("Preprocessing failed."); exit()
    logging.info("\n--- Preprocessing Validation Data (Transforming Only) ---")
    df_val_processed, _, _, _, _ = preprocess_data(df_val, scaler=scaler, fit_scaler=False)
    logging.info("\n--- Preprocessing Test Data (Transforming Only) ---")
    df_test_processed, _, _, _, _ = preprocess_data(df_test, scaler=scaler, fit_scaler=False)

    sequence_length = 24 * 3
    logging.info(f"\n--- Creating Sequences (Length={sequence_length}) ---")
    X_train_price, X_train_features, y_train = create_sequences(df_train_processed, scaled_price_col_name, feature_cols_list, sequence_length)
    X_val_price, X_val_features, y_val = create_sequences(df_val_processed, scaled_price_col_name, feature_cols_list, sequence_length)
    X_test_price, X_test_features, y_test_scaled = create_sequences(df_test_processed, scaled_price_col_name, feature_cols_list, sequence_length)
    if y_train.size == 0 or y_val.size == 0 or y_test_scaled.size == 0: logging.critical("Sequence creation failed."); exit()
    test_data_index = df_test_processed.index[sequence_length:] if len(df_test_processed) > sequence_length else pd.Index([])

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

    if not model_loaded_from_cache:
        if model is None: logging.critical("Model is None, cannot train."); exit()
        model = train_model(
            model,
            X_train_price, X_train_features, y_train,
            X_val_price, X_val_features, y_val,
            epochs=100, batch_size=64
        )
        try:
            model.save(model_path)
            logging.info(f"Trained model saved to: {model_path}")
        except Exception as e: logging.error(f"Failed to save trained model: {e}")
    else:
         logging.info("Skipping training as model was loaded from cache.")

    logging.info("\n--- Evaluating Final Model ---")
    if model is None: logging.critical("Model is None, cannot evaluate."); exit()
    rmse_test, mae_test, r2_test, acc_within_tol_test = evaluate_model(
        model, scaler,
        X_test_price, X_test_features, y_test_scaled,
        test_data_index, scaled_cols_list
    )

    if model is not None:
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
                     present_feature_cols = feature_cols_list

                     df_future = predict_future_prices(
                         model=model,
                         scaler=scaler,
                         last_known_sequence_scaled_df=last_historical_scaled_df,
                         weather_forecast_df=df_weather_forecast_filtered,
                         n_future_hours=N_FUTURE_HOURS,
                         sequence_length = sequence_length,
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

    logging.info("\n--- Final Test Set Evaluation Results ---")
    if mae_test is not None:
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