import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib
matplotlib.use('TkAgg')  # Or try 'Qt5Agg', 'WXAgg'
import matplotlib.pyplot as plt

# ========== 1. FETCH ELECTRICITY PRICE DATA FROM ENTSO-E ==========
def fetch_entsoe_prices(api_key, start_date, end_date, country_code='10Y10YBG-CEEG-0C'):
    start = start_date.strftime('%Y%m%d%H%M')
    end = end_date.strftime('%Y%m%d%H%M')
    url = (
        f"https://web-api.tp.entsoe.eu/api?securityToken={api_key}"
        f"&documentType=A44&in_Domain={country_code}&out_Domain={country_code}"
        f"&periodStart={start}&periodEnd={end}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"ENTSO-E API error: {response.status_code}")
    # NOTE: Real parsing logic for XML would go here
    # Simulating dummy data for now
    hours = pd.date_range(start=start_date, end=end_date, freq='H')
    dummy_prices = np.random.uniform(50, 150, size=len(hours))
    return pd.DataFrame({'Datetime': hours, 'Price': dummy_prices})

# ========== 2. PREPROCESSING ==========
def preprocess_data(df):
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    scaler = MinMaxScaler()
    df['ScaledPrice'] = scaler.fit_transform(df[['Price']])
    return df, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def prepare_future_sequence(data, seq_length):
    # Use the last 'seq_length' data points to predict the future
    return data[-seq_length:].reshape(1, seq_length, 1)

# ========== 3. BUILD & TRAIN MODEL ==========
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ========== 4. PLOT RESULTS ==========
def plot_predictions(true, predicted, future_predictions=None):
    plt.figure(figsize=(14, 5))
    plt.plot(true, label='Actual Prices')
    plt.plot(predicted, label='Predicted Prices (Test Data)')
    if future_predictions is not None:
        # Generate time indices for the future predictions
        last_time = true.index[-1]
        future_time = pd.date_range(start=last_time + timedelta(hours=1), periods=len(future_predictions), freq='H')
        plt.plot(future_time, future_predictions, label='Predicted Prices (Next Hours)')
    plt.title('Electricity Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# ========== 5. MAIN LOGIC ==========
if __name__ == '__main__':
    API_KEY = 'a5298d45-1477-4ecf-8335-dd4d99fa969f'  # Replace with actual key
    now = datetime.now()
    # Set end_date to the beginning of the current hour
    end_date = now.replace(minute=0, second=0, microsecond=0)
    # Set start_date to 30 days before the beginning of the current hour
    start_date = end_date - timedelta(days=30)

    df = fetch_entsoe_prices(API_KEY, start_date, end_date)
    df, scaler = preprocess_data(df)

    sequence_length = 24
    X, y = create_sequences(df['ScaledPrice'].values, sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_model((sequence_length, 1))
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0) # Reduced verbosity

    # Make predictions on the test set
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Prepare data for future predictions
    last_sequence_scaled = df['ScaledPrice'].values[-sequence_length:]
    future_input = prepare_future_sequence(last_sequence_scaled, sequence_length)

    # Predict the next few hours (e.g., next 5 hours)
    num_future_hours = 5
    future_predictions_scaled = []
    current_sequence = future_input.copy()

    for _ in range(num_future_hours):
        predicted_scaled = model.predict(current_sequence)[0, 0]
        future_predictions_scaled.append(predicted_scaled)
        # Update the sequence by shifting and appending the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = predicted_scaled

    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

    plot_predictions(pd.DataFrame(y_test_rescaled, index=df.index[split+sequence_length:]),
                     pd.DataFrame(predictions_rescaled, index=df.index[split+sequence_length:]),
                     future_predictions_rescaled)

    print("Forecasted prices for the next {} hours:".format(num_future_hours))
    current_hour = end_date
    for i, price in enumerate(future_predictions_rescaled):
        current_hour += timedelta(hours=1)
        print(f"{current_hour.strftime('%Y-%m-%d %H:%M')}: {price[0]:.2f}")