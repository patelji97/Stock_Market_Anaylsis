

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="📈  Stock  Market Analyzer + Prediction",
    page_icon="💹",
    layout="wide"
)

# ===== TITLE =====
st.markdown("<h1 style='text-align:center; color:darkblue;'>📊 Interactive Stock Market Analyzer + 30-days Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Analyze, visualize & predict stock trends easily</h4>", unsafe_allow_html=True)
st.write("---")

# ===== SIDEBAR =====
st.sidebar.header("📌 Select Options")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# ===== FETCH DATA =====
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("❌ No data found. Check stock symbol or date range.")
else:
    st.success(f"✅ Successfully fetched data for {ticker} ({len(data)} records)")

    # ===== DATA PREVIEW =====
    st.subheader("📄 Data Preview")
    st.dataframe(data.tail(10))

    # ===== MOVING AVERAGES =====
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    data['Daily Return'] = data['Close'].pct_change()

    # ===== METRICS =====
    st.subheader("📊 Stock Statistics")
    col1, col2, col3 = st.columns(3)
    try:
        latest_close = float(data['Close'].iat[-1])
    except:
        latest_close = 0.0
    try:
        mean_return = float(data['Daily Return'].mean() * 100)
    except:
        mean_return = 0.0
    try:
        price_std = float(data['Close'].std())
    except:
        price_std = 0.0

    col1.metric("Latest Close", f"${latest_close:.2f}", "")
    col2.metric("Mean Daily Return", f"{mean_return:.2f}%", "")
    col3.metric("Price Std Dev", f"{price_std:.2f}", "")

    # ===== PLOT CLOSING PRICE =====
    st.subheader("💰 Closing Price Chart")
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Close', color='darkblue')
    plt.title(f'{ticker} Closing Price', color='darkred', fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(plt)

    # ===== MOVING AVERAGES PLOT =====
    st.subheader("📈 Closing Price with Moving Averages")
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Close', color='darkblue')
    plt.plot(data['MA50'], label='MA50', color='orange')
    plt.plot(data['MA200'], label='MA200', color='green')
    plt.title(f'{ticker} Stock + MA50 & MA200', color='darkred', fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(plt)

    # ===== DAILY RETURNS PLOT =====
    st.subheader("📉 Daily Returns")
    plt.figure(figsize=(12,6))
    plt.plot(data['Daily Return'], label='Daily Returns', color='purple')
    plt.title(f'{ticker} Daily Returns', color='darkred', fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(plt)

    # ===== LSTM PREDICTION =====
    st.subheader("🤖 Predict Next 30 Days Closing Price")
    if st.button("Predict Next 30 Days"):
        st.info("⏳ Training LSTM model... please wait")

        # Prepare data
        close_prices = data['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(close_prices)

        # Train on last 80%
        train_size = int(len(scaled_data)*0.8)
        train_data = scaled_data[:train_size]
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i,0])
            y_train.append(train_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

        # LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Prepare input for prediction (last 60 days)
        last_60 = scaled_data[-60:]
        pred_input = last_60.reshape(1,60,1)

        future_predictions = []
        for _ in range(30):
            pred = model.predict(pred_input)
            future_predictions.append(pred[0,0])
            pred_input = np.append(pred_input[:,1:,:],[[[pred[0,0]]]], axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

        # Plot future predictions
        st.subheader("📊 Next 30 Days Predicted Closing Prices")
        plt.figure(figsize=(12,6))
        plt.plot(range(len(data)), close_prices, label='Historical Close', color='blue')
        plt.plot(range(len(data), len(data)+30), future_predictions, label='Predicted Close', color='red')
        plt.title(f'{ticker} Historical + Predicted Close Prices', color='darkred', fontsize=16)
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)
        st.success("✅ Prediction Complete!")
