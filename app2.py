import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------
# Feature Engineering Function
# -------------------------------
def create_features(df):
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # Lag features
    df['lag1'] = df['Close'].shift(1)
    df['lag2'] = df['Close'].shift(2)
    df['lag3'] = df['Close'].shift(3)

    # Returns
    df['return_1d'] = df['Close'].pct_change()

    # Rolling windows (small -> avoids NaNs)
    df['roll_mean_3'] = df['Close'].rolling(3).mean()
    df['roll_std_3'] = df['Close'].rolling(3).std()

    # Target variable (next day close)
    df['target'] = df['Close'].shift(-1)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    return df


# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("📈 Stock EDA & Real-Time Prediction App")
st.write("Enter a stock ticker and perform full analysis + next-day prediction.")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TCS.NS, RELIANCE.NS):", "AAPL")

start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
run = st.button("Run Analysis & Predict")

if run:
    st.subheader(f"📥 Downloading Data for {ticker}...")
    df = yf.download(ticker, start=start_date)
    
    if df.empty:
        st.error("Ticker not found or no data available.")
    else:
        df.reset_index(inplace=True)
        st.success("Data Loaded Successfully!")
        st.dataframe(df.head())

        # -------------------------------
        # EDA Section
        # -------------------------------
        st.subheader("📊 Exploratory Data Analysis")

        # Closing Price Trend
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df['Date'], df['Close'])
        ax.set_title("Closing Price Over Time")
        st.pyplot(fig)

        # Moving Averages
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df['Date'], df['Close'], label='Close')
        ax.plot(df['Date'], df['MA20'], label='20-Day MA')
        ax.plot(df['Date'], df['MA50'], label='50-Day MA')
        ax.legend()
        st.pyplot(fig)

        # Returns distribution
        df['Return'] = df['Close'].pct_change()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(df['Return'].dropna(), bins=50)
        ax.set_title("Histogram of Daily Returns")
        st.pyplot(fig)

        # -------------------------------
        # Feature Engineering
        # -------------------------------
        st.subheader("🧠 Feature Engineering")
        df_feat = create_features(df)
        st.dataframe(df_feat.head())

        features = ['lag1','lag2','lag3','return_1d','roll_mean_3','roll_std_3']

        X = df_feat[features]
        y = df_feat['target']

        # -------------------------------
        # Model Training
        # -------------------------------
        st.subheader("🤖 Model Training")

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)

        rmses = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            rmses.append(rmse)

        st.write(f"**Average RMSE:** {np.mean(rmses):.4f}")

        # -------------------------------
        # Next Day Prediction
        # -------------------------------
        st.subheader("🚀 Real-Time Next-Day Prediction")

        last_row = df_feat.iloc[-1]
        last_features = last_row[features].values.reshape(1, -1)
        next_day_pred = model.predict(last_features)[0]

        st.success(f"📌 **Predicted Next-Day Close for {ticker}: {next_day_pred:.2f}**")

        # -------------------------------
        # Show last 30-day comparison
        # -------------------------------
        st.subheader("🔍 Last Test Predictions Visualization")

        X_last_test = X.iloc[test_idx]
        y_last_test = y.iloc[test_idx]
        pred_last_test = model.predict(X_last_test)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_last_test.values, label="Actual Next Close")
        ax.plot(pred_last_test, label="Predicted Next Close")
        ax.legend()
        st.pyplot(fig)

        st.info("Model predicts the **next-day closing price** based on historical patterns.")

