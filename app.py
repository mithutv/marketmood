import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet

# 1. THE FIX: Aggressive Caching
# This decorator ensures the function only runs once every 24 hours (86400 seconds)
# per unique ticker input.
@st.cache_data(ttl=86400)
def get_data(ticker):
    # threads=False is more stable for cloud environments
    # auto_adjust=True fixes common indexing errors
    return yf.download(ticker, threads=False, auto_adjust=True)

st.title("MarketMood AI")

ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL)", value="NVDA")

if st.button("Generate Forecast"):
    try:
        # Fetching data using the cached function
        df = get_data(ticker)
        
        if df.empty:
            st.error("No data found for this ticker. Check your spelling.")
        else:
            st.success(f"Data successfully loaded for {ticker}")
            
            # --- PROPHET FORECAST ---
            # Reset index to get the Date column for Prophet
            df_reset = df.reset_index()
            # Rename columns to Prophet's expected format: 'ds' (date) and 'y' (price)
            prophet_df = df_reset[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # Build and fit model
            m = Prophet(daily_seasonality=True)
            m.fit(prophet_df)
            
            # Predict
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.add_trace(go.Scatter(x=df_reset['Date'], y=df_reset['Close'], name='Actual'))
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
