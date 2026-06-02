import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# 1. Caching prevents "Too Many Requests" (429) errors
@st.cache_data(ttl=86400)
def get_stock_data(ticker):
    # threads=False and multi_level_index=False are critical fixes
    df = yf.download(ticker, threads=False, multi_level_index=False)
    return df

st.title("MarketMood AI - Professional Edition")

ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL)", value="NVDA").upper()

if st.button("Generate Forecast"):
    try:
        # Fetch Data
        df = get_stock_data(ticker)
        
        if df.empty:
            st.error("No data found. Please check the ticker symbol.")
        else:
            # 2. FLATTEN DATA: Ensure it is a clean DataFrame
            # Reset index so 'Date' is a column, then select exactly what we need
            df_reset = df.reset_index()
            
            # Keep only Date and Close, renaming them for Prophet
            prophet_df = df_reset[['Date', 'Close']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Verify data is clean
            st.write("Data Sample:", prophet_df.head())

            # 3. PROPHET ENGINE
            m = Prophet(daily_seasonality=True)
            m.fit(prophet_df)
            
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Debug Info: Ensure your requirements.txt includes 'prophet' and 'yfinance>=0.2.64'")
