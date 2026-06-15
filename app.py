import streamlit as st
import time
import yfinance as yf
import pandas as pd
import feedparser
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import nltk
from textblob import TextBlob
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- SETUP ---
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

st.set_page_config(page_title="Marketmood Terminal", layout="wide")

# --- FUNCTIONS ---
@st.cache_data(ttl=3600)
def search_tickers(searchterm: str):
    if not searchterm or len(searchterm) < 3: return []
    try:
        results = yf.Search(searchterm).quotes
        return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]
    except Exception: return []

@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="10y", threads=False, multi_level_index=False)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Forecasting Engine")
    st.info("Find an asset to initiate a multi-model ensemble synthesis.")
    st.markdown("---")
    ticker = st_searchbox(search_tickers, placeholder="e.g. AAPL or Apple...", label="Select Asset")
    st.markdown("### Model Settings")
    lookback = st.slider("Lookback Period (Years)", 1, 10, 5)
    generate_btn = st.button("Generate Forecast", use_container_width=True)

# --- MAIN AREA ---
st.title("Marketmood: Financial Forecasting Terminal")

if generate_btn and ticker:
    with st.spinner(f"Running Ensemble Analysis for {ticker}..."):
        # 1. DATA PREP
        df = get_stock_data(ticker)
        target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
        
        # 2. MODELS
        m = Prophet(daily_seasonality=True).fit(prophet_df)
        forecast = m.predict(m.make_future_dataframe(periods=365))
        
        # 3. TABS
        tab1, tab2, tab3 = st.tabs(["📊 Forecasts", "🧠 ML Pattern Analysis", "📰 News & Sentiment"])

        with tab1:
            st.subheader("Market Trend & Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Random Forest Predictor")
            # --- PASTE YOUR ML LOGIC HERE ---
            st.write("ML Pattern analysis active.") 

        with tab3:
            st.subheader("Market Sentiment")
            # --- PASTE YOUR NEWS/SENTIMENT LOGIC HERE ---
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]: 
                st.markdown(f"• {entry.title}")

else:
    st.info("👈 Use the sidebar to select an asset and generate your forecast.")
