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

# Setup NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

@st.cache_data(ttl=3600)
def search_tickers(searchterm: str):
    if not searchterm or len(searchterm) < 3: return []
    time.sleep(0.2)
    try:
        results = yf.Search(searchterm).quotes
        return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]
    except Exception: return []

@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="10y", threads=False, multi_level_index=False)

# Page Config
st.set_page_config(page_title="Marketmood", layout="wide") # Changed to wide for terminal feel

st.markdown("""
    <style>
    h1 { font-size: 32px !important; font-weight: 500 !important; color: #333333 !important; }
    .stApp > header { display: none; }
    div.stButton > button:first-child { background-color: #007BFF !important; color: white !important; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ORGANIZATION ---
with st.sidebar:
    st.title("Marketmood")
    st.markdown("---")
    
    ticker = st_searchbox(
        search_tickers,
        placeholder="Search ticker (e.g. AAPL)...",
        label="Target Asset",
        key="main_ticker_search"
    )
    
    st.markdown("### Model Settings")
    # Placeholder for future sliders
    lookback = st.slider("Lookback Period (Years)", 1, 10, 5)
    
    generate_btn = st.button("Generate Forecast")
    st.markdown("---")
    st.caption("Advanced multi-model forecasting.")

# --- MAIN AREA ---
st.title("Marketmood: AI-Powered Market Forecasting")
st.subheader("Ensemble-based predictive analytics for the modern investor.")

# Snapshot Preview
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trend", "Neutral", "↔")
col2.metric("Confidence", "72%")
col3.metric("Volatility", "Medium")
col4.metric("Models", "3 Active")
st.write("---")

if generate_btn and ticker:
    try:
        df = get_stock_data(ticker)
        if df.empty:
            st.error("No data found.")
        else:
            # [Your existing logic continues here...]
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # (Keep the rest of your original logic here...)
            st.success(f"Forecast generated for {ticker}")
            st.line_chart(prophet_df.set_index('ds'))
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Use the sidebar to search for a ticker and generate your forecast.")
