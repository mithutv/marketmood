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
st.set_page_config(page_title="Marketmood", layout="wide")

st.markdown("""
    <style>
    h1 { font-size: 32px !important; font-weight: 500 !important; color: #333333 !important; }
    .stApp > header { display: none; }
    div.stButton > button:first-child { background-color: #007BFF !important; color: white !important; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Marketmood")
    st.markdown("---")
    ticker = st_searchbox(search_tickers, placeholder="Search ticker...", label="Target Asset", key="main_ticker_search")
    st.markdown("### Model Settings")
    lookback = st.slider("Lookback Period (Years)", 1, 10, 5)
    generate_btn = st.button("Generate Forecast")
    st.markdown("---")
    st.caption("Advanced multi-model forecasting.")

# --- MAIN AREA ---
st.title("Marketmood: AI-Powered Market Forecasting")
st.subheader("Ensemble-based predictive analytics for the modern investor.")

# --- ACCORDION (ABOUT) ---
with st.expander("🛈 About Marketmood: Objective & Methodology"):
    st.markdown("""
    **Objective:** To provide data-driven market context that helps investors organize ambiguity.
    
    **The Ensemble Engine:**
    - **Meta’s Prophet:** For seasonal time-series analysis.
    - **Random Forest:** For pattern recognition using technical indicators.
    - **Monte Carlo:** For probabilistic volatility and risk assessment.
    
    *Disclaimer: This tool is for informational purposes only and does not constitute financial advice.*
    """)

st.write("---")

# Snapshot Preview
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trend", "Neutral", "↔")
col2.metric("Confidence", "72%")
col3.metric("Volatility", "Medium")
col4.metric("Models", "3 Active")

if generate_btn and ticker:
    try:
        df = get_stock_data(ticker)
        if df.empty:
            st.error("No data found.")
        else:
            st.success(f"Forecast generated for {ticker}")
            # [Your existing logic continues here...]
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Use the sidebar to search for a ticker and generate your forecast.")
