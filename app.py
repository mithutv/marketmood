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

# --- UI & FONT STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    h1 { font-size: 28px !important; font-weight: 600 !important; margin-bottom: 5px !important; }
    h2 { font-size: 20px !important; font-weight: 400 !important; color: #4a4a4a !important; }
    .stMetric label { font-size: 12px !important; text-transform: uppercase; color: #888888; }
    div.stButton > button:first-child { background-color: #007BFF !important; color: white !important; border-radius: 8px !important; }
    </style>
""", unsafe_allow_html=True)

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

# --- APP LAYOUT ---
st.set_page_config(page_title="Marketmood", layout="centered")
st.title("Marketmood: AI-Driven Financial Forecasting")

# Snapshot Preview
st.subheader("Market Snapshot Preview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trend", "Neutral", "↔")
col2.metric("Confidence", "72%")
col3.metric("Volatility", "Medium")
col4.metric("Models", "3 Active")
st.write("---")

# Search Box - Defined ONCE
ticker = st_searchbox(
    search_tickers, 
    placeholder="Enter symbol or company name (e.g., AAPL)...", 
    label="Search market data...",
    key="main_ticker_search"
)

# Forecast Execution
if st.button("Generate Forecast"):
    if not ticker:
        st.warning("Please select a ticker first.")
    else:
        try:
            df = get_stock_data(ticker)
            if df.empty:
                st.error("No data found for this ticker.")
            else:
                # Logic block
                st.success(f"Forecast generated for {ticker}")
                # [Insert your Prophet/ML/Monte Carlo logic here]
        except Exception as e:
            st.error(f"An error occurred: {e}")
