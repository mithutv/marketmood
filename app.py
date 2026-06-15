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

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    h1 { font-size: 28px !important; font-weight: 600 !important; margin-bottom: 5px !important; }
    h2 { font-size: 20px !important; font-weight: 400 !important; color: #4a4a4a !important; }
    
    .stMetric label {
        font-size: 12px !important;
        font-weight: 400 !important;
        color: #888888 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stMetric .css-1xarl3l { font-size: 18px !important; }
    
    div.stButton > button:first-child {
        background-color: #007BFF !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# Setup NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@st.cache_data(ttl=3600)
def search_tickers(searchterm: str):
    if not searchterm or len(searchterm) < 3: 
        return []
    time.sleep(0.2) 
    try:
        results = yf.Search(searchterm).quotes
        return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) 
                for q in results if 'symbol' in q]
    except Exception:
        return []

# Page Config
st.set_page_config(page_title="Marketmood", layout="centered")

# Header & Scope Note
st.title("Marketmood: AI-Driven Financial Forecasting")
st.subheader("Multi-model stock forecasting using Prophet, Random Forest, and Monte Carlo simulations with confidence scoring and backtesting")
st.caption("Advanced multi-model forecasting powered by Prophet, Random Forest, and Monte Carlo simulations.")

with st.expander("🛈 About Marketmood: How it works & Disclaimer"):
    st.markdown("""
    Marketmood is an ensemble-based financial forecasting suite designed to bridge the gap between complex machine learning and actionable market insights.
    * **The Ensemble Engine:** We synthesize three distinct AI methodologies—**Meta’s Prophet** for seasonality, **Random Forest** for pattern recognition, and **Monte Carlo** for risk assessment.
    """)

# --- MARKET SNAPSHOT PREVIEW ---
st.subheader("Market Snapshot Preview")
col1, col2, col3
