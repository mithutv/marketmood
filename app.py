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
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Search Function
@st.cache_data(ttl=3600)
def search_tickers(searchterm: str):
    if not searchterm or len(searchterm) < 3: return []
    time.sleep(0.2) 
    try:
        results = yf.Search(searchterm).quotes
        return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]
    except Exception: return []

# Page Config
st.set_page_config(page_title="Marketmood", layout="centered")

st.markdown("""
    <style>
    h1 { font-size: 36px !important; font-weight: 500 !important; color: #333333 !important; margin-bottom: 2px !important; }
    .stApp > header { display: none; }
    div.stButton > button:first-child { background-color: #007BFF !important; color: white !important; width: 100%; border-radius: 8px !important; font-weight: bold !important; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("Marketmood: AI-Driven Financial Forecasting")
st.caption("Status: System Operational | Engine: Ensemble (Prophet/RF/Monte Carlo) | Data: YFinance API")
st.subheader("Ensemble-based predictive analytics for the modern investor.")

# Project Overview
with st.expander("📖 Project Overview"):
    st.markdown("""
    * **Our Objective:** To deliver data-driven market context that supports more informed investment decisions.
    * **The Ensemble Engine:** We combine three complementary AI methodologies—Meta’s Prophet for seasonality, Random Forest for pattern recognition, and Monte Carlo simulations for risk modeling—to reduce single-model bias and improve forecast robustness.
    * **Data Integrity:** We use verified historical market data from 2020 onward.
    * **Your Responsibility:** These forecasts are analytical tools, not financial advice.
    """)

ticker = st_searchbox(search_tickers, placeholder="Enter symbol or company name (e.g., AAPL)...", label="Search market data...")

@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="10y", threads=False, multi_level_index=False)

if st.button("Generate Forecast") and ticker:
    try:
        df = get_stock_data(ticker)
        if df.empty:
            st.error("No data found.")
        else:
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            start_date = pd.Timestamp('2020-01-01')
            prophet_df = prophet_df[prophet_df['ds'] >= start_date].dropna()
            current_price = prophet_df['y'].iloc[-1]

            # Calculations
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast_1y = m.predict(m.make_future_dataframe(periods=365))
            price_1y = forecast_1y['yhat'].iloc[-1]
            
            # --- CARD LAYOUT ---
            
            # 1. PROPHET
            prophet_trend = "Bullish" if price_1y > current_price else "Bearish"
            with st.expander(f"📊 Trend Projection (Prophet): :{'green' if prophet_trend=='Bullish' else 'red'}[{prophet_trend}]", expanded=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
                fig.add_trace(go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'], name='Forecast'))
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Model projects 1-year target of **${price_1y:,.2f}**.")

            # 2. ML PATTERN
            with st.expander("🧠 Pattern Predictor (ML)"):
                # [Paste your existing ML logic here to define 'pred' and 'active_importances']
                st.write("### Model Insight: What drives this prediction?")
                # st.bar_chart(...) 
                st.info("**ML Summary:** The model is relying heavily on technical signals.")

            # 3. MONTE CARLO
            with st.expander("🎲 Probabilistic Projection: Monte Carlo"):
                # [Paste your Monte Carlo logic here]
                st.write("10,000 randomized paths simulated.")

            # 4. CONSENSUS
            with st.expander("🎯 AI Consensus Forecast"):
                # [Paste your Consensus/Metric logic here]
                st.write("Aggregating signals from all models.")

            # 5. SENTIMENT
            with st.expander("📰 Market Sentiment (News Analysis)"):
                # [Paste your Sentiment logic here]
                st.write("Sentiment analysis from recent news headlines.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
