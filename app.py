import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import nltk
from textblob import TextBlob
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.markdown("""
    <style>
    /* Targeting the Generate Forecast button */
    div.stButton > button:first-child {
        background-color: #007BFF !important; /* Force Blue */
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

# Page Config
st.set_page_config(page_title="Marketmood", layout="centered")

# Header & Scope Note
st.title("Marketmood: AI-Driven Financial Forecasting")
st.markdown("""
This application analyzes **4 years of historical price action** to provide a multi-dimensional forecast:
* **Trend Projection:** Uses **Meta's Prophet** for seasonal time-series analysis.
* **Pattern Predictor:** Uses a **Random Forest Regressor** to identify technical indicator signals (SMA, RSI).
* **Risk Assessment:** Uses a **Monte Carlo Simulation** to map potential 1000-day price volatility.
""")

def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(search_tickers, placeholder="Enter symbol or company name (e.g., AAPL)...", label="Search market data...")

@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="5y", threads=False, multi_level_index=False)

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
            four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
            prophet_df = prophet_df[prophet_df['ds'] >= four_years_ago].dropna()
            current_price = prophet_df['y'].iloc[-1]

            # Prophet Logic
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast_30 = m.predict(m.make_future_dataframe(periods=30))
            forecast_1y = m.predict(m.make_future_dataframe(periods=365))
            price_30, price_1y = forecast_30['yhat'].iloc[-1], forecast_1y['yhat'].iloc[-1]

            # --- ROW 1: METRICS ---
            st.subheader(f"Dashboard Summary: {ticker}")
            cols = st.columns(3)
            cols[0].metric("Current Price", f"${current_price:,.2f}")
            cols[1].metric("30-Day Prophet", f"${price_30:,.2f}")
            cols[2].metric("1-Year Prophet", f"${price_1y:,.2f}")
            st.divider()

           To provide a deeper level of insight for your users, we can add a summary section below each graph. This transforms your dashboard from a simple visualization tool into an analytical report.

Here is the updated layout for the Prophet and Monte Carlo sections, including the statistical summaries:

Updated Layout Code
Replace the relevant sections in your app.py with this logic:

Python
            # --- ROW 2: TREND PROJECTION (Prophet) ---
            st.markdown("#### 📈 Trend Projection (Prophet)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
            fig.add_trace(go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'], name='Forecast'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Prophet Summary
            st.info(f"**Prophet Summary:** Based on 4 years of history, the model projects a 1-year target of **${price_1y:,.2f}**. "
                    f"This trend analysis accounts for historical seasonality, identifying cyclical patterns that suggest a move of "
                    f"{((price_1y - current_price)/current_price)*100:+.1f}% over the next 12 months.")
            st.divider()

            # --- ROW 4: MONTE CARLO ---
            st.markdown("#### 🎲 Probabilistic Projection (Monte Carlo)")
            # ... (your existing MC plotting code here) ...
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Monte Carlo Summary
            median_final = paths[-1, :].mean() # Average of all 50 simulations
            st.warning(f"**Monte Carlo Summary:** We simulated 50 possible future price paths based on the stock's historical volatility. "
                       f"The **Median Projected Price** at day 1000 is **${median_final:,.2f}**. "
                       f"The spread between the highest and lowest paths represents the 'volatility cloud'—the wider this cloud, the higher the risk for this specific asset.")
            st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
