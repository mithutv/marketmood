import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import nltk
from textblob import TextBlob
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- GLOBAL STYLES ---
st.set_page_config(page_title="Market Mood", layout="centered")
st.markdown("""
    <style>
    [data-testid="stDataFrame"] thead tr th { background-color: #000000; color: #FFFFFF; font-weight: 800; font-size: 20px; text-align: center !important; }
    div.stButton > button:first-child { background-color: #007BFF; color: white; border: none; padding: 10px 24px; font-size: 16px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Market Mood: AI-Driven Financial Forecasting")
st.markdown("This application analyzes the **last 4 years** of historical price action to provide projections using **Meta's Prophet**.")

# --- SEARCH BOX ---
def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(search_tickers, placeholder="Start typing a company name...", label="Search for a Company")
if ticker: st.write(f"### Selected Ticker: {ticker}")

@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="5y", threads=False, multi_level_index=False)

# --- FORECAST LOGIC ---
if st.button("Generate Forecast"):
    if not ticker:
        st.warning("Please search for and select a ticker first.")
    else:
        try:
            df = get_stock_data(ticker)
            if df.empty:
                st.error("No data found.")
            else:
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
                
                four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                prophet_df = prophet_df[prophet_df['ds'] >= four_years_ago].dropna()
                current_price = prophet_df['y'].iloc[-1]

                m = Prophet(daily_seasonality=True).fit(prophet_df)
                
                # Predictions
                forecast_30 = m.predict(m.make_future_dataframe(periods=30))
                forecast_6m = m.predict(m.make_future_dataframe(periods=180))
                forecast_1y = m.predict(m.make_future_dataframe(periods=365))
                
                price_30 = forecast_30['yhat'].iloc[-1]
                price_6m = forecast_6m['yhat'].iloc[-1]
                price_1y = forecast_1y['yhat'].iloc[-1]
                
                def get_delta_text(forecasted, current):
                    return f"{forecasted - current:+.2f}"

                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current", f"${current_price:,.2f}")
                col2.metric("30-Day", f"${price_30:,.2f}", get_delta_text(price_30, current_price))
                col3.metric("6-Month", f"${price_6m:,.2f}", get_delta_text(price_6m, current_price))
                col4.metric("1-Year", f"${price_1y:,.2f}", get_delta_text(price_1y, current_price))
                
               # Monte Carlo Simulation
                with st.expander("View 1-Year Monte Carlo Simulation"):
                    st.write("### 1-Year Monte Carlo Projection")
                    returns = prophet_df['y'].pct_change().dropna()
                    mu, sigma = returns.mean(), returns.std()
                    days, sims = 252, 500
                    daily_returns = np.random.normal(mu, sigma, (days, sims))
                    price_paths = current_price * (1 + daily_returns).cumprod(axis=0)
                    median_path = np.median(price_paths, axis=1)
                    
                    fig_mc = go.Figure()
                    for i in range(50):
                        fig_mc.add_trace(go.Scatter(x=list(range(days)), y=price_paths[:, i], 
                                         line=dict(color='lightgray', width=1), showlegend=False))
                    
                    fig_mc.add_trace(go.Scatter(x=list(range(days)), y=median_path, 
                                     line=dict(color='blue', width=3), name='Median (50%) Path'))
                    
                    fig_mc.update_layout(template="plotly_white", xaxis_title="Days", yaxis_title="Price")
                    st.plotly_chart(fig_mc, use_container_width=True)
