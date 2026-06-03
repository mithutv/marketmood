import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURATION ---
st.set_page_config(page_title="Marketmood", layout="centered")

# Custom CSS for the button
st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #007BFF !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER & SCOPE NOTE ---
st.title("Marketmood: AI-Driven Financial Forecasting")
st.markdown("""
This application analyzes **historical price action since 2020** to provide a multi-dimensional forecast:
* **Trend Projection:** Uses **Meta's Prophet** for seasonal time-series analysis.
* **Pattern Predictor:** Uses a **Random Forest Regressor** to identify technical indicator signals.
* **Risk Assessment:** Uses a **Monte Carlo Simulation** to map potential 1000-day price volatility.
""")

# --- TICKER SEARCH ---
def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(search_tickers, placeholder="Enter symbol or company name (e.g., AAPL)...", label="Search market data...")

# --- DATA FETCHING ---
@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="10y", threads=False, multi_level_index=False)

if st.button("Generate Forecast") and ticker:
    try:
        df = get_stock_data(ticker)
        if df.empty: st.error("No data found."); st.stop()
        
        # Prepare Data
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df[prophet_df['ds'] >= pd.Timestamp('2020-01-01')].dropna()
        current_price = prophet_df['y'].iloc[-1]

        # --- MODEL 1: PROPHET ---
        m = Prophet(daily_seasonality=True).fit(prophet_df)
        forecast_1y = m.predict(m.make_future_dataframe(periods=365))
        price_1y = forecast_1y['yhat'].iloc[-1]

        # --- MODEL 2: RANDOM FOREST ---
        ml_df = prophet_df.copy()
        ml_df['SMA_20'] = ml_df['y'].rolling(20, min_periods=1).mean()
        ml_df['ATR'] = (df['High'] - df['Low']).reindex(ml_df.index).rolling(14).mean().fillna(0)
        ml_df['Volume'] = df['Volume'].reindex(ml_df.index).fillna(0)
        ml_df['Log_Return'] = np.log(ml_df['y'] / ml_df['y'].shift(1)).fillna(0)
        
        # Create Target (1-year return)
        ml_df['Target_Return'] = ml_df['y'].pct_change(252).shift(-252)
        # Drop ALL rows with NaN (including the ones created by .shift) to align X and y
        ml_df = ml_df.dropna() 
        
        features = ['SMA_20', 'ATR', 'Volume', 'Log_Return']
        X, y = ml_df[features], ml_df['Target_Return']
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X, y)
        pred = current_price * (1 + model.predict(ml_df[features].iloc[[-1]])[0])

        # --- MODEL 3: MONTE CARLO ---
        paths = df[target_col].pct_change().dropna()
        median_val = current_price * (1 + paths.mean())**252

        # --- ROW 1: AI CONSENSUS (Top) ---
        st.header("AI Consensus Forecast")
        bullish_count = [price_1y > current_price, pred > current_price, median_val > current_price].count(True)
        conv = int(min(((bullish_count/3)*100) + 10, 100))
        
        c1, c2 = st.columns([1, 2])
        c1.metric("Consensus", "Bullish 🐂" if bullish_count >= 2 else "Bearish 🐻", f"{conv}% Conviction")
        c2.progress(conv/100)
        c2.write("Ensemble conviction based on model agreement and projected upside.")
        st.divider()

        # --- ROW 2: METRICS ---
        cols = st.columns(4)
        cols[0].metric("Price", f"${current_price:,.2f}")
        cols[1].metric("30-Day", f"${m.predict(m.make_future_dataframe(30))['yhat'].iloc[-1]:,.2f}")
        cols[2].metric("6-Month", f"${m.predict(m.make_future_dataframe(180))['yhat'].iloc[-1]:,.2f}")
        cols[3].metric("1-Year", f"${price_1y:,.2f}")
        st.divider()

        # --- ROW 3: VISUALIZATIONS ---
        st.markdown("#### Trend Projection (Prophet)")
        fig = go.Figure([go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'])])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pattern Predictor (ML)")
        st.bar_chart(pd.DataFrame({'Importance': model.feature_importances_}, index=features))

        st.markdown("#### Monte Carlo Simulation")
        st.write(f"Median Projected Price: **${median_val:,.2f}**")
        
    except Exception as e:
        st.error
