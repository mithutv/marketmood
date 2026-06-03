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

# Setup NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="Marketmood", layout="centered")

st.title("Marketmood: AI-Driven Financial Forecasting")

def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(search_tickers, placeholder="Enter symbol...", label="Search market data...")

@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="10y", threads=False, multi_level_index=False)

if st.button("Generate Forecast") and ticker:
    try:
        df = get_stock_data(ticker)
        if df.empty: st.error("No data found."); st.stop()
        
        target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df[prophet_df['ds'] >= pd.Timestamp('2020-01-01')].dropna()
        current_price = prophet_df['y'].iloc[-1]

        # --- CALCULATIONS (All models must run before Consensus) ---
        # 1. Prophet
        m = Prophet(daily_seasonality=True).fit(prophet_df)
        forecast_1y = m.predict(m.make_future_dataframe(periods=365))
        price_1y = forecast_1y['yhat'].iloc[-1]

        # 2. ML (Pattern Predictor)
        ml_df = prophet_df.copy()
        ml_df['SMA_20'] = ml_df['y'].rolling(20, min_periods=1).mean()
        ml_df['RSI'] = 50 # Simplified placeholder for brevity, use full logic here
        ml_df['ATR'] = (df['High'] - df['Low']).reindex(ml_df.index).rolling(14).mean().fillna(0)
        ml_df['Volume'] = df['Volume'].reindex(ml_df.index).fillna(0)
        ml_df['Log_Return'] = np.log(ml_df['y'] / ml_df['y'].shift(1)).fillna(0)
        features = ['SMA_20', 'RSI', 'ATR', 'Volume', 'Log_Return']
        ml_df = ml_df.ffill().bfill().dropna()
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(ml_df[features], ml_df['y'].pct_change(252).shift(-252).dropna())
        pred = current_price * (1 + model.predict(ml_df[features].iloc[[-1]])[0])

        # 3. Monte Carlo
        paths = df[target_col].pct_change().dropna().pipe(lambda r: (1 + np.random.normal(r.mean(), r.std(), (252, 10000))).cumprod(axis=0) * current_price)
        median_val = np.median(paths[-1, :])

        # --- ROW 1: METRICS ---
        cols = st.columns(4)
        cols[0].metric("Current", f"${current_price:,.2f}")
        cols[1].metric("30-Day", f"${m.predict(m.make_future_dataframe(30))['yhat'].iloc[-1]:,.2f}")
        cols[2].metric("6-Month", f"${m.predict(m.make_future_dataframe(180))['yhat'].iloc[-1]:,.2f}")
        cols[3].metric("1-Year", f"${price_1y:,.2f}")
        st.divider()

        # --- ROW 1.5: CONSENSUS (THE SUMMARY) ---
        st.header("AI Consensus Forecast")
        bullish = [price_1y > current_price, pred > current_price, median_val > current_price].count(True)
        conviction = int(min((bullish / 3 * 100) + 10, 100))
        st.metric("Consensus", "Bullish 🐂" if bullish >= 2 else "Bearish 🐻", f"{conviction}% Conviction")
        st.progress(conviction / 100)
        st.divider()

        # --- ROW 2: PROPHET CHART ---
        st.markdown("#### Trend Projection (Prophet)")
        fig = go.Figure([go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'])])
        st.plotly_chart(fig, use_container_width=True)

        # --- ROW 3: ML CHART ---
        st.markdown("#### Pattern Predictor (ML)")
        st.bar_chart(pd.DataFrame({'Importance': model.feature_importances_}, index=features).sort_values('Importance', ascending=False))

        # --- ROW 4: MONTE CARLO CHART ---
        st.markdown("#### Monte Carlo Simulation")
        st.line_chart(paths[:, :100])

    except Exception as e: st.error(f"Error: {e}")
