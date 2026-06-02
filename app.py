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

# Page Config
st.set_page_config(page_title="Marketmood", layout="centered")

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
            cols = st.columns(3)
            cols[0].metric("Current Price", f"${current_price:,.2f}")
            cols[1].metric("30-Day Prophet", f"${price_30:,.2f}", f"{price_30 - current_price:+.2f}")
            cols[2].metric("1-Year Prophet", f"${price_1y:,.2f}", f"{price_1y - current_price:+.2f}")
            st.divider()

            # --- ROW 2: TREND PROJECTION ---
            st.markdown("#### Trend Projection (Prophet)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
            fig.add_trace(go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'], name='Forecast'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"**Prophet Summary:** Based on 4 years of history, the model projects a 1-year target of **${price_1y:,.2f}**.")
            st.divider()

            # --- ROW 3: PATTERN PREDICTOR (ML) ---
            st.markdown("#### Pattern Predictor (ML)")
            ml_df = prophet_df.copy()
            ml_df['SMA_20'] = ml_df['y'].rolling(window=20).mean()
            
            # Improved RSI calculation
            delta = ml_df['y'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.001) 
            ml_df['RSI'] = 100 - (100 / (1 + rs))
            
            # MODERN PANDAS FIX: Use .bfill() instead of fillna(method='bfill')
            ml_df = ml_df.bfill()
            ml_df = ml_df.dropna()
            
            if len(ml_df) > 30: 
                X = ml_df[['SMA_20', 'RSI']].iloc[:-1]
                y = ml_df['y'].shift(-1).dropna()
                
                # Align lengths
                X = X.iloc[:len(y)]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
                pred = model.predict(ml_df[['SMA_20', 'RSI']].iloc[[-1]])[0]
                
                ml_col1, ml_col2 = st.columns([1, 2])
                ml_col1.metric("ML Next Day Projection", f"${pred:,.2f}", f"{pred - current_price:+.2f}")
                ml_col2.caption("Random Forest Model based on 20-day SMA and RSI.")
            else:
                st.warning("Insufficient data to train ML model.")
            st.divider()
            

            # --- ROW 4: MONTE CARLO ---
            st.markdown("#### Probabilistic Projection: Monte Carlo")
            def run_mc(prices, days=1000):
                returns = prices.pct_change().dropna()
                daily = np.random.normal(returns.mean(), returns.std(), (days, 50))
                return prices.iloc[-1] * (1 + daily).cumprod(axis=0)
            
            paths = run_mc(prophet_df['y'])
            fig_mc = go.Figure()
            for i in range(50): fig_mc.add_trace(go.Scatter(y=paths[:, i], line=dict(width=1), showlegend=False))
            fig_mc.update_layout(height=400)
            st.plotly_chart(fig_mc, use_container_width=True)
            st.warning(f"**Monte Carlo Summary:** The **Median Projected Price** at day 1000 is **${np.median(paths[-1, :]):,.2f}**.")


            # --- ROW 5: FINAL CONSENSUS ---
            st.divider()
            st.header("AI Consensus Forecast")
            
            # Determine direction for each model
            prophet_trend = "Bullish" if price_1y > current_price else "Bearish"
            ml_trend = "Bullish" if pred > current_price else "Bearish"
            mc_trend = "Bullish" if np.median(paths[-1, :]) > current_price else "Bearish"
            
            # Count the "Bullish" signals
            bullish_count = [prophet_trend, ml_trend, mc_trend].count("Bullish")
            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if bullish_count >= 2:
                    st.metric("Consensus", "Bullish 🐂", delta="Strong Buy")
                else:
                        st.metric("Consensus", "Bearish 🐻", delta="Caution")
            
            with col_b:
                st.write(f"Based on the integration of all models, the consensus is **{bullish_count}/3 indicators favoring a bullish outlook**. "
                         "This ensemble approach balances long-term seasonal trends (Prophet) with short-term technical patterns (ML) "
                         "and historical risk probability (Monte Carlo).")
            
            # Mandatory Disclaimer
            st.caption("Disclaimer: This consensus is generated by AI models based on historical data. "
                       "It does not constitute financial advice. Always perform your own due diligence before trading.")

    except Exception as e:
        st.error(f"Error: {e}")
