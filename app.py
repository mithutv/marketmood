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

st.title("Marketmood: AI-Driven Financial Forecasting")
st.markdown("""
This application analyzes **4 years of historical price action** to provide a multi-dimensional forecast.
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

            # Calculations
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast_1y = m.predict(m.make_future_dataframe(periods=365))
            price_30 = m.predict(m.make_future_dataframe(periods=30))['yhat'].iloc[-1]
            price_6m = m.predict(m.make_future_dataframe(periods=180))['yhat'].iloc[-1]
            price_1y = forecast_1y['yhat'].iloc[-1]

            # Tab Layout
            tab1, tab2 = st.tabs(["📊 Forecast Dashboard", "🧠 Methodology & Logic"])

            with tab1:
                # Metrics
                cols = st.columns(4)
                cols[0].metric("Current Price", f"${current_price:,.2f}")
                cols[1].metric("30-Day", f"${price_30:,.2f}", f"{price_30 - current_price:+.2f}")
                cols[2].metric("6-Month", f"${price_6m:,.2f}", f"{price_6m - current_price:+.2f}")
                cols[3].metric("1-Year", f"${price_1y:,.2f}", f"{price_1y - current_price:+.2f}")
                st.divider()

                # Prophet
                st.markdown("#### Trend Projection (Prophet)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
                fig.add_trace(go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'], name='Forecast'))
                st.plotly_chart(fig, use_container_width=True)

                # ML
                st.markdown("#### Pattern Predictor (ML)")
                ml_df = prophet_df.copy()
                ml_df['SMA_20'] = ml_df['y'].rolling(window=20, min_periods=1).mean()
                delta = ml_df['y'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                ml_df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
                ml_df['ATR'] = (df['High'] - df['Low']).rolling(window=14, min_periods=1).mean()
                ml_df['PE_Ratio'] = yf.Ticker(ticker).info.get('trailingPE', 20)
                news = yf.Ticker(ticker).news
                ml_df['Sentiment'] = TextBlob(news[0]['title']).sentiment.polarity if (news and isinstance(news, list) and len(news) > 0 and 'title' in news[0]) else 0
                ml_df = ml_df.ffill().bfill()
                
                features = ['SMA_20', 'RSI', 'ATR', 'PE_Ratio', 'Sentiment']
                days_ahead = 252
                if len(ml_df) > days_ahead:
                    X = ml_df[features].iloc[:-days_ahead]
                    y = ml_df['y'].shift(-days_ahead).dropna()
                    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X.iloc[:len(y)], y)
                    pred = model.predict(ml_df[features].iloc[[-1]])[0]
                    
                    st.write("### Model Insight: What drives this prediction?")
                    importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
                    st.bar_chart(importances.set_index('Feature'))
                    ml_col1, ml_col2 = st.columns([1, 2])
                    ml_col1.metric("ML 1-Year Projection", f"${pred:,.2f}", f"{pred - current_price:+.2f}")
                    ml_col2.caption("Random Forest Model: Estimating price for 1 year ahead.")
                
                # Monte Carlo
                st.divider()
                st.markdown("#### Probabilistic Projection: Monte Carlo")
                paths = (prophet_df['y'].iloc[-1] * (1 + np.random.normal(prophet_df['y'].pct_change().mean(), prophet_df['y'].pct_change().std(), (1000, 50))).cumprod(axis=0))
                fig_mc = go.Figure()
                for i in range(50): fig_mc.add_trace(go.Scatter(y=paths[:, i], line=dict(width=1), showlegend=False))
                st.plotly_chart(fig_mc, use_container_width=True)

            with tab2:
                st.header("Methodology: Behind the Forecast")
                st.markdown("""
                ### Why Random Forest?
                Financial markets are non-linear, high-noise environments. Traditional linear models often fail because they assume constant, proportional relationships. I chose **Random Forest Regressor** to identify complex, non-linear patterns by building an ensemble of decision trees. This allows the model to learn conditional logic—for example, how the significance of an RSI signal changes based on market volatility (ATR) or news sentiment.
                
                ### Explainable AI
                My implementation uses **Feature Importance** analysis to turn the model from a "black box" into an explainable tool. This identifies which features—technical indicators, fundamentals (P/E), or news sentiment—are currently driving the prediction.
                
                ### Ensemble Strategy
                By integrating **Prophet** (seasonal trends), **Random Forest** (short-term patterns), and **Monte Carlo** (probabilistic risk), this dashboard provides a rigorous, multi-dimensional forecast.
                """)

    except Exception as e:
        st.error(f"Error: {e}")
