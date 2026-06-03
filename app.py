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
This application analyzes **historical price action since 2020** to provide a multi-dimensional forecast:
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
            
            # --- FILTER DATA FROM 2020 ---
            start_date = pd.Timestamp('2020-01-01')
            prophet_df = prophet_df[prophet_df['ds'] >= start_date].dropna()
            current_price = prophet_df['y'].iloc[-1]

            # Prophet Logic
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast_30 = m.predict(m.make_future_dataframe(periods=30))
            forecast_6m = m.predict(m.make_future_dataframe(periods=180))
            forecast_1y = m.predict(m.make_future_dataframe(periods=365))
            
            # Define the prices
            price_30 = forecast_30['yhat'].iloc[-1]
            price_6m = forecast_6m['yhat'].iloc[-1]
            price_1y = forecast_1y['yhat'].iloc[-1]

            # --- ROW 1: METRICS ---
            delta_30 = price_30 - current_price
            delta_6m = price_6m - current_price
            delta_1y = price_1y - current_price

            cols = st.columns(4)
            cols[0].metric("Current Price", f"${current_price:,.2f}")
            cols[1].metric("30-Day", f"${price_30:,.2f}", f"{delta_30:+.2f}")
            cols[2].metric("6-Month", f"${price_6m:,.2f}", f"{delta_6m:+.2f}")
            cols[3].metric("1-Year", f"${price_1y:,.2f}", f"{delta_1y:+.2f}")
            st.divider()

            # --- ROW 2: TREND PROJECTION ---
            st.markdown("#### Trend Projection (Prophet)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
            fig.add_trace(go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'], name='Forecast'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"**Prophet Summary:** Based on history since 2020, the model projects a 1-year target of **${price_1y:,.2f}**.")
            st.divider()

           # --- ROW 3: PATTERN PREDICTOR (ML) ---
            st.markdown("#### Pattern Predictor (ML)")
            ml_df = prophet_df.copy()
            days_ahead = 252 # We target a 1-year horizon (252 trading days in a year)
            
            # 1. TECHNICAL INDICATORS (Momentum & Volatility)
            # SMA_20: Captures the 'trend direction' over the last month.
            ml_df['SMA_20'] = ml_df['y'].rolling(window=20, min_periods=1).mean()
            
            # RSI (Relative Strength Index): Detects overbought/oversold conditions (0-100 scale).
            delta = ml_df['y'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 0.001)
            ml_df['RSI'] = 100 - (100 / (1 + rs))
            
            # ATR (Average True Range): Measures 'market volatility'—how much the stock swings daily.
            # We reindex to ensure dates align perfectly with the price history.
            ml_df['ATR'] = (df['High'] - df['Low']).reindex(ml_df.index).rolling(window=14, min_periods=1).mean()
            
            # 2. MARKET SIGNAL FEATURES (Historical Momentum)
            # Volume: Investors trade more heavily on conviction; volume spikes often precede price moves.
            ml_df['Volume'] = df['Volume'].reindex(ml_df.index)
            
            # Log_Return: Normalizes price changes. Log returns are statistically 'stationary', 
            # meaning they are better for ML models to learn from than raw dollar prices.
            ml_df['Log_Return'] = np.log(ml_df['y'] / ml_df['y'].shift(1))

            # 3. DATA CLEANING
            # Fill gaps caused by rolling windows or missing dates with the nearest available data.
            ml_df = ml_df.ffill().bfill()
            
            # Define our input features (X). These are the 'clues' the model uses to guess the future.
            features = ['SMA_20', 'RSI', 'ATR', 'Volume', 'Log_Return']
            ml_df[features] = ml_df[features].fillna(0)

            # 4. TARGET CALCULATION (The 'Label')
            # We want to predict the % return 1 year (252 days) from now.
            # .shift(-days_ahead) 'looks into the future' to label our historical data 
            # with what actually happened later.
            ml_df['Target_Return'] = ml_df['y'].pct_change(days_ahead).shift(-days_ahead)
            ml_df = ml_df.dropna() # Drop rows where we don't have a future target

            if len(ml_df) > days_ahead:
                X = ml_df[features]
                y = ml_df['Target_Return']
                
                # 5. MODEL TRAINING
                # Random Forest: An ensemble of decision trees. It captures non-linear relationships 
                # (e.g., if RSI > 70 AND Volume is high, then price usually drops).
                # max_depth=10 prevents the model from just 'memorizing' data (overfitting).
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X, y)
                
                # 6. PREDICTION
                # We predict the future return using the most recent data (the last row).
                predicted_return = model.predict(ml_df[features].iloc[[-1]])[0]
                pred = current_price * (1 + predicted_return)
                
               # 7. MODEL EXPLAINABILITY & CLEAN PLOTTING
                importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                
                # Filter: Only keep features that actually contributed (Importance > 0)
                active_importances = importances[importances['Importance'] > 0].sort_values(by='Importance', ascending=False)
                
                if not active_importances.empty:
                    st.write("### Model Insight: What drives this prediction?")
                    # This chart now only shows columns that have signal
                    st.bar_chart(active_importances.set_index('Feature'))
                else:
                    # Fallback if the model finds absolutely no signal
                    st.info("The model could not identify significant patterns in the current features.")
            
            st.divider()

            # --- ROW 4: PROBABILISTIC PROJECTION (MONTE CARLO) ---
            st.markdown("#### Probabilistic Projection: Monte Carlo (10,000 Paths)")
            
            def run_mc(prices, days=252, n_sims=10000):
                returns = prices.pct_change().dropna()
                mu = returns.mean()
                sigma = returns.std()
                daily_returns = np.random.normal(mu, sigma, (days, n_sims))
                paths = prices.iloc[-1] * (1 + daily_returns).cumprod(axis=0)
                return paths

            # Using full df to capture long-term historical volatility for the MC simulation
            paths = run_mc(df[target_col])
            
            fig_mc = go.Figure()
            for i in range(100): 
                fig_mc.add_trace(go.Scatter(y=paths[:, i], line=dict(width=0.5, color='gray'), showlegend=False))
            median_path = np.median(paths, axis=1)
            fig_mc.add_trace(go.Scatter(y=median_path, line=dict(width=3, color='blue'), name='Median Path'))
            fig_mc.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_mc, use_container_width=True)
            
            st.warning(f"**Monte Carlo Summary:** Across 10,000 simulations, the **Median Projected Price** is **${np.median(paths[-1, :]):,.2f}**.")
            st.write(f"**Confidence Interval (95%):** Between **${np.percentile(paths[-1, :], 2.5):,.2f}** and **${np.percentile(paths[-1, :], 97.5):,.2f}**.")

            # --- ROW 5: FINAL CONSENSUS ---
            st.divider()
            st.header("AI Consensus Forecast")
            prophet_trend = "Bullish" if price_1y > current_price else "Bearish"
            ml_trend = "Bullish" if pred > current_price else "Bearish"
            mc_trend = "Bullish" if np.median(paths[-1, :]) > current_price else "Bearish"
            bullish_count = [prophet_trend, ml_trend, mc_trend].count("Bullish")
            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if bullish_count >= 2:
                    st.metric("Consensus", "Bullish 🐂", delta="Strong Buy")
                else:
                    st.metric("Consensus", "Bearish 🐻", delta="Caution")
            with col_b:
                st.write(f"Based on the integration of all models, the consensus is **{bullish_count}/3 indicators favoring a bullish outlook**.")
            
            st.caption("Disclaimer: This consensus is generated by AI models based on historical data. Perform your own due diligence.")

    except Exception as e:
        st.error(f"Error: {e}")
