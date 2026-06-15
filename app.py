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

@st.cache_data(ttl=3600) # Cache search results for 1 hour
def search_tickers(searchterm: str):
    # CRITICAL: Only search if the user has typed at least 3 characters
    # This prevents the app from searching for "A", "AA", "AAP"...
    if not searchterm or len(searchterm) < 3: 
        return []
    
    # Add a tiny delay to avoid rapid-fire requests
    time.sleep(0.2) 
    
    try:
        results = yf.Search(searchterm).quotes
        return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) 
                for q in results if 'symbol' in q]
    except Exception:
        return []
# Page Config
st.set_page_config(page_title="Marketmood", layout="centered")

st.markdown("""
    <style>

    /* Targeting the Title specifically for an elegant, smaller look */
    h1 { 
        font-size: 38px !important; 
        font-weight: 500 !important; 
        color: #333333 !important; 
        margin-bottom: 4px !important; 
        letter-spacing: -0.5px;
    }
    
    /* Optional: Style the subtitle to be even smaller/muted */
    .stApp > header { display: none; } /* Hides the default Streamlit header */
    
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
st.subheader("Ensemble-based predictive analytics for the modern investor.")
st.caption("Advanced multi-model forecasting powered by Prophet, Random Forest, and Monte Carlo simulations.")

# --- SCOPE NOTE / ABOUT SECTION ---
with st.expander("🛈 About Marketmood"):
    
    st.markdown("### Summary")
    st.markdown("""
    Marketmood is an ensemble-driven AI forecasting suite that transforms complex machine learning models into actionable market context for investors.
    """)

    st.markdown("### Methodology")
    st.markdown("""
    Marketmood combines multiple AI models to generate robust forecasts:

    - **Meta’s Prophet** → Captures seasonality and trend structure  
    - **Random Forest** → Identifies nonlinear patterns in market behavior  
    - **Monte Carlo Simulation** → Models uncertainty and risk distributions  

    These models are aggregated into an ensemble system to reduce single-model bias and improve stability across different market conditions.

    **Data Integrity:**  
    We use verified historical market data from 2020 onward, emphasizing consistency and regime relevance in all projections.
    """)

    st.markdown("### Risk Disclosure")
    st.markdown("""
    Marketmood outputs are probabilistic forecasts, not guarantees.

    - This tool does not provide financial advice  
    - All projections are for informational and educational use only  
    - Markets are inherently uncertain and past performance does not imply future results  

    Use these insights as one input among broader independent research and due diligence.
    """)

def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(
    search_tickers,
    placeholder="AAPL, NVDA, TSLA...",
    label="Search market data"
)

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
            days_ahead = 252 
            ml_df['SMA_20'] = ml_df['y'].rolling(window=20, min_periods=1).mean()
            delta = ml_df['y'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 0.001)
            ml_df['RSI'] = 100 - (100 / (1 + rs))
            ml_df['ATR'] = (df['High'] - df['Low']).reindex(ml_df.index).rolling(window=14, min_periods=1).mean()
            ml_df['Volume'] = df['Volume'].reindex(ml_df.index)
            ml_df['Log_Return'] = np.log(ml_df['y'] / ml_df['y'].shift(1))
            ml_df = ml_df.ffill().bfill()
            features = ['SMA_20', 'RSI', 'ATR', 'Volume', 'Log_Return']
            ml_df[features] = ml_df[features].fillna(0)
            ml_df['Target_Return'] = ml_df['y'].pct_change(days_ahead).shift(-days_ahead)
            ml_df = ml_df.dropna() 

            if len(ml_df) > days_ahead:
                X = ml_df[features]
                y = ml_df['Target_Return']
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X, y)
                predicted_return = model.predict(ml_df[features].iloc[[-1]])[0]
                pred = current_price * (1 + predicted_return)
                importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                active_importances = importances[importances['Importance'] > 0].sort_values(by='Importance', ascending=False)
                if not active_importances.empty:
                    st.write("### Model Insight: What drives this prediction?")
                    st.bar_chart(active_importances.set_index('Feature'))
                    top_feature = active_importances.iloc[0]['Feature']
                    st.info(f"**ML Model Summary:** The model is relying most heavily on **{top_feature}** for its 1-year projection.")
            st.divider()
            
            # --- ROW 4: MONTE CARLO ---
            st.markdown("#### Probabilistic Projection: Monte Carlo (10,000 Paths)")
            def run_mc(prices, days=252, n_sims=10000):
                returns = prices.pct_change().dropna()
                mu = returns.mean()
                sigma = returns.std()
                daily_returns = np.random.normal(mu, sigma, (days, n_sims))
                paths = prices.iloc[-1] * (1 + daily_returns).cumprod(axis=0)
                return paths

            paths = run_mc(df[target_col])
            fig_mc = go.Figure()
            for i in range(100): 
                fig_mc.add_trace(go.Scatter(y=paths[:, i], line=dict(width=0.5, color='gray'), showlegend=False))
            median_path = np.median(paths, axis=1)
            fig_mc.add_trace(go.Scatter(y=median_path, line=dict(width=3, color='blue'), name='Median Path'))
            fig_mc.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_mc, use_container_width=True)

            # --- ROW 5: FINAL CONSENSUS & CONVICTION ---
            st.divider()
            st.header("AI Consensus Forecast")
            prophet_trend = "Bullish" if price_1y > current_price else "Bearish"
            ml_trend = "Bullish" if pred > current_price else "Bearish"
            mc_trend = "Bullish" if np.median(paths[-1, :]) > current_price else "Bearish"
            bullish_count = [prophet_trend, ml_trend, mc_trend].count("Bullish")
            is_bullish = bullish_count >= 2
            reasons = []
            if prophet_trend == "Bullish": reasons.append("Prophet identifies upward seasonal trends.")
            if ml_trend == "Bullish": reasons.append("Machine Learning pattern predictors signal growth.")
            if mc_trend == "Bullish": reasons.append("Monte Carlo simulations lean toward price appreciation.")
            conviction_score = int(min((bullish_count / 3) * 100 + 10, 100))
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Consensus", "Bullish 🐂" if is_bullish else "Bearish 🐻", f"{conviction_score}% Conviction")
            with col_b:
                st.progress(conviction_score / 100)
                st.write(f"**Why this result?**")
                for reason in reasons:
                    st.markdown(f"- {reason}")

            # --- NEWS SENTIMENT ---
            st.subheader("Market Sentiment (News Analysis)")
            with st.container(border=True):
                try:
                    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
                    feed = feedparser.parse(rss_url)
                    if feed.entries:
                        sentiments = [TextBlob(e.title).sentiment.polarity for e in feed.entries[:5]]
                        avg_polarity = sum(sentiments) / len(sentiments)
                        st.markdown(f"**Market Sentiment: ** {'Positive 😋' if avg_polarity > 0 else 'Neutral/Negative'}")
                        st.markdown("**Recent Headlines:**")
                        for entry in feed.entries[:5]: st.markdown(f"• {entry.title}")
                        sentiment_score = int(((avg_polarity + 1) / 2) * 100)
                        st.markdown(f"**Sentiment Score: {'+' if sentiment_score >= 50 else ''}{sentiment_score}%**")
                    else:
                        st.write("No recent news found.")
                except Exception:
                    st.write("Sentiment analysis currently unavailable.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
