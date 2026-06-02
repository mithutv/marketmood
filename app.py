import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import nltk
from textblob import TextBlob
import numpy as np  # Required for Monte Carlo simulation calculations

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
st.markdown("""
This application analyzes the **last 4 years** of historical price action to provide a 30-day projection using **Meta's Prophet**. 
The dashboard provides an executive summary of current pricing, future forecasts, and real-time sentiment analysis.
""")

# --- SEARCH BOX ---
def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(search_tickers, placeholder="Start typing a company name...", label="Search for a Company")
if ticker: st.write(f"### Selected Ticker: {ticker}")

# Fetch 5 years to guarantee the 4-year filter works perfectly
@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="5y", threads=False, multi_level_index=False)

# --- FORECAST LOGIC ---
if st.button("Generate Forecast") and ticker:
    try:
        df = get_stock_data(ticker)
        if df.empty:
            st.error("No data found.")
        else:
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
            
            # Apply 4-Year Filter
            four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df = prophet_df[prophet_df['ds'] >= four_years_ago].dropna()
            current_price = prophet_df['y'].iloc[-1]
            
            # Prophet Engine
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - current_price
            growth_pct = ((forecasted_price - current_price) / current_price) * 100
            trend_emoji = "📈 (Bullish)" if forecasted_price > current_price else "📉 (Bearish)"
            
            # Sentiment
            news_items = yf.Search(ticker).news
            valid_news = [item for item in news_items if item.get('title')]
            
            if valid_news:
                sentiment_scores = [TextBlob(i['title']).sentiment.polarity for i in valid_news[:3]]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            else:
                avg_sentiment = 0
            
            if avg_sentiment > 0.1: 
                gauge_color, status_label = "#4CAF50", "🟢 Bullish"
            elif avg_sentiment < -0.1: 
                gauge_color, status_label = "#F44336", "🔴 Bearish"
            else: 
                gauge_color, status_label = "#9E9E9E", "⚪ Neutral"

            # Metrics Row
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:,.2f}")
            col2.metric("Forecast (30 Days)", f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
            with col3:
                st.markdown(f"""
                <div style="text-align:center;">
                    <div style="font-size:0.8rem; font-weight:bold;">Sentiment Gauge</div>
                    <div style="background: conic-gradient(from 270deg, #F44336 0deg, #E0E0E0 90deg, #4CAF50 180deg); width: 80px; height: 40px; border-radius: 40px 40px 0 0; margin: 5px auto;"></div>
                    <div style="font-weight:bold; color:{gauge_color};">{status_label}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Graph
            st.subheader(f"Prediction Trend: {trend_emoji}")
            plot_df = prophet_df.sort_values('ds')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['y'], name='Actual', line=dict(color='#000000')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF', dash='dash')))
            fig.update_layout(title=f"4-Year Price History & 30-Day Forecast", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"The model suggests a {trend_emoji} trend. Projected price: **${forecasted_price:,.2f}** ({delta:+.2f} | {growth_pct:+.2f}%).")
            
            # Quarterly Historical Table
            with st.expander("View Quarterly Historical Summary"):
                summary_df = prophet_df.copy()
                summary_df['Year'] = summary_df['ds'].dt.year
                summary_df['Quarter'] = summary_df['ds'].dt.quarter
                
                # Group by Year and Quarter
                quarterly = summary_df.groupby(['Year', 'Quarter'])['y'].agg(['mean', 'max', 'min']).reset_index()
                quarterly = quarterly.sort_values(by=['Year', 'Quarter'], ascending=[False, False])
                
                # Format a nice 'Period' column (e.g., Q1 2026)
                quarterly['Period'] = 'Q' + quarterly['Quarter'].astype(str) + ' ' + quarterly['Year'].astype(str)
                quarterly = quarterly[['Period', 'mean', 'max', 'min']]
                quarterly.columns = ['Period', 'Avg Price', 'High', 'Low']
                
                # Fixed the list logic here to prevent syntax errors on copy-paste
                cols_to_format = ['Avg Price', 'High', 'Low']
                for col in cols_to_format:
                    quarterly[col] = quarterly[col].map('${:,.2f}'.format)
                    
                st.dataframe(quarterly, use_container_width=True, hide_index=True)

            # News
            st.write('<h3 style="margin-bottom: 0px;">Recent Market News</h3>', unsafe_allow_html=True)
            for item in valid_news[:3]:
                link = item.get('link') or item.get('clickThroughUrl') or "#"
                st.markdown(f"**{item.get('title')}**")
                st.caption(f"Source: {item.get('publisher')} | [Read More]({link})" if link != "#" else f"Source: {item.get('publisher')}")


             # --- MONTE CARLO SIMULATION SECTION ---
                st.header("Probabilistic Projection: Monte Carlo Simulation")
                with st.expander("View 1-Year Monte Carlo Simulation Details"):
                    st.write("""
                    **Understanding This Chart:**
                    * **The Gray Cloud:** Represents 500 different possible price paths based on the stock's historical volatility. The wider the cloud, the more volatile (risky) the stock has been historically.
                    * **The Blue Line (Median):** The 50th percentile outcome. Mathematically, 50% of the simulations ended above this line, and 50% ended below it.
                    * **Risk Assessment:** This tool does not predict the future; it shows the **range of probability**.
                    """)
                    
                    # Simulation Function
                    def run_monte_carlo(prices, days=252, sims=500):
                        returns = prices.pct_change().dropna()
                        mu, sigma = returns.mean(), returns.std()
                        daily_returns = np.random.normal(mu, sigma, (days, sims))
                        price_paths = prices.iloc[-1] * (1 + daily_returns).cumprod(axis=0)
                        return price_paths

                    # Execute
                    price_paths = run_monte_carlo(prophet_df['y'])
                    median_path = np.median(price_paths, axis=1)
                    
                    # Plotting
                    fig_mc = go.Figure()
                    for i in range(50): # Plot 50 paths for visual clarity
                        fig_mc.add_trace(go.Scatter(x=list(range(252)), y=price_paths[:, i], 
                                                     line=dict(color='lightgray', width=1), showlegend=False))
                    
                    fig_mc.add_trace(go.Scatter(x=list(range(252)), y=median_path, 
                                                 line=dict(color='blue', width=3), name='Median (50%) Path'))
                    
                    fig_mc.update_layout(template="plotly_white", xaxis_title="Trading Days", yaxis_title="Projected Price")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    st.caption("Each gray line is a 'random walk' based on past performance. The blue line is the midpoint of all those possibilities.")
    
    except Exception as e:
        st.error(f"Error: {e}")
