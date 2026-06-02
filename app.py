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
This application analyzes the **last 4 years** of historical price action to provide a projection using **Meta's Prophet**. 
The dashboard provides an executive summary of current pricing, future forecasts, and real-time sentiment analysis.
""")

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

            # --- PROPHET FORECASTING & METRICS ---
            m = Prophet(daily_seasonality=True).fit(prophet_df)                
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
            col1.metric("**Current**", f"${current_price:,.2f}")
            col2.metric("**30-Day**", f"${price_30:,.2f}", get_delta_text(price_30, current_price))
            col3.metric("**6-Month**", f"${price_6m:,.2f}", get_delta_text(price_6m, current_price))
            col4.metric("**1-Year**", f"${price_1y:,.2f}", get_delta_text(price_1y, current_price))
            
            # Graph
            st.subheader("Price Projection (Prophet Model)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.add_trace(go.Scatter(x=forecast_1y['ds'], y=forecast_1y['yhat'], name='1-Year Forecast', line=dict(color='#0000FF', dash='dash')))
            fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        

            # Sentiment Analysis
            news_items = yf.Search(ticker).news
            valid_news = [item for item in news_items if item.get('title')]
            sentiment_scores = [TextBlob(i['title']).sentiment.polarity for i in valid_news[:3]] if valid_news else [0]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Gauge Logic
            normalized_sentiment = (avg_sentiment + 1) / 2
            if avg_sentiment > 0.1: status_label = "Bullish"
            elif avg_sentiment < -0.1: status_label = "Bearish"
            else: status_label = "Neutral"

            st.markdown("#### Market Sentiment Gauge")
            
            # Display score prominently
            st.metric(label="Sentiment Score", value=f"{avg_sentiment:.2f}", delta=status_label)
            
            # The Sentiment Meter (Progress Bar)
            st.progress(normalized_sentiment)
            
            # Add range labels
            col_left, col_right = st.columns(2)
            col_left.caption("Bearish (-1.0)")
            col_right.caption("Bullish (+1.0)")

            # Quarterly Historical Table
            with st.expander("View Quarterly Historical Summary"):
                summary_df = prophet_df.copy()
                summary_df['Year'] = summary_df['ds'].dt.year
                summary_df['Quarter'] = summary_df['ds'].dt.quarter
                quarterly = summary_df.groupby(['Year', 'Quarter'])['y'].agg(['mean', 'max', 'min']).reset_index()
                quarterly = quarterly.sort_values(by=['Year', 'Quarter'], ascending=[False, False])
                quarterly['Period'] = 'Q' + quarterly['Quarter'].astype(str) + ' ' + quarterly['Year'].astype(str)
                quarterly = quarterly[['Period', 'mean', 'max', 'min']]
                quarterly.columns = ['Period', 'Avg Price', 'High', 'Low']
                for col in ['Avg Price', 'High', 'Low']:
                    quarterly[col] = quarterly[col].map('${:,.2f}'.format)
                st.dataframe(quarterly, use_container_width=True, hide_index=True)

            # News
            st.write('<h3 style="margin-bottom: 0px;">Recent Market News</h3>', unsafe_allow_html=True)
            for item in valid_news[:3]:
                link = item.get('link') or item.get('clickThroughUrl') or "#"
                st.markdown(f"**{item.get('title')}**")
                st.caption(f"Source: {item.get('publisher')} | [Read More]({link})")

           # --- MONTE CARLO SIMULATION SECTION ---
            st.header("Probabilistic Projection: Monte Carlo Simulation")
            with st.expander("View 1000-Day Monte Carlo Simulation Details"):
                st.write("""
                **Understanding This Chart:**
                * **The Gray Cloud:** Represents 500 different possible price paths based on the stock's historical volatility. 
                * **The Blue Line (Median):** The 50th percentile outcome.
                * **1000-Day Outlook:** This projects the potential price range over the next ~4 years.
                """)
                
                # Simulation Function (1000 Days)
                def run_monte_carlo(prices, days=1000, sims=500):
                    returns = prices.pct_change().dropna()
                    mu, sigma = returns.mean(), returns.std()
                    daily_returns = np.random.normal(mu, sigma, (days, sims))
                    return prices.iloc[-1] * (1 + daily_returns).cumprod(axis=0)

                price_paths = run_monte_carlo(prophet_df['y'])
                median_path = np.median(price_paths, axis=1)
                
                # Plotting
                fig_mc = go.Figure()
                for i in range(50): 
                    fig_mc.add_trace(go.Scatter(x=list(range(1000)), y=price_paths[:, i], 
                                                 line=dict(color='lightgray', width=1), showlegend=False))
                
                fig_mc.add_trace(go.Scatter(x=list(range(1000)), y=median_path, 
                                             line=dict(color='blue', width=3), name='Median (50%) Path'))
                
                fig_mc.update_layout(template="plotly_white", xaxis_title="Trading Days", yaxis_title="Projected Price")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Data Readout
                median_final_price = median_path[-1]
                st.metric("Median Projected Price (1000 Days)", f"${median_final_price:,.2f}")
                st.caption(f"At day 1000, the median projected price is ${median_final_price:,.2f}. Note: This is a probabilistic estimate based on {len(prophet_df)} days of historical data.")

    
    except Exception as e:
        st.error(f"Error: {e}")
