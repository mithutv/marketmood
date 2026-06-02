import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import nltk
from textblob import TextBlob

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
This application leverages **Meta's Prophet**, an additive time-series forecasting model, to analyze historical price trends. 
It accounts for seasonality—capturing daily, weekly, and yearly patterns—while providing a 30-day projection 
complete with uncertainty intervals to help visualize potential market volatility.
""")

# --- SEARCH BOX ---
def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(search_tickers, placeholder="Start typing a company name...", label="Search for a Company")
if ticker: st.write(f"### Selected Ticker: {ticker}")

@st.cache_data(ttl=86400)
def get_stock_data(ticker): return yf.download(ticker, threads=False, multi_level_index=False)

# --- FORECAST LOGIC ---
if st.button("Generate Forecast") and ticker:
    try:
        df = get_stock_data(ticker)
        if df.empty:
            st.error("No data found.")
        else:
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'}).dropna()
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            current_price = prophet_df['y'].iloc[-1]
            
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - current_price
            growth_pct = ((forecasted_price - current_price) / current_price) * 100
            trend_emoji = "📈 (Bullish)" if forecasted_price > current_price else "📉 (Bearish)"
            
            # Sentiment Calculation
            news_items = yf.Search(ticker).news
            valid_news = [item for item in news_items if item.get('title')]
            avg_sentiment = sum([TextBlob(i['title']).sentiment.polarity for i in valid_news[:3]]) / len(valid_news[:3]) if valid_news else 0
            
            if avg_sentiment > 0.1: gauge_color, status_label = "#4CAF50", "🟢 Bullish"
            elif avg_sentiment < -0.1: gauge_color, status_label = "#F44336", "🔴 Bearish"
            else: gauge_color, status_label = "#9E9E9E", "⚪ Neutral"

            # 1. Row of 3 Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:,.2f}")
            col2.metric("Forecast (30 Days)", f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
            
            # Gauge Meter in Col 3
            with col3:
                st.markdown(f"""
                <div style="text-align:center;">
                    <div style="font-size:0.8rem; font-weight:bold;">Sentiment Gauge</div>
                    <div style="background: conic-gradient(from 270deg, #F44336 0deg, #E0E0E0 90deg, #4CAF50 180deg); width: 80px; height: 40px; border-radius: 40px 40px 0 0; margin: 5px auto;"></div>
                    <div style="font-weight:bold; color:{gauge_color};">{status_label}</div>
                    <div style="font-size:0.8rem;">Score: {avg_sentiment:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 2. Graph & Summary
            st.subheader(f"Prediction Trend: {trend_emoji}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"The model suggests a {trend_emoji} trend. Projected price: **${forecasted_price:,.2f}** ({delta:+.2f} | {growth_pct:+.2f}%).")
            
            # 3. Collapsible Table
            with st.expander("View Historical Data"):
                st.dataframe(prophet_df.rename(columns={'ds': 'Date', 'y': 'Closing Price'}), use_container_width=True, hide_index=True)

            # 4. News
            st.markdown('<h3 style="margin-bottom: 0px;">Recent Market News</h3>', unsafe_allow_html=True)
            for item in valid_news[:3]:
                link = item.get('link') or item.get('clickThroughUrl') or "#"
                st.markdown(f"**{item.get('title')}**")
                st.caption(f"Source: {item.get('publisher')} | [Read More]({link})" if link != "#" else f"Source: {item.get('publisher')}")
    except Exception as e:
        st.error(f"Error: {e}")
