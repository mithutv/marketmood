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
    [data-testid="stDataFrame"] thead tr th {
        background-color: #000000;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 20px;
        text-align: center !important;
    }
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Market Mood: AI-Driven Financial Forecasting")
st.markdown("""
This application leverages **Meta's Prophet**, an additive time-series forecasting model, to analyze historical price trends. 
It accounts for seasonality—capturing daily, weekly, and yearly patterns—while providing a 30-day projection 
complete with uncertainty intervals to help visualize potential market volatility.
""")

# --- SEARCH BOX LOGIC ---
def search_tickers(searchterm: str):
    if not searchterm:
        return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

selected_ticker = st_searchbox(
    search_tickers,
    placeholder="Start typing a company name (e.g., Costco)...",
    label="Search for a Company"
)

ticker = selected_ticker
if ticker:
    st.write(f"### Selected Ticker: {ticker}")

@st.cache_data(ttl=86400)
def get_stock_data(ticker):
    return yf.download(ticker, threads=False, multi_level_index=False)

# --- FORECAST LOGIC ---
if st.button("Generate Forecast"):
    try:
        # 1. Fetch Data
        df = get_stock_data(ticker)
        
        if df.empty:
            st.error("No data found for this ticker.")
        else:
            # 2. Data Preparation
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
            
            # --- Filter for last 4 years ---
            four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df = prophet_df[prophet_df['ds'] >= four_years_ago]
            # -------------------------------
            
            prophet_df = prophet_df.dropna()
            current_price = prophet_df['y'].iloc[-1]
            
            # 3. Prophet Engine
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            
            # 4. Metrics & Sentiment Calculation
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - current_price
            growth_pct = ((forecasted_price - current_price) / current_price) * 100
            trend_emoji = "📈 (Bullish)" if forecasted_price > current_price else "📉 (Bearish)"
            
            search = yf.Search(ticker)
            news_items = search.news 
            valid_news = [item for item in news_items if item.get('title')]
            
            avg_sentiment = 0.0
            if valid_news:
                sentiment_scores = [TextBlob(item.get('title')).sentiment.polarity for item in valid_news[:3]]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            if avg_sentiment > 0.1:
                status_label = "🟢 Bullish"
            elif avg_sentiment < -0.1:
                status_label = "🔴 Bearish"
            else:
                status_label = "⚪ Neutral"

            # 5. UI Output - REORGANIZED
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:,.2f}")
            col2.metric("Forecast Price (30 Days)", f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
            col3.metric("**Aggregate Sentiment**", f"{avg_sentiment:.2f}", status_label)
            
            st.subheader(f"Prediction Trend: {trend_emoji}")
            
            # Graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='Confidence Interval'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.info(f"Based on the analysis of historical price patterns, the model suggests a {trend_emoji} trend. The projected price 30 days from now is **${forecasted_price:,.2f}**, which represents a movement of **{delta:+.2f}** ({growth_pct:+.2f}%) from the current price of **${current_price:,.2f}**.")
            
            # Collapsible Table
            with st.expander("View Historical Data"):
                display_df = prophet_df.copy()
                display_df['ds'] = display_df['ds'].dt.strftime('%b %d, %Y')
                display_df.columns = ['Date', 'Closing Price']
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # News Section
            st.markdown("### Recent Market News")
            if valid_news:
                for item in valid_news[:3]:
                    link = item.get('link') or item.get('clickThroughUrl') or "#"
                    st.markdown(f"**{item.get('title')}**")
                    st.caption(f"Source: {item.get('publisher')} | [Read More]({link})" if link != "#" else f"Source: {item.get('publisher')}")
            else:
                st.info("No recent news headlines available.")

    except Exception as e:
        st.error(f"Error generating forecast: {e}")
