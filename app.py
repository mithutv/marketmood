import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import nltk

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

ticker = selected_ticker  # It will be None until the user searches
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
        ticker_obj = yf.Ticker(ticker)
        
        if df.empty:
            st.error("No data found for this ticker.")
        else:
            # 2. Data Preparation
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
            
            # Clean the data
            prophet_df = prophet_df.dropna()
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Use the most recent price from our data to avoid API rate limits
            current_price = prophet_df['y'].iloc[-1]
            
            # 3. Prophet Engine
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            
            # 4. Metrics Calculation
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - current_price
            trend_emoji = "📈 (Bullish)" if forecasted_price > current_price else "📉 (Bearish)"
            
            # 5. UI Output
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"${current_price:,.2f}")
            col2.metric("Forecast Price (30 Days)", f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
            
            st.subheader(f"Prediction Trend: {trend_emoji}")
            
            # 6. Graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='Confidence Interval'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- SUMMARY ---
            st.markdown("### Forecast Summary")
            growth_pct = ((forecasted_price - current_price) / current_price) * 100
            summary_text = f"""
            Based on the analysis of historical price patterns, the model suggests a {trend_emoji} trend. 
            The projected price 30 days from now is **${forecasted_price:,.2f}**, which represents 
            a movement of **{delta:+.2f}** ({growth_pct:+.2f}%) from the current price of **${current_price:,.2f}**.
            """
            st.info(summary_text)
            
            # 7. Historical Table
            display_df = prophet_df.copy()
            display_df['ds'] = display_df['ds'].dt.strftime('%b %d, %Y')
            display_df.columns = ['Date', 'Closing Price']
            st.markdown(f"### Historical Data for {ticker}")
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)


            # 8. News Section & Sentiment Analysis
            st.markdown("### Recent Market News")
search = yf.Search(ticker)
news = search.news

if news:
    # Process the news list (it is usually a list of dictionaries)
    for item in news[:3]:
        st.markdown(f"**{item.get('title')}**")
        st.caption(f"Source: {item.get('publisher')}")
else:
    st.info("No news found via Search API. Trying Ticker object...")
                
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
