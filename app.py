import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox # Make sure this is installed

# --- GLOBAL STYLES ---
st.set_page_config(page_title="QuantLens", layout="wide")
st.markdown("""
    <style>
    [data-testid="stDataFrame"] thead tr th { background-color: #000000; color: #FFFFFF; font-weight: 800; font-size: 20px; text-align: center !important; }
    div.stButton > button:first-child { background-color: #007BFF; color: white; border: none; padding: 10px 24px; font-size: 16px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("QuantLens: AI-Driven Financial Forecasting")

# --- SEARCH BOX LOGIC ---
def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

selected_ticker = st_searchbox(
    search_tickers,
    placeholder="Start typing a company name (e.g., Costco)...",
    label="Search for a Company"
)

# Use the selected ticker if available, otherwise default to NVDA
ticker = selected_ticker if selected_ticker else "NVDA"
st.write(f"### Selected Ticker: {ticker}")

# --- FORECAST LOGIC ---
@st.cache_data(ttl=86400)
def get_stock_data(t):
    return yf.download(t, threads=False, multi_level_index=False)

if st.button("Generate Forecast"):
    ticker_obj = yf.Ticker(ticker)
    try:
        df = get_stock_data(ticker)
        
        if df.empty:
            st.error("No data found for this ticker.")
        else:
            # Data Prep (Downsampled for speed)
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).tail(365)
            
            # Prophet Engine
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            
            # Metrics
            latest_price = prophet_df['y'].iloc[-1]
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - latest_price
            
            st.metric(label="Forecast Price (30 Days)", value=f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # News
            st.markdown("### Recent Market News")
            news = getattr(ticker_obj, 'news', [])
            valid_news = [item for item in news if item.get('title')]
            if valid_news:
                for item in valid_news[:3]:
                    st.markdown(f"**{item.get('title')}**")
                    st.caption(f"Source: {item.get('publisher')} | [Read More]({item.get('link')})")
            else:
                st.info("No recent news headlines available.")
                
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
