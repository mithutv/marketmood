import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- CONFIGURATION & STYLES ---
st.set_page_config(page_title="QuantLens Terminal", layout="centered")
st.markdown("""
    <style>
    [data-testid="stDataFrame"] thead tr th { background-color: #000000; color: #FFFFFF; font-weight: 800; font-size: 20px; text-align: center !important; }
    div.stButton > button:first-child { background-color: #007BFF; color: white; border: none; padding: 10px 24px; font-size: 16px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("QuantLens: AI-Driven Financial Forecasting")

# --- TICKER LOOKUP LOGIC ---
st.subheader("1. Find Ticker")
search_query = st.text_input("Enter Company Name (e.g., Costco, Nvidia):")

# Initialize ticker in session state if not present
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = "NVDA"

if search_query:
    results = yf.Search(search_query)
    quotes = results.quotes
    if quotes:
        # Create a dropdown for selection
        ticker_options = {q['symbol']: f"{q['shortname']} ({q['symbol']})" for q in quotes if 'symbol' in q}
        selected = st.selectbox("Select the correct ticker:", options=list(ticker_options.keys()), format_func=lambda x: ticker_options[x])
        if st.button("Confirm Selection"):
            st.session_state['ticker'] = selected
            st.success(f"Ticker set to: {st.session_state['ticker']}")
    else:
        st.warning("No tickers found. Try a different name.")

# --- FORECAST LOGIC ---
st.subheader(f"2. Forecast for: {st.session_state['ticker']}")
ticker = st.session_state['ticker']

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
            # 365-day downsample to save CPU
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).tail(365)
            
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            
            latest_price = prophet_df['y'].iloc[-1]
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - latest_price
            
            st.metric("Forecast Price (30 Days)", f"${forecasted_price:,.2f}", f"{delta:+.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # News logic
            st.markdown("### Recent Market News")
            news = getattr(ticker_obj, 'news', [])
            valid_news = [item for item in news if item.get('title')]
            if valid_news:
                for item in valid_news[:3]:
                    st.markdown(f"**{item.get('title')}**")
                    st.caption(f"Source: {item.get('publisher')} | [Read More]({item.get('link')})")
            else:
                st.info("No news headlines currently available.")
                
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
