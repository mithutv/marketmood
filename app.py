import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- CONFIGURATION ---
st.set_page_config(page_title="QuantLens Terminal", layout="wide")

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #007BFF; color: white; border-radius: 4px; border: none; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("QuantLens Financial Terminal")
st.markdown("Institutional-grade trend analysis and projection.")

@st.cache_data(ttl=86400)
def get_stock_data(ticker):
    return yf.download(ticker, threads=False, multi_level_index=False)

# --- USER INPUT ---
ticker = st.text_input("Enter Ticker Symbol:", value="NVDA").upper()

if st.button("Generate Forecast"):
    try:
        df = get_stock_data(ticker)
        ticker_obj = yf.Ticker(ticker)
        
        if df.empty:
            st.error("Data not found for the specified ticker.")
        else:
            # 1. --- METRIC SUMMARY ---
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            
            latest_price = prophet_df['y'].iloc[-1]
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - latest_price
            
            st.subheader(f"Analysis: {ticker}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${latest_price:,.2f}")
            col2.metric("30-Day Forecast", f"${forecasted_price:,.2f}", f"{delta:+.2f}")
            col3.metric("Trend Status", "Bullish" if delta > 0 else "Bearish")

            # 2. --- FORECAST VISUALIZATION ---
            st.subheader("Price Projection")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='Uncertainty'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#007BFF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # 3. --- CONTEXTUAL DATA ---
            tab1, tab2, tab3 = st.tabs(["📑 News", "⚙️ Fundamentals", "🌍 Market Context"])
            
            with tab1:
                news = getattr(ticker_obj, 'news', [])
                if news:
                    for item in news[:3]:
                        st.markdown(f"**{item.get('title')}**")
                        st.caption(f"Source: {item.get('publisher')} | [Read More]({item.get('link')})")
                else:
                    st.write("No news data currently available.")

            with tab2:
                info = ticker_obj.info
                st.write("### Financial Metrics")
                f_col1, f_col2 = st.columns(2)
                f_col1.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                f_col2.write(f"**Market Cap:** {info.get('marketCap', 0) / 1e9:.2f}B")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 0) * 100:.2f}%")

            with tab3:
                st.write("### S&P 500 Market Pulse")
                spy_news = yf.Ticker("SPY").news
                for item in spy_news[:2]:
                    st.markdown(f"- [{item.get('title')}]({item.get('link')})")

            # 4. --- HISTORICAL DATA ---
            with st.expander("View Historical Pricing Table"):
                display_df = prophet_df.sort_values(by='ds', ascending=False)
                st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"System Error: {e}")
