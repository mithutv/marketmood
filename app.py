import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- GLOBAL STYLES ---
st.set_page_config(page_title="QuantLens", layout="centered")
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
st.title("QuantLens: AI-Driven Financial Forecasting")
st.markdown("This application utilizes Meta's Prophet library to analyze trends and generate a 30-day predictive forecast.")

@st.cache_data(ttl=86400)
def get_stock_data(ticker):
    return yf.download(ticker, threads=False, multi_level_index=False)

# --- USER INPUT ---
ticker = st.text_input("Analyze a new stock (e.g., NVDA, AAPL):", value="NVDA").upper()

# --- FORECAST LOGIC ---
if st.button("Generate Forecast"):
    ticker_obj = yf.Ticker(ticker)
    try:
        # 1. Fetch Data
        df = get_stock_data(ticker)
        
        if df.empty:
            st.error("No data found for this ticker.")
        else:
            # 2. Data Preparation
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # 3. Prophet Engine
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            
            # 4. Metrics Calculation
            latest_price = prophet_df['y'].iloc[-1]
            forecasted_price = forecast['yhat'].iloc[-1]
            delta = forecasted_price - latest_price
            trend_emoji = "📈 (Bullish)" if forecasted_price > latest_price else "📉 (Bearish)"
            
            # 5. UI Output
            st.subheader(f"Prediction Trend: {trend_emoji}")
            st.metric(label="Forecast Price (30 Days)", value=f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
            
            # 6. Graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='Confidence Interval'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # 7. Historical Table
            display_df = prophet_df.copy()
            display_df['ds'] = display_df['ds'].dt.strftime('%b %d, %Y')
            display_df.columns = ['Date', 'Closing Price']
            st.markdown(f"### Historical Data for {ticker}")
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
            
            # 8. News Section
            st.markdown("### Recent Market News")
            news = getattr(ticker_obj, 'news', [])
            valid_news = [item for item in news if item.get('title')]
            
            if valid_news:
                for item in valid_news[:3]:
                    st.markdown(f"**{item.get('title')}**")
                    st.caption(f"Source: {item.get('publisher')} | [Read More]({item.get('link')})")
            else:
                st.info("No recent news headlines are currently available for this ticker.")
                
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
