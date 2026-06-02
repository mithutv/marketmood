import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    /* Clean, professional header styling */
    [data-testid="stDataFrame"] thead tr th {
        background-color: #000000;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 20px;
        text-align: center !important;
    }
    [data-testid="stDataFrame"] thead tr th div {
        justify-content: center !important;
    }
    /* Blue Button Styling */
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
            
            # 5. UI Output: Prediction Header & Metric
            st.subheader(f"Prediction Trend: {trend_emoji}")
            st.metric(label="Forecast Price (30 Days)", value=f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
            
            # 6. Graph with Confidence Interval
            fig = go.Figure()

            # Uncertainty Interval (Shaded Area)
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='Confidence Interval'))

            # Forecast and Actual Lines
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # 7. Historical Table
            display_df = prophet_df.copy()
            display_df['ds'] = display_df['ds'].dt.strftime('%b %d, %Y')
            display_df.columns = ['Date', 'Closing Price']
            
            st.markdown(f"### Historical Data for {ticker}")
            st.dataframe(
                display_df, 
                use_container_width=True, 
                hide_index=True,
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="medium", alignment="center"),
                    "Closing Price": st.column_config.NumberColumn("Closing Price", format="$%.2f", alignment="center")
                }
            )


    # News Section
            st.markdown("### Recent Market News")
            news = getattr(ticker_obj, 'news', [])
            if news:
                for item in news[:3]:
                    st.markdown(f"**{item.get('title')}**")
                    st.caption(f"Source: {item.get('publisher')} | [Read More]({item.get('link')})")
            else:
                st.write("No news data available.")
            
            # Fundamentals
            st.markdown("### Financial Metrics")
            info = ticker_obj.info
            f1, f2 = st.columns(2)
            f1.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            f2.write(f"**Market Cap:** {info.get('marketCap', 0) / 1e9:.2f}B")
            
            # Market Pulse
            st.markdown("### S&P 500 Market Pulse")
            spy_news = yf.Ticker("SPY").news
            for item in spy_news[:2]:
                st.markdown(f"- [{item.get('title')}]({item.get('link')})")
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
