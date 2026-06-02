import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- CONFIG & CACHE ---
st.set_page_config(page_title="MarketMood: Sentiment & Trend Analysis Tool", layout="wide")
st.title("QuantLens: AI-Driven Financial Forecasting")
st.caption("Precision predictive modeling for modern investors.")

@st.cache_data(ttl=86400)
def get_stock_data(ticker):
    df = yf.download(ticker, threads=False, multi_level_index=False)
    return df

# --- UI: BLUE BUTTON ---
# We use custom CSS to force the button to be blue
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("MarketMood AI")
ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL)", value="NVDA").upper()

if st.button("Generate Forecast"):
    try:
        df = get_stock_data(ticker)
        if df.empty:
            st.error("No data found.")
        else:
            # Prepare Data
            df_reset = df.reset_index()
            prophet_df = df_reset[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # --- TABLE: BORDERS, SORTABLE, LEFT ALIGN ---
            st.subheader(f"Historical Data for {ticker}")
            display_df = prophet_df.copy()
            display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Date', 'Closing Price']
            
            st.dataframe(
                display_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="medium"),
                    "Closing Price": st.column_config.NumberColumn("Closing Price", format="$%.2f")
                }
            )

            # --- FORECAST ENGINE ---
            m = Prophet(daily_seasonality=True)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            # --- GRAPH: LABELED ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='black')))
            fig.update_layout(title=f"Price Forecast for {ticker}", xaxis_title="Date", yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- SUMMARY PARAGRAPH ---
            latest_price = prophet_df['y'].iloc[-1]
            future_price = forecast['yhat'].iloc[-1]
            st.markdown(f"""
                ### Forecast Summary
                The model has analyzed historical data for **{ticker}**. 
                The current closing price is **${latest_price:.2f}**. 
                Based on current trends, the projected price in 30 days is estimated to be **${future_price:.2f}**. 
                *Note: This is an AI projection and should not be used as financial advice.*
            """)
            
    except Exception as e:
        st.error(f"Error: {e}")
