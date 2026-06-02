import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet


# --- GLOBAL STYLES (Blue Button, Table Headers, Alignment) ---
st.markdown("""
    <style>
    /* Clean, simple header styling */
    [data-testid="stDataFrame"] thead tr th {
        background-color: #000000;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 16px;
    }

    /* Center align header text */
  [data-testid="stDataFrame"] thead tr th {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
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

# --- HEADER & SCOPE NOTE ---
st.title("QuantLens: AI-Driven Financial Forecasting")
st.markdown("""
    <div style="font-size: 20px; color: #333; margin-bottom: 20px;">
        This application utilizes Meta's Prophet time-series library to 
        analyze historical stock price trends and generate a 30-day predictive forecast for retail investors.
    </div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=86400)
def get_stock_data(ticker):
    return yf.download(ticker, threads=False, multi_level_index=False)

# --- USER INPUT ---
st.markdown('<div style="font-size: 20px; font-weight: bold;">Analyze a new stock:</div>', unsafe_allow_html=True)
ticker = st.text_input(label="Hidden", label_visibility="collapsed", placeholder="e.g., NVDA, AAPL", value="NVDA").upper()

# --- FORECAST LOGIC ---
if st.button("Generate Forecast"):
    try:
        df = get_stock_data(ticker)
        if df.empty:
            st.error("No data found.")
        else:
            # Data Prep
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            display_df = prophet_df.copy()
            display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Date', 'Closing Price']
            # --- PREPARE DATA FOR DISPLAY ---
            display_df = prophet_df.copy()
            # Format date as 'May 26, 2026'
            display_df['ds'] = display_df['ds'].dt.strftime('%b %d, %Y')
            display_df.columns = ['Date', 'Closing Price']
            
            # Show Table Header (Styled)
            st.markdown(f"""
                <div style="font-size: 24px; font-weight: bold; color: #00008B; margin-top: 20px; margin-bottom: 10px;">
                    Historical Data for {ticker}
                </div>
            """, unsafe_allow_html=True)
            
            # Show Table (Clean and center-aligned)
            st.dataframe(
                display_df, 
                use_container_width=True, 
                hide_index=True,
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn(
                        "Date", 
                        width="medium",
                        alignment="center"
                    ),
                    "Closing Price": st.column_config.NumberColumn(
                        "Closing Price", 
                        format="$%.2f",
                        alignment="center"
                    )
                }
            )
            # Table Header
            st.markdown(f'<div style="font-size: 20px; font-weight: bold; color: #00008B; margin-top: 20px;">Historical Data for {ticker}</div>', unsafe_allow_html=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Prophet Engine
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            
            # Graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF')))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual', line=dict(color='#000000')))
            fig.update_layout(title=f"Price Forecast for {ticker}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Graph Description
            st.markdown("*The graph above displays historical daily closing prices (black) against the AI-generated 30-day forecast trend (blue).*")
            
            # Summary
            st.markdown(f"### Forecast Summary: {ticker}")
            st.write(f"The model estimates the price will move to **${forecast['yhat'].iloc[-1]:.2f}** over the next 30 days based on historical volatility.")
            
    except Exception as e:
        st.error(f"Error: {e}")
