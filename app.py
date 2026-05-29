import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob

# 1. Page Configuration
st.set_page_config(page_title="MarketMood AI Alpha", page_icon="⚡", layout="wide")

st.title("⚡ MarketMood AI — Quantitative Analytics Engine")
st.markdown("Dynamic macro trend forecasting, risk-adjusted metrics, and multi-factor news sentiment analysis.")
st.markdown("---")

# 2. Sidebar for User Input
st.sidebar.header("Asset Selection")
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., NVDA, AAPL, MSFT)", value="NVDA").upper()

# --- OPTIMIZATION ENGINE: CACHED DATA FETCHING CORNERSTONES ---

@st.cache_data(ttl=3600)  # Caches stock historical dataframe for 1 hour
def fetch_historical_data(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1461)  # 4 years
    df = stock.history(start=start_date, end=end_date)
    company_name = stock.info.get('longName', ticker)
    return df, company_name

@st.cache_data(ttl=1800)  # Caches news streams separately for 30 minutes
def fetch_stock_news(ticker):
    stock = yf.Ticker(ticker)
    return stock.news

# 3. Main App Logic
if ticker_input:
    try:
        # Pull data safely from the local Streamlit cache instead of hammering Yahoo
        df, company_name = fetch_historical_data(ticker_input)
        
        if not df.empty:
            
            # --- QUANT ENGINE: RISK & MOMENTUM CALCULATIONS ---
            df['Daily_Return'] = df['Close'].pct_change()
            
            # Calculate standard performance metrics
            latest_price = df['Close'].iloc[-1]
            price_change = latest_price - df['Close'].iloc[0]
            pct_change = (price_change / df['Close'].iloc[0]) * 100
            
            # Annualized Sharpe Ratio calculation (4% risk-free hurdle) over 4 years
            risk_free_daily = 0.04 / 252
            excess_returns = df['Daily_Return'].dropna() - risk_free_daily
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0.0
            
            # --- UI VISUALIZATION: HISTORICAL PERFORMANCE ---
            st.subheader(f"📊 {company_name} ({ticker_input}) — 4-Year Macro Price Action")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='#00FFCC', width=2.5)))
            
            fig.update_layout(
                template='plotly_dark',
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Latest Close Price", f"${latest_price:.2f}")
            m2.metric("4-Year Price Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
            m3.metric("4-Year Sharpe Ratio", f"{sharpe_ratio:.2f}", "Risk-Adjusted Efficiency")
            
            st.markdown("---")
            
            # --- 1-YEAR FORECAST ENGINE ---
            st.subheader("🔮 1-Year Volatility Funnel Projections")
            st.markdown("Calculated via statistical trend-line distribution models.")
            
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 366)]
            std_dev = df['Daily_Return'].std()
            avg_return = df['Daily_Return'].mean()
            days_future = np.arange(1, 366)
            
            base_pred = latest_price * (1 + avg_return) ** days_future
            bull_pred = latest_price * (1 + avg_return + (1.5 * std_dev)) ** days_future
            bear_pred = latest_price * (1 + avg_return - (1.5 * std_dev)) ** days_future
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=future_dates, y=bull_pred, mode='lines', name='🟢 Target Upper Bound (Bull)', line=dict(dash='dash', color='#2ecc71')))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=base_pred, mode='lines', name='🔵 Central Tendency (Base)', line=dict(color='#3498db', width=2)))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=bear_pred, mode='lines', name='🔴 Target Lower Bound (Bear)', line=dict(dash='dash', color='#e74c3c')))
            
            fig_pred.update_layout(template='plotly_dark', xaxis_title="Future Date", yaxis_title="Projected Price (USD)", margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown("---")
            
            # --- LIVE NEWS & SENTIMENT VERDICT ENGINE ---
            news_list = fetch_stock_news(ticker_input)
            if news_list:
                headlines_to_show = news_list[:5]
                total_polarity = 0
                scored_headlines = []
                
                for item in headlines_to_show:
                    title = item.get('title', '')
                    blob = TextBlob(title)
                    polarity = blob.sentiment.polarity
                    total_polarity += polarity
                    sentiment_tag = "🟢 Positive" if polarity > 0.05 else ("🔴 Negative" if polarity < -0.05 else "⚪ Neutral")
                    scored_headlines.append({"title": title, "publisher": item.get('publisher', 'News'), "link": item.get('link', '#'), "tag": sentiment_tag})
                
                avg_polarity = total_polarity / len(headlines_to_show) if headlines_to_show else 0
                
                if avg_polarity > 0.05:
                    aggregated_verdict = "😊 POSITIVE SENTIMENT (BULLISH)"
                    verdict_color = "#2ecc71"
                    ai_summary = f"The analysis engine has flagged a primarily bullish sentiment landscape for {ticker_input}. Media coverage vectors reflect high commercial optimism and positive market perception, driving favorable forward momentum."
                elif avg_polarity < -0.05:
                    aggregated_verdict = "⚠️ NEGATIVE SENTIMENT (BEARISH)"
                    verdict_color = "#e74c3c"
                    ai_summary = f"The analysis engine has flagged a primarily bearish sentiment landscape for {ticker_input}. Parsed media vectors reveal prominent ecosystem headwinds, regulatory pressures, or cautious market demand patterns."
                else:
                    aggregated_verdict = "😐 NEUTRAL SENTIMENT BALANCE"
                    verdict_color = "#ff9f43"
                    ai_summary = f"The analysis engine registers an equilibrium state for {ticker_input}. Digital media streams present a closely matched blend of positive and defensive signals, leaving the broader macro perception unbiased."
                
                # Display Master Matrix Box
                st.subheader("🧠 Systemic Market Alpha Matrix")
                st.markdown(
                    f"""
                    <div style="background-color: #1e222d; padding: 25px; border-radius: 10px; border-left: 6px solid {verdict_color}; margin-bottom: 25px;">
                        <h3 style="margin: 0; color: white; letter-spacing: 0.5px;">CURRENT MARKET VIBE: <span style="color: {verdict_color};">{aggregated_verdict}</span></h3>
                        <p style="color: #f1f2f6; margin: 12px 0 0 0; font-size: 15px; line-height: 1.6; font-style: italic;">" {ai_summary} "</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Display News Headlines
                st.subheader("📰 Parsed Data Stream Vector")
                for h in scored_headlines:
                    col_text, col_tag = st.columns([4, 1])
                    with col_text:
                        st.markdown(f"**[{h['title']}]({h['link']})**")
                        st.caption(f"Source: {h['publisher']}")
                    with col_tag:
                        st.markdown(f"**{h['tag']}**")
                    st.markdown("---")
                
        else:
            st.error("No historical transaction data returned for this asset class.")
    except Exception as e:
        st.error(f"Execution fault: {e}")