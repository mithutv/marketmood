import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob

# The ttl (Time-To-Live) ensures data refreshes only once every 24 hours
@st.cache_data(ttl=86400) 
def get_stock_data(ticker):
    data = yf.download(ticker)
    return data


# 1. Page Configuration
st.set_page_config(page_title="MarketMood AI Alpha", page_icon="⚡", layout="wide")

# --- INJECT PREMIUM UI STYLING ---
st.markdown("""
    <style>
        /* Customize the look of container cards */
        .stElementContainer div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #2b303c !important;
            background-color: #131722;
        }
        /* Make subheaders crisp and clean */
        h2, h3 {
            letter-spacing: -0.5px;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ MarketMood AI — Quant Engine")
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
    df = stock.history(period="1y")
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
         # --- 1-YEAR FORECAST ENGINE (MONTE CARLO GBM) ---
            st.subheader("🔮 1-Year Monte Carlo Probability Projections")
            st.markdown("Simulating 100 random asset trajectories using Geometric Brownian Motion (GBM) stochastic calculus.")
            
            # 1. Setup Simulation Parameters
            days_future = 365
            num_simulations = 100
            future_dates = [df.index[-1] + timedelta(days=int(i)) for i in range(1, days_future + 1)]
            
            std_dev = float(df['Daily_Return'].std())
            avg_return = float(df['Daily_Return'].mean())
            
            # Calculate Risk-Adjusted Drift for GBM: (μ - 0.5 * σ^2)
            drift = avg_return - (0.5 * (std_dev ** 2))
            
            # Create a matrix to hold simulation paths: shape (365, 100)
            simulation_matrix = np.zeros((days_future, num_simulations))
            
            # Generate random shocks for all days and all simulations simultaneously
            # np.random.normal represents the random "shock" variable Z
            random_shocks = np.random.normal(0, 1, (days_future, num_simulations))
            
            # 2. Compute the Stochastic Paths
            for sim in range(num_simulations):
                current_price = latest_price
                for day in range(days_future):
                    # GBM Formula: S_t = S_{t-1} * exp(drift + std_dev * Z)
                    exponent = drift + (std_dev * random_shocks[day, sim])
                    current_price = current_price * np.exp(exponent)
                    simulation_matrix[day, sim] = current_price
            
            # 3. Extract Distribution Targets at Terminal Day (Day 365)
            terminal_prices = simulation_matrix[-1, :]
            
            # Base Case = Median path (50th percentile)
            terminal_base = float(np.percentile(terminal_prices, 50))
            # Bull Case = Top tier outcome (85th percentile)
            terminal_bull = float(np.percentile(terminal_prices, 85))
            # Bear Case = Downside tier outcome (15th percentile)
            terminal_bear = float(np.percentile(terminal_prices, 15))
            
            # 4. Render the Interactive Plotly Chart
            fig_pred = go.Figure()
            
            # Plot each individual random walk line as a semi-transparent background strand
            for sim in range(num_simulations):
                fig_pred.add_trace(go.Scatter(
                    x=future_dates, 
                    y=simulation_matrix[:, sim], 
                    mode='lines', 
                    line=dict(color='rgba(100, 149, 237, 0.15)', width=1),
                    showlegend=False
                ))
            
            # Highlight the Statistical Aggregates over the top
            fig_pred.add_trace(go.Scatter(x=future_dates, y=np.percentile(simulation_matrix, 85, axis=1), mode='lines', name='🟢 Bull Bound (85th Pctl)', line=dict(color='#2ecc71', width=2, dash='dash')))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=np.percentile(simulation_matrix, 50, axis=1), mode='lines', name='🔵 Median Path (Expected)', line=dict(color='#3498db', width=3)))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=np.percentile(simulation_matrix, 15, axis=1), mode='lines', name='🔴 Bear Bound (15th Pctl)', line=dict(color='#e74c3c', width=2, dash='dash')))
            
            fig_pred.update_layout(
                template='plotly_dark', 
                xaxis_title="Future Horizon", 
                yaxis_title="Simulated Price (USD)", 
                margin=dict(l=20, r=20, t=20, b=20), 
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # --- PREDICTIVE METRICS MATRIX ---
            st.markdown("#### 🎯 12-Month Monte Carlo Target Distribution")
            
            # Calculate implied returns from the current market price
            base_return = ((terminal_base - latest_price) / latest_price) * 100
            bull_return = ((terminal_bull - latest_price) / latest_price) * 100
            bear_return = ((terminal_bear - latest_price) / latest_price) * 100
            
            # Display target cards using Streamlit columns
            p1, p2, p3 = st.columns(3)
            p1.metric("Bear Distribution (15%)", f"${terminal_bear:.2f}", f"{bear_return:.2f}% Downside", delta_color="inverse")
            p2.metric("Median Expectation (50%)", f"${terminal_base:.2f}", f"+{base_return:.2f}% Expected")
            p3.metric("Bull Distribution (85%)", f"${terminal_bull:.2f}", f"+{bull_return:.2f}% Upside")
            
            # Add a structural data insight footer
            st.caption(
                f"**Engine Note:** This matrix reflects 100 paths modeling standard geometric drift "
                f"integrated with random market variance shocks. Percentiles capture density boundaries."
            )

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
