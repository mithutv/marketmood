import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from streamlit_searchbox import st_searchbox
import nltk
from textblob import TextBlob

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- GLOBAL STYLES ---
st.set_page_config(page_title="Market Mood", layout="centered")
st.markdown("""
    <style>
    [data-testid="stDataFrame"] thead tr th { background-color: #000000; color: #FFFFFF; font-weight: 800; font-size: 20px; text-align: center !important; }
    div.stButton > button:first-child { background-color: #007BFF; color: white; border: none; padding: 10px 24px; font-size: 16px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Market Mood: AI-Driven Financial Forecasting")
st.markdown("""
This application analyzes the **last 4 years** of historical price action to provide a 30-day projection using **Meta's Prophet**. 
The dashboard provides an executive summary of current pricing, future forecasts, and real-time sentiment analysis.
""")

# --- SEARCH BOX ---
def search_tickers(searchterm: str):
    if not searchterm: return []
    results = yf.Search(searchterm).quotes
    return [(f"{q.get('shortname', '')} ({q.get('symbol', '')})", q.get('symbol', '')) for q in results if 'symbol' in q]

ticker = st_searchbox(search_tickers, placeholder="Start typing a company name...", label="Search for a Company")
if ticker: st.write(f"### Selected Ticker: {ticker}")

@st.cache_data(ttl=86400)
def get_stock_data(ticker): 
    return yf.download(ticker, period="5y", threads=False, multi_level_index=False)

# --- FORECAST LOGIC ---
# Removed 'and ticker' from button logic to ensure it always renders
if st.button("Generate Forecast"):
    if not ticker:
        st.warning("Please search for and select a ticker first.")
    else:
        try:
            df = get_stock_data(ticker)
            if df.empty:
                st.error("No data found.")
            else:
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
                
                # Apply 4-Year Filter
                four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                prophet_df = prophet_df[prophet_df['ds'] >= four_years_ago].dropna()
                current_price = prophet_df['y'].iloc[-1]

              # --- Prophet Engine ---
            m = Prophet(daily_seasonality=True).fit(prophet_df)
            
            # Create horizons
            future_30 = m.make_future_dataframe(periods=30)
            future_6m = m.make_future_dataframe(periods=180)
            future_1y = m.make_future_dataframe(periods=365)
            
            # Generate predictions
            forecast_30 = m.predict(future_30)
            forecast_6m = m.predict(future_6m)
            forecast_1y = m.predict(future_1y)
            
            # Extract final prices
            price_30 = forecast_30['yhat'].iloc[-1]
            price_6m = forecast_6m['yhat'].iloc[-1]
            price_1y = forecast_1y['yhat'].iloc[-1]
            
            # Helper to calculate dollar delta
            def get_delta_text(forecasted, current):
                return f"{forecasted - current:+.2f}"

            # Metrics Row (Displayed ONLY once)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current", f"${current_price:,.2f}")
            col2.metric("30-Day", f"${price_30:,.2f}", get_delta_text(price_30, current_price))
            col3.metric("6-Month", f"${price_6m:,.2f}", get_delta_text(price_6m, current_price))
            col4.metric("1-Year", f"${price_1y:,.2f}", get_delta_text(price_1y, current_price))
            
            # Use 30-day forecast for the trend description below the graph
            trend_emoji = "📈 (Bullish)" if price_30 > current_price else "📉 (Bearish)"

                # Displaying these as metrics with dollar delta
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current", f"${current_price:,.2f}")
                col2.metric("30-Day", f"${price_30:,.2f}", get_delta_text(price_30, current_price))
                col3.metric("6-Month", f"${price_6m:,.2f}", get_delta_text(price_6m, current_price))
                col4.metric("1-Year", f"${price_1y:,.2f}", get_delta_text(price_1y, current_price))
                
                
                # Sentiment
                news_items = yf.Search(ticker).news
                valid_news = [item for item in news_items if item.get('title')]
                sentiment_scores = [TextBlob(i['title']).sentiment.polarity for i in valid_news[:3]] if valid_news else [0]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                if avg_sentiment > 0.1: 
                    gauge_color, status_label = "#4CAF50", "🟢 Bullish"
                elif avg_sentiment < -0.1: 
                    gauge_color, status_label = "#F44336", "🔴 Bearish"
                else: 
                    gauge_color, status_label = "#9E9E9E", "⚪ Neutral"

                # Metrics Row
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${current_price:,.2f}")
                col2.metric("Forecast (30 Days)", f"${forecasted_price:,.2f}", delta=f"{delta:+.2f}")
                
                with col3:
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <div style="font-size:0.8rem; font-weight:bold;">Sentiment: {avg_sentiment:.2f}</div>
                        <div style="background: conic-gradient(from 270deg, #F44336 0deg, #E0E0E0 90deg, #4CAF50 180deg); width: 80px; height: 40px; border-radius: 40px 40px 0 0; margin: 5px auto;"></div>
                        <div style="font-weight:bold; color:{gauge_color};">{status_label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Score (-1.0 to +1.0): Indicates tone of recent news from negative to positive.")
                
                # Graph
                st.subheader(f"Prediction Trend: {trend_emoji}")
                plot_df = prophet_df.sort_values('ds')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['y'], name='Actual', line=dict(color='#000000')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0000FF', dash='dash')))
                fig.update_layout(title=f"4-Year Price History & 30-Day Forecast", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"The model suggests a {trend_emoji} trend. Projected price: **${forecasted_price:,.2f}** ({delta:+.2f} | {growth_pct:+.2f}%).")
                
                # Quarterly Historical Table
                with st.expander("View Quarterly Historical Summary"):
                    summary_df = prophet_df.copy()
                    summary_df['Year'] = summary_df['ds'].dt.year
                    summary_df['Quarter'] = summary_df['ds'].dt.quarter
                    quarterly = summary_df.groupby(['Year', 'Quarter'])['y'].agg(['mean', 'max', 'min']).reset_index()
                    quarterly = quarterly.sort_values(by=['Year', 'Quarter'], ascending=[False, False])
                    quarterly['Period'] = 'Q' + quarterly['Quarter'].astype(str) + ' ' + quarterly['Year'].astype(str)
                    quarterly = quarterly[['Period', 'mean', 'max', 'min']]
                    quarterly.columns = ['Period', 'Avg Price', 'High', 'Low']
                    for col in ['Avg Price', 'High', 'Low']: quarterly[col] = quarterly[col].map('${:,.2f}'.format)
                    st.dataframe(quarterly, use_container_width=True, hide_index=True)

                # News
                st.write('<h3 style="margin-bottom: 0px;">Recent Market News</h3>', unsafe_allow_html=True)
                for item in valid_news[:3]:
                    link = item.get('link') or item.get('clickThroughUrl') or "#"
                    st.markdown(f"**{item.get('title')}**")
                    st.caption(f"Source: {item.get('publisher')} | [Read More]({link})" if link != "#" else f"Source: {item.get('publisher')}")
        except Exception as e:
            st.error(f"Error: {e}")
