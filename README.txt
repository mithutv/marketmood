Market Mood: AI-Driven Financial Forecasting
Market Mood is a robust, user-friendly web application built with Streamlit and Meta’s Prophet library. It provides financial analysts and retail investors with data-driven, 30-day price trend projections based on historical market performance.
Features
Smart Search: Quickly find any stock ticker using the intuitive search-as-you-type interface.

Predictive Modeling: Utilizes Prophet to account for daily, weekly, and annual seasonality in stock prices.

Visual Uncertainty: Interactive charts displaying confidence intervals to visualize potential market volatility.

Performance Metrics: Instantly compare the current market price against 30-day forecast projections.

Historical Insights: View and filter historical data directly within the application.
How It Works
Data Acquisition: The app pulls historical Adj Close price data using the yfinance library.

Preprocessing: It cleans missing values and formats the temporal data for time-series analysis.

Forecasting: Meta's Prophet model is trained on the historical trend to project future price action.

Reporting: Results are rendered via interactive Plotly graphs and summary statistics.

Built With
Streamlit - The framework for the web interface.

Prophet - The forecasting engine developed by Meta.

yfinance - For downloading market data.

Plotly - For interactive financial visualizations.

Disclaimer: This tool is for informational and educational purposes only. It is not financial advice. Market forecasting is inherently uncertain; always conduct your own research before making investment decisions.
