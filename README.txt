Marketmood: AI-Driven Financial Forecasting

MarketMood is a financial analysis platform that uses machine learning and statistical models to generate stock market forecasts and measure prediction confidence. 

Project Goal
The goalis to combine multiple modeling approaches to provide more reliable and transparent market insights.

Core Features
Allows users to quickly search and select stocks using real-time data.

Multi-Model Forecasting System:  MarketMood uses multiple models rather than relying on a single prediction method:

1. Prophet model for time-series trend and seasonality analysis
2. Random Forest model using technical indicators such as SMA, RSI, ATR, and volume
3. Monte Carlo simulation with 10,000 iterations to estimate volatility and price ranges


Consensus Engine
It Combines outputs from all models to generate a unified forecast. It also produces a confidence or “conviction” score based on how closely the models agree.

Model Performance Tracking
Includes a backtesting system that compares historical predictions with actual market outcomes to evaluate model reliability over time.

How It Works
1. Data Acquisition: Real-time data is ingested via yfinance, pulling up to 10 years of historical performance to ensure the models have adequate context.
2. Feature Engineering: Raw price data is converted into technical indicators and structured features for modeling.
3. Ensemble Synthesis: Results from the trend, pattern, and probabilistic engines are combined into a weighted Consensus Forecast.
4. Testing: The system evaluates past predictions using walk-forward validation to measure accuracy.

Technologies Used

Streamlit for the web interface
Prophet for time-series forecasting
Scikit-learn for machine learning models
yfinance for market data
Plotly for data visualization
Plotly: High-fidelity, interactive financial charting.

Disclaimer
This tool is for informational and educational purposes only and does not constitute financial advice. Market forecasting is inherently uncertain; always conduct your own due diligence before making investment decisions.
