Marketmood: AI-Driven Financial Forecasting
Marketmood is a comprehensive, professional-grade financial analysis suite designed to bridge the gap between complex machine learning and actionable market insights. Built with a focus on transparency and ensemble intelligence, it provides investors with a multi-layered view of market dynamics.

Core Features
Smart Search: Rapidly identify stocks and market assets using an intuitive, real-time search-as-you-type interface.

Multi-Model Forecasting: Instead of relying on a single source of truth, Marketmood synthesizes three distinct analytical engines:

Trend Projection: Meta’s Prophet captures complex seasonal patterns (daily, weekly, annual).

Pattern Predictor: A Random Forest Regressor identifies non-linear relationships using technical indicators (SMA, RSI, ATR, and Volume).

Risk Assessment: A Monte Carlo Simulation runs 10,000 potential future price paths to quantify volatility and provide confidence intervals.

AI Consensus Engine: An intelligent ensemble module that evaluates agreement between models and calculates a Conviction Score, alerting the user to how much weight the system places on its own projection.

Model Performance Transparency: A dedicated "Track Record" module allows users to see how the model performed historically, fostering trust and enabling informed risk management.

How It Works
Data Acquisition: Real-time data is ingested via yfinance, pulling up to 10 years of historical performance to ensure the models have adequate context.

Feature Engineering: The app transforms raw price data into meaningful technical signals, creating a "stationary" feature set that allows the machine learning model to learn patterns effectively.

Ensemble Synthesis: Results from the trend, pattern, and probabilistic engines are combined into a weighted Consensus Forecast.

Verification: The system performs a "walk-forward" backtest to compare its 6-month historical projections against actual market outcomes, providing a reliability rating to the user.

Built With
Streamlit: The interactive framework for the application interface.

Prophet: Meta’s advanced time-series forecasting library.

Scikit-Learn: Powers the Random Forest Regressor and pattern recognition logic.

yfinance: Reliable interface for high-quality market data.

Plotly: High-fidelity, interactive financial charting.

Disclaimer
This tool is for informational and educational purposes only and does not constitute financial advice. Market forecasting is inherently uncertain; always conduct your own due diligence before making investment decisions.
