# Week 9: Time Series Forecasting for Portfolio Management

## Project Overview
Guide Me in Finance (GMF) Investments specializes in personalized portfolio management. This project focuses on applying **time series forecasting models** to historical financial data to enhance portfolio management strategies. The main goal is to predict future trends in Tesla (TSLA) stock prices and compare classical and deep learning approaches for forecasting.

---

## Data
- **Assets:** TSLA (Tesla), BND (Vanguard Total Bond Market ETF), SPY (S&P 500 ETF)  
- **Period:** January 1, 2015 – January 15, 2026  
- **Data Fields:** Date, Open, High, Low, Close, Adjusted Close, Volume  
- **Source:** [YFinance Python Library](https://pypi.org/project/yfinance/)

---

## Task 1: Data Preprocessing and EDA
**Objective:** Load, clean, and explore data for modeling.  

### Steps:
1. Extract historical data for TSLA, BND, SPY using YFinance.  
2. Clean data:
   - Handle missing values  
   - Ensure correct data types  
   - Feature engineering: calculate daily returns and rolling volatility  
3. Exploratory Data Analysis (EDA):
   - Visualize closing prices and daily returns  
   - Detect outliers and unusual price movements  
4. Stationarity and trend analysis:
   - Augmented Dickey-Fuller test for stationarity  
   - Differencing applied if required for ARIMA models  
5. Risk metrics:
   - Sharpe Ratio  
   - Value at Risk (VaR)  

**Deliverables:**  
- Cleaned dataset saved in `data/processed/task1_clean_data.csv`  
- EDA visualizations and summary insights  

---

## Task 2: Time Series Forecasting

**Objective:** Develop, train, and evaluate ARIMA and LSTM models to forecast TSLA stock prices.

### Steps:

#### 1. Data Preparation
- Split data chronologically:  
  - Training: 2015–2024  
  - Testing: 2025–2026  
- Scale data for LSTM (MinMaxScaler), ARIMA uses raw or differenced series.

#### 2. ARIMA Modeling
- Determine optimal `(p,d,q)` using `auto_arima` or grid search.  
- Fit ARIMA model to training data.  
- Forecast test period and visualize predictions.  

#### 3. LSTM Modeling
- Prepare sequences (e.g., 60-day window → next-day prediction).  
- Build LSTM network:
  - Input layer matching sequence length  
  - One or more LSTM layers  
  - Dense output layer  
- Train using appropriate hyperparameters (epochs, batch size).  
- Forecast test period and inverse-scale predictions.

#### 4. Model Optimization
- ARIMA: tune `(p,d,q)` parameters for lowest AIC / RMSE  
- LSTM: experiment with window size, layers, neurons, epochs, batch size  

#### 5. Model Evaluation
- Metrics: MAE, RMSE, MAPE  

**Comparison Table:**

| Model  | MAE     | RMSE    | MAPE (%) |
|--------|---------|---------|----------|
| ARIMA  | 69.50   | 82.93   | 22.56    |
| LSTM   | 10.75   | 13.94   | 2.97     |

**Rationale for Model Selection:**  
- LSTM outperforms ARIMA in all metrics, capturing non-linear patterns and temporal dependencies in TSLA prices.  
- ARIMA is simpler and interpretable but less accurate for high-volatility assets.  
- For predictive accuracy, LSTM is preferred; ARIMA may still be used for quick or explainable forecasts.

---

## Project Structure

portfolio-optimization/

├── .vscode/

│   └── settings.json

├── .github/

│   └── workflows/

│    └── unittests.yml

├── .venv/

├── .gitignore

├── requirements.txt

├── README.md

├── data/

│   └── processed/task1_clean_data.csv

├── notebooks/

│   ├── __init__.py

│   └──Task_1_preprocess_and_EDA.ipynb
    └── modeling.ipynb
├── src/

│   └── __init__.py

├── tests/

│   └── __init__.py

└── scripts/
       └── __init__.py