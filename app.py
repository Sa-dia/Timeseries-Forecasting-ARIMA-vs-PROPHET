
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from scipy.stats import ttest_rel
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

# =======================
# Load Data
# =======================
df = pd.read_csv("AAL_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Available tickers
TICKERS = df['Name'].unique().tolist()

# =======================
# Helper Functions
# =======================
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] < 0.05

def calculate_metrics(y_true, y_pred):
    rmse_val = math.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape_val = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return rmse_val, mae_val, mape_val

def rolling_metrics(y_true, y_pred, window_size):
    rolling_rmse, rolling_mae, rolling_mape = [], [], []
    for i in range(len(y_true) - window_size + 1):
        window_true = y_true[i:i + window_size]
        window_pred = y_pred[i:i + window_size]
        rmse_val = math.sqrt(mean_squared_error(window_true, window_pred))
        mae_val = mean_absolute_error(window_true, window_pred)
        mask = window_true != 0
        mape_val = np.mean(np.abs((window_true[mask] - window_pred[mask]) / window_true[mask])) * 100 if mask.any() else np.nan
        rolling_rmse.append(rmse_val)
        rolling_mae.append(mae_val)
        rolling_mape.append(mape_val)
    return rolling_rmse, rolling_mae, rolling_mape

# ARIMA Walk-forward
def arima_walk_forward(train_series, test_series, max_p=3, max_d=2, max_q=3):
    history = train_series.copy().tolist()
    predictions = []
    best_order, best_aic = None, np.inf

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = sm.tsa.SARIMAX(train_series, order=(p, d, q),
                                           enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic, best_order = res.aic, (p, d, q)
                except:
                    continue

    for t in range(len(test_series)):
        try:
            model = sm.tsa.SARIMAX(history, order=best_order,
                                   enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            yhat = res.forecast(steps=1)[0]
        except:
            yhat = history[-1]
        predictions.append(yhat)
        history.append(float(test_series.iloc[t]))

    return np.array(predictions), best_order

# =======================
# Forecast Function
# =======================
def forecast_app(ticker, model_choice, test_days, window_size):
    data = df[df['Name'] == ticker][['date', 'close']].copy()
    data.columns = ['Date', 'Close']
    data['Close'] = data['Close'].ffill().bfill()

    train = data[:-test_days].copy().reset_index(drop=True)
    test = data[-test_days:].copy().reset_index(drop=True)

    if model_choice == "ARIMA":
        preds, order = arima_walk_forward(train['Close'], test['Close'])
    else:
        prophet_train = train[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_test = test[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model_p = Prophet(daily_seasonality=True)
        model_p.fit(prophet_train)
        forecast = model_p.predict(prophet_test[['ds']])
        preds = forecast['yhat'].values

    # Metrics
    rmse, mae, mape = calculate_metrics(test['Close'].values, preds)

    # Rolling Metrics
    rolling_rmse, rolling_mae, rolling_mape = rolling_metrics(test['Close'].values, preds, window_size)
    rolling_dates = test['Date'][window_size - 1:].reset_index(drop=True)

    # Forecast Plot
    plt.figure(figsize=(12, 5))
    plt.plot(train['Date'], train['Close'], label='Train', color='green')
    plt.plot(test['Date'], test['Close'], label='Test', color='blue')
    plt.plot(test['Date'], preds, label=f'{model_choice} Predictions', color='red', linestyle='--')
    plt.legend()
    plt.title(f'{model_choice} Forecast for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.savefig("forecast.png")
    plt.close()

    # Rolling Plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(rolling_dates, rolling_rmse, label='Rolling RMSE', color='red')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('RMSE', color='red')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(test['Date'][-len(rolling_dates):], test['Close'][-len(rolling_dates):], label='Actual Price', color='blue', alpha=0.5)
    ax2.set_ylabel('Price', color='blue')
    ax2.legend(loc='upper right')
    plt.title(f'Rolling RMSE (window={window_size}) with Actual Price')
    plt.savefig("rolling.png")
    plt.close()

    return "forecast.png", "rolling.png", f"RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%"

# =======================
# Gradio Interface
# =======================
demo = gr.Interface(
    fn=forecast_app,
    inputs=[
        gr.Dropdown(choices=TICKERS, value="AAPL", label="Select Ticker"),
        gr.Radio(["ARIMA", "Prophet"], label="Choose Model", value="ARIMA"),
        gr.Slider(30, 200, step=10, value=90, label="Test Days"),
        gr.Slider(10, 60, step=5, value=30, label="Rolling Window Size")
    ],
    outputs=[
        gr.Image(type="filepath", label="Forecast Plot"),
        gr.Image(type="filepath", label="Rolling Error Plot"),
        gr.Textbox(label="Performance Metrics")
    ],
    title="Stock Forecasting (ARIMA vs Prophet)",
    description="Upload AAL_data.csv in the repo. Choose a always AAL as ticker, model, and forecast parameters."
)

if __name__ == "__main__":
    demo.launch()

