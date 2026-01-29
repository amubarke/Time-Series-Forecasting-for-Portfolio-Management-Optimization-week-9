
# task3_forecasting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_future_forecast(
    model,
    model_type,
    steps,
    historical_index,
    scaler=None,
    last_window=None,
    lookback=None,
    n_simulations=100
):
    if model_type.lower() in ["arima", "sarima"]:
        forecast = model.get_forecast(steps=steps)
        mean = forecast.predicted_mean.values
        conf_int = forecast.conf_int()
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values

    elif model_type.lower() == "lstm":
        simulations = []

        for _ in range(n_simulations):
            window = last_window.copy()
            preds = []

            for _ in range(steps):
                pred = model.predict(
                    window.reshape(1, lookback, 1),
                    verbose=0
                )
                preds.append(pred[0, 0])
                window = np.append(window[1:], pred)

            preds = scaler.inverse_transform(
                np.array(preds).reshape(-1, 1)
            ).flatten()
            simulations.append(preds)

        simulations = np.array(simulations)
        mean = simulations.mean(axis=0)
        lower = np.percentile(simulations, 5, axis=0)
        upper = np.percentile(simulations, 95, axis=0)

    else:
        raise ValueError("model_type must be 'arima', 'sarima', or 'lstm'")

    forecast_index = pd.date_range(
        start=historical_index[-1],
        periods=steps,
        freq="B"
    )

    return mean, lower, upper, forecast_index

def plot_forecast(
    historical_data,
    test_predictions,
    forecast,
    lower_ci,
    upper_ci,
    forecast_index,
    title
):
    plt.figure(figsize=(14, 6))

    plt.plot(historical_data, label="Historical Data")
    plt.plot(test_predictions, label="Test Predictions", linestyle="--")
    plt.plot(forecast_index, forecast, label="Future Forecast", color="black")

    plt.fill_between(
        forecast_index,
        lower_ci,
        upper_ci,
        alpha=0.3,
        label="Confidence Interval"
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_trend(forecast):
    if forecast[-1] > forecast[0]:
        return "Upward Trend"
    elif forecast[-1] < forecast[0]:
        return "Downward Trend"
    return "Stable Trend"


def assess_opportunities_and_risks(forecast, lower_ci, upper_ci):
    uncertainty = upper_ci - lower_ci

    opportunities = []
    risks = []

    if forecast[-1] > forecast[0]:
        opportunities.append("Expected price appreciation")

    if uncertainty.mean() > np.mean(forecast) * 0.15:
        risks.append("High forecast uncertainty and volatility")

    risks.append("Model-based forecasts ignore external market shocks")

    return opportunities, risks


def assess_forecast_reliability(lower_ci, upper_ci):
    return {
        "Short-term": "High reliability (narrow confidence intervals)",
        "Medium-term": "Moderate reliability",
        "Long-term": "Low reliability due to widening uncertainty"
    }
