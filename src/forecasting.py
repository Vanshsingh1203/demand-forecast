import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_test_split_ts(df, date_col="date", value_col="sales", test_ratio=0.2):
    """Split time series data into train and test sets."""
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate accuracy metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    return {
        "model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape_val, 2),
    }


def forecast_moving_average(train, test, value_col="sales", window=7):
    """Simple Moving Average forecast."""
    predictions = []
    history = list(train[value_col].values)
    for i in range(len(test)):
        avg = np.mean(history[-window:])
        predictions.append(max(0, avg))
        history.append(test[value_col].values[i])
    metrics = evaluate_model(test[value_col].values, predictions, f"Moving Avg ({window}d)")
    return predictions, metrics


def forecast_exp_smoothing(train, test, value_col="sales", seasonal_periods=7):
    """Holt-Winters Exponential Smoothing forecast."""
    try:
        train_vals = train[value_col].values.astype(float)
        train_vals = np.maximum(train_vals, 0)
        model = ExponentialSmoothing(
            train_vals, trend="add", seasonal="add", seasonal_periods=seasonal_periods,
        ).fit(optimized=True)
        predictions = model.forecast(len(test))
        predictions = np.maximum(predictions, 0)
        metrics = evaluate_model(test[value_col].values, predictions, "Holt-Winters")
        return list(predictions), metrics
    except Exception as e:
        print(f"Holt-Winters failed: {e}")
        return [0] * len(test), {"model": "Holt-Winters", "MAE": 0, "RMSE": 0, "MAPE": 0}


def forecast_arima(train, test, value_col="sales", order=(2, 1, 2)):
    """ARIMA forecast."""
    try:
        train_vals = train[value_col].values.astype(float)
        model = ARIMA(train_vals, order=order).fit()
        predictions = model.forecast(steps=len(test))
        predictions = np.maximum(predictions, 0)
        metrics = evaluate_model(test[value_col].values, predictions, f"ARIMA{order}")
        return list(predictions), metrics
    except Exception as e:
        print(f"ARIMA failed: {e}")
        return [0] * len(test), {"model": f"ARIMA{order}", "MAE": 0, "RMSE": 0, "MAPE": 0}


def forecast_prophet(train, test, value_col="sales", date_col="date"):
    """Facebook Prophet forecast."""
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed, skipping.")
        return [0] * len(test), {"model": "Prophet", "MAE": 0, "RMSE": 0, "MAPE": 999}
    try:
        prophet_train = train[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
        prophet_train["y"] = np.maximum(prophet_train["y"], 0)
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
        model.fit(prophet_train)
        future = test[[date_col]].rename(columns={date_col: "ds"})
        forecast = model.predict(future)
        predictions = np.maximum(forecast["yhat"].values, 0)
        metrics = evaluate_model(test[value_col].values, predictions, "Prophet")
        return list(predictions), metrics
    except Exception as e:
        print(f"Prophet failed: {e}")
        return [0] * len(test), {"model": "Prophet", "MAE": 0, "RMSE": 0, "MAPE": 999}


def forecast_xgboost(train, test, value_col="sales"):
    """XGBoost forecast with lag features."""
    try:
        from xgboost import XGBRegressor

        def create_features(df, value_col, lags=[7, 14, 21, 28]):
            d = df.copy()
            for lag in lags:
                d[f"lag_{lag}"] = d[value_col].shift(lag)
            d["rolling_7"] = d[value_col].shift(1).rolling(7).mean()
            d["rolling_14"] = d[value_col].shift(1).rolling(14).mean()
            d["rolling_28"] = d[value_col].shift(1).rolling(28).mean()
            d["rolling_std_7"] = d[value_col].shift(1).rolling(7).std()
            return d

        full = pd.concat([train, test], ignore_index=True)
        full = create_features(full, value_col)
        full = full.dropna()
        feature_cols = [c for c in full.columns if c.startswith(("lag_", "rolling_"))]
        split = len(full) - len(test)
        if split <= 0:
            split = int(len(full) * 0.8)
        tr = full.iloc[:split]
        te = full.iloc[split:]
        if len(te) == 0 or len(tr) == 0:
            raise ValueError("Not enough data after feature creation")
        model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, verbosity=0)
        model.fit(tr[feature_cols], tr[value_col])
        predictions = model.predict(te[feature_cols])
        predictions = np.maximum(predictions, 0)
        actuals = te[value_col].values[:len(predictions)]
        metrics = evaluate_model(actuals, predictions, "XGBoost")
        return list(predictions), metrics
    except Exception as e:
        print(f"XGBoost failed: {e}")
        return [0] * len(test), {"model": "XGBoost", "MAE": 0, "RMSE": 0, "MAPE": 0}


def run_all_forecasts(train, test, value_col="sales", date_col="date"):
    """Run all forecasting methods and return results."""
    results = []
    print("Running Moving Average...")
    preds_ma, metrics_ma = forecast_moving_average(train, test, value_col, window=7)
    results.append({"predictions": preds_ma, **metrics_ma})
    print("Running Holt-Winters...")
    preds_hw, metrics_hw = forecast_exp_smoothing(train, test, value_col)
    results.append({"predictions": preds_hw, **metrics_hw})
    print("Running ARIMA...")
    preds_ar, metrics_ar = forecast_arima(train, test, value_col)
    results.append({"predictions": preds_ar, **metrics_ar})
    print("Running Prophet...")
    preds_pr, metrics_pr = forecast_prophet(train, test, value_col, date_col)
    results.append({"predictions": preds_pr, **metrics_pr})
    print("Running XGBoost...")
    preds_xg, metrics_xg = forecast_xgboost(train, test, value_col)
    results.append({"predictions": preds_xg, **metrics_xg})
    results_df = pd.DataFrame(results).sort_values("MAPE")
    print("\n--- Forecast Comparison ---")
    print(results_df[["model", "MAE", "RMSE", "MAPE"]].to_string(index=False))
    return results_df