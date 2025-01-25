import pandas as pd
from typing import Dict

def simple_forecast(df: pd.DataFrame, value_col: str="value", periods: int=7) -> pd.DataFrame:
    """
    A naive forecast using exponential smoothing or ARIMA. 
    We'll do a simple approach: the last value repeated. 
    Real logic can incorporate 'statsmodels' if needed.
    """
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    df = df.copy().sort_values("date")
    series = df[value_col]
    if len(series) < 2:
        return pd.DataFrame()
    model = SimpleExpSmoothing(series, initialization_method="heuristic").fit()
    forecast_vals = model.forecast(periods)
    future_dates = pd.date_range(df["date"].iloc[-1], periods=periods+1, freq="D")[1:]
    return pd.DataFrame({"date": future_dates, "forecast": forecast_vals})

def forecast_upstream_metrics(drivers: Dict[str, pd.DataFrame], periods: int=7):
    """
    For each upstream metric, call simple_forecast. Returns dict driver_id-> forecast df
    """
    forecasts = {}
    for driver_id, driver_df in drivers.items():
        forecasts[driver_id] = simple_forecast(driver_df, value_col="value", periods=periods)
    return forecasts

def forecast_metric_dimensions():
    """
    Stub. For each dimension slice, forecast individually. 
    """
    pass

def forecast_best_and_worst_case(forecast_df: pd.DataFrame, buffer_pct: float=10.0):
    """
    Create best/worst case by ± buffer_pct around the forecast.
    """
    df = forecast_df.copy()
    df["best_case"] = df["forecast"] * (1 + buffer_pct/100.0)
    df["worst_case"] = df["forecast"] * (1 - buffer_pct/100.0)
    return df

def forecast_target_achievement(forecast_df: pd.DataFrame, target_df: pd.DataFrame):
    """
    Compare forecast to target. 
    Returns df with on/off track flags.
    """
    pass

def calculate_forecast_accuracy():
    """
    Compare past forecast vs. actual (RMSE, MAPE). Stub for demonstration.
    """
    pass

def assess_forecast_uncertainty():
    """
    Summarize how wide forecast intervals are. Stub.
    """
    pass

def decompose_forecast_drivers():
    """
    If using a driver-based forecast, show partial effect from each driver. Stub.
    """
    pass
