import pandas as pd
import numpy as np
from typing import Optional, Union
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pmdarima import arima as pmd_arima  # optional, if installed

def simple_forecast(
    df: pd.DataFrame,
    value_col: str = "value",
    periods: int = 7,
    method: str = "ses",  # "naive", "ses", "holtwinters", "auto_arima"
    seasonal_periods: Optional[int] = None,
    date_col: Optional[str] = None,
    freq: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    A basic forecast function that can do:
      - 'naive': just repeat the last value
      - 'ses': Simple Exponential Smoothing (statsmodels)
      - 'holtwinters': ExponentialSmoothing with optional seasonality
      - 'auto_arima': pmdarima.auto_arima approach

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the numeric column value_col, optionally a date_col for indexing.
    value_col : str, default='value'
        Name of the numeric column to forecast.
    periods : int, default=7
        Number of future periods to forecast.
    method : str, default='ses'
        Which approach to use. One of ['naive','ses','holtwinters','auto_arima'].
    seasonal_periods : int or None, default=None
        If you want Holt-Winters with seasonality or auto_arima with seasonality, set e.g. 7 for weekly if daily data, 12 for monthly, etc.
    date_col : str or None, default=None
        If provided, we parse it as a datetime and set as index. If None, we treat rows as equally spaced.
    freq : str or None, default=None
        Pandas frequency string (e.g. 'D' for daily). If you have a date_col, we can set the freq for the DatetimeIndex to produce future timestamps.
    **kwargs :
        Additional parameters passed to the forecasting model (like smoothing_level, auto_arima seasonal=True, etc.).

    Returns
    -------
    pd.DataFrame
        Contains columns [date, forecast], or if no date_col is used, an integer index for future periods.
        The forecast column name is "forecast".

    Notes
    -----
    - For 'naive', we just repeat the last known value for all future periods.
    - For 'ses', we use statsmodels SimpleExpSmoothing.
    - For 'holtwinters', we use statsmodels ExponentialSmoothing with possible seasonal_periods.
    - For 'auto_arima', we rely on pmdarima.auto_arima(...) if installed. 
    - This function is intentionally minimal. Real usage might do hyperparameter tuning, model diagnostics, etc.
    """
    df = df.copy()

    # If date_col is provided, set it as DateTimeIndex
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        df.set_index(date_col, inplace=True)
        if freq:
            df = df.asfreq(freq)
    
    series = df[value_col].dropna()

    if len(series) < 2:
        # Not enough data to forecast properly => fallback
        last_val = series.iloc[-1] if len(series) == 1 else 0
        future_index = []
        if date_col is not None and freq:
            start_date = df.index[-1] if len(df) > 0 else pd.to_datetime("2025-01-01")
            future_index = pd.date_range(start_date, periods=periods+1, freq=freq)[1:]
        else:
            # use numeric range
            future_index = np.arange(len(df), len(df)+periods)
        naive_forecast = [last_val]*periods
        return pd.DataFrame({"date": future_index, "forecast": naive_forecast})

    # depending on method:
    if method == "naive":
        last_val = series.iloc[-1]
        if date_col is not None and freq:
            start = df.index[-1]
            future_idx = pd.date_range(start, periods=periods+1, freq=freq)[1:]
            fc_vals = [last_val]*periods
            return pd.DataFrame({"date": future_idx, "forecast": fc_vals})
        else:
            future_idx = np.arange(len(series), len(series)+periods)
            fc_vals = [last_val]*periods
            return pd.DataFrame({"date": future_idx, "forecast": fc_vals})

    elif method == "ses":
        model = SimpleExpSmoothing(series, initialization_method="estimated")
        fit = model.fit(**kwargs)  # e.g. smoothing_level=...
        fc_vals = fit.forecast(periods)
    elif method == "holtwinters":
        # e.g. use ExponentialSmoothing
        model = ExponentialSmoothing(
            series,
            trend=kwargs.pop("trend", None), 
            seasonal=kwargs.pop("seasonal", None),
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        fit = model.fit(**kwargs)
        fc_vals = fit.forecast(periods)
    elif method == "auto_arima":
        # requires pmdarima
        # we might guess seasonal=True if seasonal_periods is not None
        seasonal = (seasonal_periods is not None and seasonal_periods > 1)
        model = pmd_arima.auto_arima(
            series,
            seasonal=seasonal,
            m=seasonal_periods if seasonal else 1,
            **kwargs
        )
        fc_vals = model.predict(n_periods=periods)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build future index
    if date_col is not None and freq:
        last_idx = df.index[-1]
        future_idx = pd.date_range(last_idx, periods=periods+1, freq=freq)[1:]
    else:
        # numeric
        future_idx = np.arange(len(series), len(series)+periods)

    result = pd.DataFrame({"date": future_idx, "forecast": fc_vals})
    return result.reset_index(drop=True)

def forecast_upstream_metrics(
    drivers: Dict[str, pd.DataFrame],
    periods: int = 7,
    method: str = "ses",
    date_col: Optional[str] = None,
    freq: Optional[str] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    For each driver_id => DataFrame, call simple_forecast. 
    Returns a dict driver_id-> forecast DataFrame.

    Parameters
    ----------
    drivers : dict
        driver_id -> DataFrame with at least [value_col]. Optionally a date_col if time-based.
    periods : int, default=7
        number of future periods to forecast each driver.
    method : str, default='ses'
        forecasting approach passed to simple_forecast.
    date_col : str or None
        if each DF has a date_col, pass it to simple_forecast.
    freq : str or None
        frequency string e.g. 'D'.
    **kwargs :
        additional arguments for simple_forecast

    Returns
    -------
    Dict[str, pd.DataFrame]
        driver_id -> a DF with columns [date, forecast].
    """
    from .forecasting import simple_forecast  # or relative import

    forecasts = {}
    for driver_id, df_driver in drivers.items():
        fc_df = simple_forecast(
            df_driver, 
            value_col="value", 
            periods=periods,
            method=method,
            date_col=date_col,
            freq=freq,
            **kwargs
        )
        forecasts[driver_id] = fc_df
    return forecasts

def forecast_metric_dimensions(
    df: pd.DataFrame,
    slice_col: str,
    date_col: str = "date",
    value_col: str = "value",
    periods: int = 7,
    method: str = "ses",
    freq: Optional[str] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    For each dimension slice, forecast individually. 
    Return a dict slice_value-> forecast DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        columns=[slice_col, date_col, value_col].
    slice_col : str
        dimension column e.g. region
    date_col : str, default='date'
    value_col : str, default='value'
    periods : int, default=7
    method : str, default='ses'
        passed to simple_forecast
    freq : str or None
    **kwargs :
        additional parameters for simple_forecast

    Returns
    -------
    dict
        {slice_value: DataFrame of forecast}
    """
    from .forecasting import simple_forecast

    results = {}
    unique_slices = df[slice_col].dropna().unique()
    for s in unique_slices:
        sub_df = df[df[slice_col] == s].copy()
        fc = simple_forecast(
            sub_df,
            value_col=value_col,
            periods=periods,
            method=method,
            date_col=date_col,
            freq=freq,
            **kwargs
        )
        results[s] = fc
    return results

def forecast_best_and_worst_case(
    forecast_df: pd.DataFrame, 
    buffer_pct: float = 10.0,
    forecast_col: str = "forecast"
) -> pd.DataFrame:
    """
    Create best/worst case by ± buffer_pct around the forecast values.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        must have at least [forecast_col]. Typically from simple_forecast.
    buffer_pct : float, default=10.0
        e.g. 10 => ±10%
    forecast_col : str, default='forecast'

    Returns
    -------
    pd.DataFrame
        same data plus [best_case, worst_case].

    Notes
    -----
    - If you have actual confidence intervals from the model, prefer those.
    """
    df = forecast_df.copy()
    df["best_case"] = df[forecast_col] * (1.0 + buffer_pct/100.0)
    df["worst_case"] = df[forecast_col] * (1.0 - buffer_pct/100.0)
    return df

def forecast_target_achievement(
    forecast_df: pd.DataFrame,
    target_df: pd.DataFrame,
    forecast_col: str = "forecast",
    target_col: str = "target",
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Compare forecasted values to future targets, returning columns for GvA or on_track flags.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        e.g. from simple_forecast, with [date_col, forecast_col].
    target_df : pd.DataFrame
        e.g. [date_col, target_col].
    forecast_col : str, default='forecast'
    target_col : str, default='target'
    date_col : str, default='date'

    Returns
    -------
    pd.DataFrame
        Merged with columns [date_col, forecast, target, abs_diff, pct_diff, on_track?].
    """
    df_f = forecast_df[[date_col, forecast_col]].copy()
    df_t = target_df[[date_col, target_col]].copy()
    merged = pd.merge(df_f, df_t, on=date_col, how="left")

    merged["abs_diff"] = merged[forecast_col] - merged[target_col]
    merged["pct_diff"] = merged.apply(
        lambda row: (row["abs_diff"] / row[target_col] * 100.0) 
                    if pd.notna(row[target_col]) and row[target_col] != 0 else None,
        axis=1
    )
    # define some on_track logic if forecast >= target
    merged["on_track"] = merged.apply(
        lambda row: True if pd.notna(row[target_col]) and row[forecast_col] >= row[target_col] else False,
        axis=1
    )
    return merged

def calculate_forecast_accuracy(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str = "date",
    actual_col: str = "actual",
    forecast_col: str = "forecast"
) -> dict:
    """
    Compare past forecast vs. actual for error metrics: RMSE, MAE, MAPE.

    Parameters
    ----------
    actual_df : pd.DataFrame
        e.g. columns=[date_col, actual_col].
    forecast_df : pd.DataFrame
        e.g. columns=[date_col, forecast_col], covering the same time range.
    date_col : str
    actual_col : str, default='actual'
    forecast_col : str, default='forecast'

    Returns
    -------
    dict
        { 'rmse': float, 'mae': float, 'mape': float, ...}

    Notes
    -----
    - We do a simple merge on date_col, ignoring extra rows.
    """
    df_a = actual_df[[date_col, actual_col]]
    df_f = forecast_df[[date_col, forecast_col]]

    merged = pd.merge(df_a, df_f, on=date_col, how="inner")
    merged.dropna(subset=[actual_col, forecast_col], inplace=True)
    if len(merged) == 0:
        return {"rmse": None, "mae": None, "mape": None, "n": 0}

    errors = merged[forecast_col] - merged[actual_col]
    abs_errors = errors.abs()
    rmse = (errors ** 2).mean() ** 0.5
    mae = abs_errors.mean()
    # MAPE => average of abs((actual-forecast)/actual)*100
    # watch out for zero actual
    def safe_ape(row):
        a = row[actual_col]
        f = row[forecast_col]
        if a == 0:
            return None
        return abs((f - a)/a)*100.0
    
    merged["ape"] = merged.apply(safe_ape, axis=1)
    mape = merged["ape"].mean(skipna=True)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape) if pd.notna(mape) else None,
        "n": len(merged)
    }

def assess_forecast_uncertainty(
    forecast_df: pd.DataFrame,
    lower_col: str = "lower",
    upper_col: str = "upper",
    forecast_col: str = "forecast"
) -> dict:
    """
    Summarize how wide the forecast intervals are. 
    E.g. average or max interval width.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        with columns [forecast_col, lower_col, upper_col].
    lower_col : str, default='lower'
    upper_col : str, default='upper'
    forecast_col : str, default='forecast'

    Returns
    -------
    dict
        { 'mean_interval_width': float, 'max_interval_width': float, 'relative_width': ... }

    Notes
    -----
    - If these columns don't exist, we return empty.
    """
    if lower_col not in forecast_df.columns or upper_col not in forecast_df.columns:
        return {"error": f"No columns {lower_col}/{upper_col} in df", "mean_interval_width": None}

    df = forecast_df.dropna(subset=[lower_col, upper_col, forecast_col]).copy()
    if len(df) == 0:
        return {"mean_interval_width": None, "max_interval_width": None}

    df["interval_width"] = df[upper_col] - df[lower_col]
    mean_width = df["interval_width"].mean()
    max_width = df["interval_width"].max()
    # maybe a relative measure => mean((upper-lower)/forecast)
    df["relative_width"] = df["interval_width"] / df[forecast_col].abs().replace(0, np.nan)
    rel_width = df["relative_width"].mean()

    return {
        "mean_interval_width": float(mean_width),
        "max_interval_width": float(max_width),
        "mean_relative_width": float(rel_width)
    }

def decompose_forecast_drivers(
    driver_forecasts: Dict[str, pd.DataFrame],
    driver_coefs: Dict[str, float],
    base_intercept: float = 0.0
) -> pd.DataFrame:
    """
    If we have a linear model y = intercept + sum_j(coef_j * driver_j),
    and we have forecasted each driver_j over time, we can compute the 
    partial effect for each driver in the forecast horizon.

    Parameters
    ----------
    driver_forecasts : dict
        driver_id-> DataFrame with columns [date, forecast].
    driver_coefs : dict
        driver_id-> float. The model coefficient for that driver.
    base_intercept : float, default=0.0
        The model intercept.

    Returns
    -------
    pd.DataFrame
        with columns [date, total_forecast, driver_<id>_contribution, ...].
        The sum of driver contributions + intercept = total_forecast.

    Example
    -------
    If we have 2 drivers: A,B => coefs=2.0, -1.0 => 
    forecasted A(t), B(t) => we build total_forecast(t)= intercept + 2*A(t)+ -1*B(t).
    Also store partial columns driver_A_contrib=2*A(t), driver_B_contrib=-1*B(t).
    """
    # find union of all forecast dates
    all_dates = set()
    for df in driver_forecasts.values():
        all_dates.update(df["date"].unique())
    all_dates = sorted(all_dates)

    # We'll build a table with one row per date
    rows = []
    for d in all_dates:
        row_data = {"date": d}
        total = base_intercept
        # for each driver, find forecast at date d
        for drv, df_fc in driver_forecasts.items():
            # find row in df_fc with date == d
            val = df_fc.loc[df_fc["date"]==d,"forecast"]
            if len(val) == 1:
                drv_fc = val.iloc[0]
            else:
                drv_fc = np.nan
            coef = driver_coefs.get(drv, 0.0)
            contrib = coef*drv_fc
            row_data[f"driver_{drv}_contribution"] = contrib
            total += contrib
        row_data["total_forecast"] = total
        rows.append(row_data)

    return pd.DataFrame(rows)