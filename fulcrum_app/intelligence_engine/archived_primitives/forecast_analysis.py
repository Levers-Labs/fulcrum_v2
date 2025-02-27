"""
forecast_analysis.py

This module provides a comprehensive set of forecasting tools for funnel analysis, including:
  - Helper functions for date-grain conversion and resampling.
  - A unified forecast function that supports multiple forecasting methods:
      - 'naive': repeat the last known value.
      - 'ses': Simple Exponential Smoothing.
      - 'holtwinters': Holt-Winters Exponential Smoothing with optional trend/seasonality.
      - 'auto_arima': Automated ARIMA modeling using pmdarima.
  - Methods to forecast upstream metrics and forecast individual dimensions.
  - Utility functions to compute best/worst case scenarios, compare forecasts to targets,
    assess forecast accuracy, evaluate forecast uncertainty, and decompose forecast drivers.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List, Callable
import logging

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pmdarima import arima as pmd_arima

# =============================================================================
# Helper Functions
# =============================================================================

def _convert_grain_to_freq(grain: str) -> str:
    """
    Map common grain strings to a pandas frequency alias.
    
    Examples:
      "day" or "daily"   -> "D"
      "week" or "weekly" -> "W-MON"
      "month" or "monthly" -> "MS" (month start)
      "quarter" or "quarterly" -> "QS" (quarter start)
    
    Parameters
    ----------
    grain : str
        The grain string.
    
    Returns
    -------
    str
        A pandas frequency alias.
    
    Raises
    ------
    ValueError
        If the grain is unsupported.
    """
    grain = grain.lower()
    if grain in ["day", "daily"]:
        return "D"
    elif grain in ["week", "weekly"]:
        return "W-MON"
    elif grain in ["month", "monthly"]:
        return "MS"
    elif grain in ["quarter", "quarterly"]:
        return "QS"
    else:
        raise ValueError(f"Unsupported grain '{grain}'.")

def _maybe_resample_for_grain(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    grain: Optional[str],
    agg_func: Optional[Callable] = None
) -> pd.DataFrame:
    """
    If a grain is provided, resample the DataFrame to that frequency using an aggregation function.
    
    By default, the aggregation is a sum (to mimic the old code behavior). Optionally, an
    alternative aggregation function (e.g. np.mean) can be provided.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    date_col : str
        Column name containing dates.
    value_col : str
        Column name with the numeric values.
    grain : Optional[str]
        Grain string (e.g., "day", "week", "month", "quarter"). If None, no resampling is done.
    agg_func : Optional[Callable]
        Aggregation function. Defaults to sum if None.
    
    Returns
    -------
    pd.DataFrame
        The resampled DataFrame.
    """
    if grain is None:
        return df.copy()
    
    freq_alias = _convert_grain_to_freq(grain)
    agg_func = agg_func or sum

    dff = df.copy()
    dff[date_col] = pd.to_datetime(dff[date_col])
    dff.set_index(date_col, inplace=True)
    dff = dff.resample(freq_alias).agg({value_col: agg_func})
    dff = dff.fillna(0).reset_index()
    return dff

# =============================================================================
# Main Forecast Functions
# =============================================================================

def simple_forecast(
    df: pd.DataFrame,
    value_col: str = "value",
    periods: int = 7,
    method: str = "ses",  # options: 'naive', 'ses', 'holtwinters', 'auto_arima'
    seasonal_periods: Optional[int] = None,
    date_col: Optional[str] = None,
    freq: Optional[str] = None,
    grain: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Produce a forecast using one of several methods (naive, SES, Holt-Winters, auto_arima).
    
    If a date column is provided, the forecast index will be date-based. Optionally, the data can
    be resampled based on an old-code style grain (e.g., day/week/month/quarter).

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data.
    value_col : str, default 'value'
        The numeric column to forecast.
    periods : int, default 7
        Number of future periods to forecast.
    method : str, default 'ses'
        Forecasting method: one of ['naive', 'ses', 'holtwinters', 'auto_arima'].
    seasonal_periods : int, optional
        Seasonal period used for 'holtwinters' or seasonal 'auto_arima'.
    date_col : str, optional
        Column name for dates. If not provided, a numeric index is used.
    freq : str, optional
        Frequency alias (e.g., "D", "W-MON") to generate future dates.
    grain : str, optional
        If provided, resample the data using this grain before forecasting.
    kwargs : dict
        Additional parameters passed to the underlying model fit.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["date", "forecast"].
    """
    dff = df.copy()

    # --- Fallback for missing date column ---
    if not date_col:
        # If no date column is provided, use a numeric index
        if method == "naive":
            last_val = dff[value_col].dropna().iloc[-1] if len(dff[value_col].dropna()) > 0 else 0.0
            future_idx = np.arange(len(dff[value_col]), len(dff[value_col]) + periods)
            return pd.DataFrame({"date": future_idx, "forecast": [last_val] * periods})
        # For other methods, if not enough data, fall back to naive
        if len(dff[value_col].dropna()) < 2:
            method = "naive"

    # --- Date processing and optional resampling ---
    if date_col:
        if grain is not None:
            dff = _maybe_resample_for_grain(dff, date_col, value_col, grain)
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff.sort_values(date_col, inplace=True)
        dff.set_index(date_col, inplace=True)
        if freq is not None:
            dff = dff.asfreq(freq)

    series = dff[value_col].dropna()
    if len(series) == 0:
        return pd.DataFrame(columns=["date", "forecast"])
    if len(series) < 2 and method != "naive":
        method = "naive"

    # --- Forecasting methods ---
    if method == "naive":
        last_val = series.iloc[-1]
        if date_col and freq and not series.index.empty:
            last_date = dff.index[-1]
            future_idx = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            fc_vals = [last_val] * periods
            fc_df = pd.DataFrame({"date": future_idx, "forecast": fc_vals})
        else:
            future_idx = np.arange(len(series), len(series) + periods)
            fc_df = pd.DataFrame({"date": future_idx, "forecast": [last_val] * periods})
    
    elif method == "ses":
        model = SimpleExpSmoothing(series, initialization_method="estimated")
        fit = model.fit(**kwargs)
        fc_vals = fit.forecast(periods)
    
    elif method == "holtwinters":
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
        seasonal = (seasonal_periods is not None and seasonal_periods > 1)
        default_arima_kwargs = dict(
            start_p=1,
            start_q=1,
            max_p=5,
            max_q=5,
            seasonal=seasonal,
            m=seasonal_periods if seasonal else 1,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True
        )
        for k, v in default_arima_kwargs.items():
            kwargs.setdefault(k, v)
        model = pmd_arima.auto_arima(series, **kwargs)
        fc_vals = model.predict(n_periods=periods)
    
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Build the forecast DataFrame ---
    if date_col and freq and not dff.empty:
        last_idx = dff.index[-1]
        future_idx = pd.date_range(last_idx, periods=periods + 1, freq=freq)[1:]
        fc_df = pd.DataFrame({"date": future_idx, "forecast": fc_vals}).reset_index(drop=True)
    elif date_col and grain and not dff.empty:
        freq_alias = _convert_grain_to_freq(grain)
        last_idx = dff.index[-1]
        future_idx = pd.date_range(last_idx, periods=periods + 1, freq=freq_alias)[1:]
        fc_df = pd.DataFrame({"date": future_idx, "forecast": fc_vals}).reset_index(drop=True)
    else:
        future_idx = np.arange(len(series), len(series) + periods)
        fc_df = pd.DataFrame({"date": future_idx, "forecast": fc_vals}).reset_index(drop=True)
    
    return fc_df

def forecast_upstream_metrics(
    drivers: Dict[str, pd.DataFrame],
    periods: int = 7,
    method: str = "ses",
    date_col: Optional[str] = None,
    freq: Optional[str] = None,
    grain: Optional[str] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    For each driver (given as a dict of driver_id -> DataFrame), produce a forecast using
    simple_forecast, returning a dictionary mapping driver_id to its forecast DataFrame.

    Parameters
    ----------
    drivers : Dict[str, pd.DataFrame]
        Dictionary of driver_id to DataFrame.
    periods : int, default 7
        Number of future periods to forecast.
    method : str, default 'ses'
        Forecasting method.
    date_col : str, optional
        Column name for dates.
    freq : str, optional
        Frequency alias for forecasting.
    grain : str, optional
        Grain for resampling (e.g., 'day', 'week', etc.).
    kwargs : dict
        Additional arguments for simple_forecast.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping driver_id to forecast DataFrame.
    """
    forecasts = {}
    for driver_id, df_driver in drivers.items():
        fc_df = simple_forecast(
            df_driver,
            value_col="value",
            periods=periods,
            method=method,
            date_col=date_col,
            freq=freq,
            grain=grain,
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
    grain: Optional[str] = None,
    **kwargs
) -> Dict[Any, pd.DataFrame]:
    """
    For each unique value in a dimension slice (specified by slice_col), forecast the
    corresponding time series individually using simple_forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    slice_col : str
        Column name to slice on.
    date_col : str, default "date"
        Date column.
    value_col : str, default "value"
        Value column.
    periods : int, default 7
        Forecast horizon.
    method : str, default 'ses'
        Forecasting method.
    freq : str, optional
        Frequency alias for forecasting.
    grain : str, optional
        Resampling grain.
    kwargs : dict
        Additional arguments to pass to simple_forecast.

    Returns
    -------
    Dict[Any, pd.DataFrame]
        Dictionary mapping each unique slice to its forecast DataFrame.
    """
    results = {}
    unique_slices = df[slice_col].dropna().unique()
    for slice_val in unique_slices:
        sub_df = df[df[slice_col] == slice_val].copy()
        fc = simple_forecast(
            sub_df,
            value_col=value_col,
            periods=periods,
            method=method,
            date_col=date_col,
            freq=freq,
            grain=grain,
            **kwargs
        )
        results[slice_val] = fc
    return results

def forecast_best_and_worst_case(
    forecast_df: pd.DataFrame, 
    buffer_pct: float = 10.0,
    forecast_col: str = "forecast"
) -> pd.DataFrame:
    """
    Compute best and worst case scenarios by applying a ± buffer percentage to the forecast.
    
    For example, if buffer_pct=10, best_case = forecast * 1.1 and worst_case = forecast * 0.9.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing a forecast column.
    buffer_pct : float, default 10.0
        Percentage buffer to apply.
    forecast_col : str, default "forecast"
        Column name with forecasted values.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'best_case' and 'worst_case' columns.
    """
    dff = forecast_df.copy()
    dff["best_case"] = dff[forecast_col] * (1.0 + buffer_pct / 100.0)
    dff["worst_case"] = dff[forecast_col] * (1.0 - buffer_pct / 100.0)
    return dff

def forecast_target_achievement(
    forecast_df: pd.DataFrame,
    target_df: pd.DataFrame,
    forecast_col: str = "forecast",
    target_col: str = "target",
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Compare forecasted values against targets, returning the absolute and percentage differences,
    and an on_track flag indicating whether the forecast meets or exceeds the target.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame with forecast data.
    target_df : pd.DataFrame
        DataFrame with target data.
    forecast_col : str, default "forecast"
        Column name for forecasted values.
    target_col : str, default "target"
        Column name for target values.
    date_col : str, default "date"
        Date column on which to merge.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for absolute difference, percentage difference,
        and an on_track flag.
    """
    dff = forecast_df[[date_col, forecast_col]].copy()
    tdf = target_df[[date_col, target_col]].copy()
    merged = pd.merge(dff, tdf, on=date_col, how="left")

    merged["abs_diff"] = merged[forecast_col] - merged[target_col]

    def safe_pct(row):
        if pd.isna(row[target_col]) or row[target_col] == 0:
            return None
        return (row["abs_diff"] / row[target_col]) * 100.0

    merged["pct_diff"] = merged.apply(safe_pct, axis=1)
    merged["on_track"] = merged.apply(
        lambda r: (not pd.isna(r[target_col])) and (r[forecast_col] >= r[target_col]),
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
    Compute forecast accuracy metrics (RMSE, MAE, MAPE) by merging actual and forecasted data.
    
    Parameters
    ----------
    actual_df : pd.DataFrame
        DataFrame with actual observed values.
    forecast_df : pd.DataFrame
        DataFrame with forecasted values.
    date_col : str, default "date"
        Date column on which to merge.
    actual_col : str, default "actual"
        Column name for actual values.
    forecast_col : str, default "forecast"
        Column name for forecasted values.
    
    Returns
    -------
    dict
        A dictionary containing RMSE, MAE, MAPE, and the number of observations used.
    """
    dfa = actual_df[[date_col, actual_col]].copy()
    dff = forecast_df[[date_col, forecast_col]].copy()
    merged = pd.merge(dfa, dff, on=date_col, how="inner").dropna(subset=[actual_col, forecast_col])
    if len(merged) == 0:
        return {"rmse": None, "mae": None, "mape": None, "n": 0}

    errors = merged[forecast_col] - merged[actual_col]
    abs_errors = errors.abs()
    rmse = np.sqrt((errors ** 2).mean())
    mae = abs_errors.mean()

    def safe_ape(row):
        if row[actual_col] == 0:
            return None
        return abs(row[forecast_col] - row[actual_col]) / abs(row[actual_col]) * 100.0

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
    Summarize forecast uncertainty by computing the average and maximum interval widths,
    and the mean relative interval width around the forecast.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing forecast intervals.
    lower_col : str, default "lower"
        Column name for the lower bound.
    upper_col : str, default "upper"
        Column name for the upper bound.
    forecast_col : str, default "forecast"
        Column name for the forecasted value.
    
    Returns
    -------
    dict
        Dictionary containing mean interval width, maximum interval width, and mean relative width.
    """
    if lower_col not in forecast_df.columns or upper_col not in forecast_df.columns:
        return {"error": f"No '{lower_col}' or '{upper_col}' in df.", "mean_interval_width": None}

    dff = forecast_df.dropna(subset=[lower_col, upper_col, forecast_col]).copy()
    if dff.empty:
        return {"mean_interval_width": None, "max_interval_width": None, "mean_relative_width": None}

    dff["interval_width"] = dff[upper_col] - dff[lower_col]
    mean_width = dff["interval_width"].mean()
    max_width = dff["interval_width"].max()

    def safe_rel(row):
        denom = abs(row[forecast_col])
        return row["interval_width"] / denom if denom > 1e-9 else None

    dff["relative_width"] = dff.apply(safe_rel, axis=1)
    rel_width = dff["relative_width"].mean(skipna=True)

    return {
        "mean_interval_width": float(mean_width),
        "max_interval_width": float(max_width),
        "mean_relative_width": float(rel_width) if pd.notna(rel_width) else None
    }

def decompose_forecast_drivers(
    driver_forecasts: Dict[str, pd.DataFrame],
    driver_coefs: Dict[str, float],
    base_intercept: float = 0.0
) -> pd.DataFrame:
    """
    Given forecasts for multiple drivers and their coefficients in a linear model,
    compute the partial contribution of each driver to the overall forecast.
    
    For a model y = intercept + sum_j(coef_j * driver_j), this function computes
    the contribution for each driver and the total forecast for each time point.
    
    Parameters
    ----------
    driver_forecasts : Dict[str, pd.DataFrame]
        Dictionary mapping driver identifiers to forecast DataFrames (each must contain a "date" and "forecast" column).
    driver_coefs : Dict[str, float]
        Dictionary mapping driver identifiers to their coefficients.
    base_intercept : float, default 0.0
        The intercept term in the model.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with a row per date, including each driver's contribution and the total forecast.
    """
    all_dates = set()
    for df_fc in driver_forecasts.values():
        all_dates.update(df_fc["date"].unique())
    all_dates = sorted(all_dates)

    rows = []
    for d in all_dates:
        row_data = {"date": d}
        total = base_intercept
        for drv, df_fc in driver_forecasts.items():
            sub = df_fc.loc[df_fc["date"] == d, "forecast"]
            drv_fc = sub.iloc[0] if len(sub) == 1 else np.nan
            coef = driver_coefs.get(drv, 0.0)
            contrib = coef * drv_fc
            row_data[f"driver_{drv}_contribution"] = contrib
            total += contrib
        row_data["total_forecast"] = total
        rows.append(row_data)

    return pd.DataFrame(rows)