# =============================================================================
# Forecasting
#
# This file includes primitives for time series forecasting:
# - Statistical forecasting (Naive, SES, Holt-Winters, ARIMA)
# - Upstream metric forecasting
# - Scenario generation (best/worst cases)
# - Forecast accuracy evaluation
# - Forecast uncertainty assessment
# - Driver-based decomposition of forecasts
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - statsmodels.tsa.holtwinters
#   - pmdarima for auto_arima
# =============================================================================

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Any, Optional, Union, Tuple
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Try to import pmdarima, but provide fallback if not available
try:
    from pmdarima import arima as pmd_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

def _convert_grain_to_freq(grain: str) -> str:
    """
    Convert a textual grain like 'day','daily','week','weekly','month','monthly'
    into a pandas frequency alias.
    
    Parameters
    ----------
    grain : str
        Time grain description
        
    Returns
    -------
    str
        Pandas frequency alias
        
    Raises
    ------
    ValueError
        If the grain is not supported
    """
    g = grain.lower()
    if g in ["day", "daily", "d"]:
        return "D"
    elif g in ["week", "weekly", "w"]:
        return "W-MON"  # Start week on Monday
    elif g in ["month", "monthly", "m"]:
        return "MS"  # Month start
    elif g in ["quarter", "quarterly", "q"]:
        return "QS"  # Quarter start
    elif g in ["year", "yearly", "annual", "y"]:
        return "YS"  # Year start
    else:
        raise ValueError(f"Unsupported grain '{grain}'. Use day, week, month, quarter, or year.")

def simple_forecast(
    df: pd.DataFrame,
    value_col: str = "value",
    periods: int = 7,
    method: str = "ses",
    seasonal_periods: Optional[int] = None,
    date_col: Optional[str] = None,
    freq: Optional[str] = None,
    grain: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Produce a forecast using one of various methods: naive, ses, holtwinters, or auto_arima.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    value_col : str, default="value"
        Column containing values to forecast
    periods : int, default=7
        Number of periods to forecast
    method : str, default="ses"
        Forecasting method: 'naive', 'ses', 'holtwinters', or 'auto_arima'
    seasonal_periods : Optional[int], default=None
        Number of periods in a seasonal cycle (e.g., 7 for weekly, 12 for monthly)
    date_col : Optional[str], default=None
        Column containing dates
    freq : Optional[str], default=None
        Pandas frequency string for resampling (e.g., 'D', 'W', 'M')
    grain : Optional[str], default=None
        Time grain description (e.g., 'day', 'week', 'month')
    **kwargs
        Additional parameters for the forecasting methods
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'forecast']
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=30),
    ...     'value': np.arange(30) + np.random.normal(0, 3, 30)
    ... })
    >>> result = simple_forecast(df, date_col='date', periods=7)
    """
    # Validate inputs
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in DataFrame")
    if date_col is not None and date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame")
        
    valid_methods = ["naive", "ses", "holtwinters", "auto_arima"]
    if method not in valid_methods:
        raise ValueError(f"method '{method}' not recognized. Use one of {valid_methods}")
    
    if method == "auto_arima" and not PMDARIMA_AVAILABLE:
        raise ImportError("pmdarima package is required for auto_arima method. Install with 'pip install pmdarima'.")
    
    # Create a copy of the DataFrame
    dff = df.copy()
    
    # Time series preparation
    if date_col is None:
        # No date column provided, use index as time
        series = dff[value_col].dropna()
        if len(series) < 2 and method != "naive":
            method = "naive"  # Fallback to naive if not enough data
    else:
        # Convert date column to datetime
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff.sort_values(date_col, inplace=True)
        dff.set_index(date_col, inplace=True)

        # If grain is provided, convert to frequency string
        if grain is not None:
            freq = _convert_grain_to_freq(grain)
            
        # Resample if frequency is specified
        if freq is not None:
            # Resample to regular frequency
            dff = dff[value_col].resample(freq).mean()
            series = dff.fillna(method='ffill')  # Forward fill gaps
        else:
            series = dff[value_col].dropna()
            
        if len(series) < 2 and method != "naive":
            method = "naive"  # Fallback to naive if not enough data
    
    # Execute forecasting method
    if method == "naive":
        # Naive forecast: use the last observed value
        last_val = series.iloc[-1]
        
        # Generate future dates if date_col and freq provided
        if date_col and freq and not series.index.empty:
            last_date = series.index[-1]
            future_idx = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
            fc_vals = [last_val] * periods
            return pd.DataFrame({
                "date": future_idx, 
                "forecast": fc_vals
            })
        else:
            # Use numeric indices for future periods
            idx_future = np.arange(len(series), len(series) + periods)
            fc_vals = [last_val] * periods
            return pd.DataFrame({
                "date": idx_future, 
                "forecast": fc_vals
            })
    
    elif method == "ses":
        # Simple Exponential Smoothing
        smoothing_level = kwargs.get('smoothing_level', 0.2)
        # Default to estimated initialization for better results
        model = SimpleExpSmoothing(
            series, 
            initialization_method=kwargs.get('initialization_method', 'estimated')
        )
        fit = model.fit(smoothing_level=smoothing_level, **{k: v for k, v in kwargs.items() 
                                                           if k != 'smoothing_level'})
        fc_vals = fit.forecast(periods)
    
    elif method == "holtwinters":
        # Holt-Winters Exponential Smoothing
        trend = kwargs.pop("trend", None)
        seasonal = kwargs.pop("seasonal", None)
        
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method=kwargs.pop("initialization_method", "estimated")
        )
        fit = model.fit(**kwargs)
        fc_vals = fit.forecast(periods)
    
    elif method == "auto_arima":
        # ARIMA model selection with pmdarima
        do_seasonal = (seasonal_periods is not None and seasonal_periods > 1)
        
        default_arima_kwargs = dict(
            start_p=1, start_q=1, max_p=5, max_q=5,
            seasonal=do_seasonal,
            m=seasonal_periods if do_seasonal else 1,
            stepwise=True, 
            error_action='ignore', 
            suppress_warnings=True
        )
        
        # Update defaults with any provided kwargs
        for k, v in default_arima_kwargs.items():
            kwargs.setdefault(k, v)
            
        model = pmd_arima.auto_arima(series, **kwargs)
        fc_vals = model.predict(n_periods=periods)
    
    # Generate output DataFrame with dates
    if date_col:
        # Future dates based on frequency
        if freq is None:
            # Fallback to numeric indices
            idx_future = np.arange(len(series), len(series) + periods)
            return pd.DataFrame({
                "date": idx_future, 
                "forecast": fc_vals
            }).reset_index(drop=True)
        else:
            # Generate proper date range
            last_idx = series.index[-1]
            future_idx = pd.date_range(last_idx, periods=periods+1, freq=freq)[1:]
            return pd.DataFrame({
                "date": future_idx, 
                "forecast": fc_vals
            }).reset_index(drop=True)
    else:
        # Use numeric indices
        idx_future = np.arange(len(series), len(series) + periods)
        return pd.DataFrame({
            "date": idx_future, 
            "forecast": fc_vals
        }).reset_index(drop=True)


def forecast_upstream_metrics(
    drivers_dict: Dict[str, pd.DataFrame], 
    periods: int = 7, 
    method: str = "ses", 
    date_col: Optional[str] = None, 
    freq: Optional[str] = None, 
    grain: Optional[str] = None, 
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Forecast multiple upstream metrics (drivers) using the specified method.
    
    Parameters
    ----------
    drivers_dict : Dict[str, pd.DataFrame]
        Dictionary mapping driver_id to DataFrame containing its time series
    periods : int, default=7
        Number of periods to forecast
    method : str, default="ses"
        Forecasting method: 'naive', 'ses', 'holtwinters', or 'auto_arima'
    date_col : Optional[str], default=None
        Column containing dates
    freq : Optional[str], default=None
        Pandas frequency string for resampling
    grain : Optional[str], default=None
        Time grain description (e.g., 'day', 'week', 'month')
    **kwargs
        Additional parameters for the forecasting method
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping driver_id to forecast DataFrame
    """
    results = {}
    
    for driver_id, driver_df in drivers_dict.items():
        try:
            fc_df = simple_forecast(
                driver_df,
                value_col="value",
                periods=periods,
                method=method,
                date_col=date_col,
                freq=freq,
                grain=grain,
                **kwargs
            )
            results[driver_id] = fc_df
        except Exception as e:
            print(f"Error forecasting driver {driver_id}: {str(e)}")
            # Create an empty forecast DataFrame
            if date_col is not None and date_col in driver_df.columns:
                last_date = pd.to_datetime(driver_df[date_col]).max()
                if freq is None and grain is not None:
                    freq = _convert_grain_to_freq(grain)
                if freq is not None:
                    future_idx = pd.date_range(last_date, periods=periods+1, freq=freq)[1:]
                    results[driver_id] = pd.DataFrame({
                        "date": future_idx,
                        "forecast": [np.nan] * periods
                    })
            
    return results


def forecast_metric_dimensions(
    df: pd.DataFrame,
    slice_col: str = "slice",
    date_col: str = "date",
    value_col: str = "value",
    periods: int = 7,
    method: str = "ses",
    freq: Optional[str] = None,
    grain: Optional[str] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Forecast a metric for each dimension slice separately.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data with dimension slices
    slice_col : str, default="slice"
        Column containing dimension slices
    date_col : str, default="date"
        Column containing dates
    value_col : str, default="value"
        Column containing values to forecast
    periods : int, default=7
        Number of periods to forecast
    method : str, default="ses"
        Forecasting method: 'naive', 'ses', 'holtwinters', or 'auto_arima'
    freq : Optional[str], default=None
        Pandas frequency string for resampling
    grain : Optional[str], default=None
        Time grain description (e.g., 'day', 'week', 'month')
    **kwargs
        Additional parameters for the forecasting method
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping slice value to forecast DataFrame
    """
    # Validate inputs
    if slice_col not in df.columns:
        raise ValueError(f"slice_col '{slice_col}' not found in DataFrame")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in DataFrame")
    
    # Group data by slice
    slices = df[slice_col].unique()
    results = {}
    
    for slice_val in slices:
        slice_df = df[df[slice_col] == slice_val].copy()
        
        # Skip slices with insufficient data
        if len(slice_df) < 2:
            continue
        
        try:
            # Forecast for this slice
            fc = simple_forecast(
                slice_df,
                value_col=value_col,
                periods=periods,
                method=method,
                date_col=date_col,
                freq=freq,
                grain=grain,
                **kwargs
            )
            results[slice_val] = fc
        except Exception as e:
            print(f"Error forecasting slice '{slice_val}': {str(e)}")
            # Create an empty forecast DataFrame with NaN values
            if date_col is not None:
                last_date = pd.to_datetime(slice_df[date_col]).max()
                if freq is None and grain is not None:
                    freq = _convert_grain_to_freq(grain)
                if freq is not None:
                    future_idx = pd.date_range(last_date, periods=periods+1, freq=freq)[1:]
                    results[slice_val] = pd.DataFrame({
                        "date": future_idx,
                        "forecast": [np.nan] * periods
                    })
    
    return results


def forecast_best_and_worst_case(
    forecast_df: pd.DataFrame, 
    buffer_pct: float = 10.0, 
    forecast_col: str = "forecast",
    best_case_col: str = "best_case",
    worst_case_col: str = "worst_case"
) -> pd.DataFrame:
    """
    Create best and worst case scenarios around an existing forecast.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing the forecast
    buffer_pct : float, default=10.0
        Percentage buffer to apply (e.g., 10.0 for ±10%)
    forecast_col : str, default="forecast"
        Column containing the forecast values
    best_case_col : str, default="best_case"
        Name for the best case column
    worst_case_col : str, default="worst_case"
        Name for the worst case column
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional best_case and worst_case columns
    """
    # Validate inputs
    if forecast_col not in forecast_df.columns:
        raise ValueError(f"forecast_col '{forecast_col}' not found in DataFrame")
    
    # Create a copy of the DataFrame
    result_df = forecast_df.copy()
    
    # Ensure forecast column is numeric
    if not pd.api.types.is_numeric_dtype(result_df[forecast_col]):
        result_df[forecast_col] = pd.to_numeric(result_df[forecast_col], errors='coerce')
    
    # Calculate best and worst case values
    buffer_factor = buffer_pct / 100.0
    result_df[best_case_col] = result_df[forecast_col] * (1.0 + buffer_factor)
    result_df[worst_case_col] = result_df[forecast_col] * (1.0 - buffer_factor)
    
    return result_df


def forecast_target_achievement(
    forecast_df: pd.DataFrame, 
    target_df: pd.DataFrame, 
    forecast_col: str = "forecast", 
    target_col: str = "target", 
    date_col: str = "date",
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compare a forecast with targets to determine whether targets will be achieved.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing the forecast
    target_df : pd.DataFrame
        DataFrame containing the targets
    forecast_col : str, default="forecast"
        Column containing the forecast values
    target_col : str, default="target"
        Column containing the target values
    date_col : str, default="date"
        Column containing dates for joining
    threshold : float, default=0.05
        Threshold ratio for determining on_track status.
        Forecast is on_track if forecast >= target * (1-threshold)
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with additional columns:
        - 'abs_diff': Forecast - Target
        - 'pct_diff': (Forecast - Target) / |Target| * 100
        - 'on_track': Boolean indicating whether the forecast is on track
    """
    # Validate inputs
    if forecast_col not in forecast_df.columns:
        raise ValueError(f"forecast_col '{forecast_col}' not found in forecast_df")
    if target_col not in target_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in target_df")
    if date_col not in forecast_df.columns:
        raise ValueError(f"date_col '{date_col}' not found in forecast_df")
    if date_col not in target_df.columns:
        raise ValueError(f"date_col '{date_col}' not found in target_df")
    
    # Convert date columns to datetime for proper joining
    forecast_df_copy = forecast_df.copy()
    target_df_copy = target_df.copy()
    
    forecast_df_copy[date_col] = pd.to_datetime(forecast_df_copy[date_col])
    target_df_copy[date_col] = pd.to_datetime(target_df_copy[date_col])
    
    # Merge forecast and target
    merged = pd.merge(
        forecast_df_copy[[date_col, forecast_col]],
        target_df_copy[[date_col, target_col]],
        on=date_col, 
        how="left"
    )
    
    # Calculate differences
    merged["abs_diff"] = merged[forecast_col] - merged[target_col]
    
    # Calculate percentage difference safely
    def safe_pct(row):
        t = row[target_col]
        if pd.isna(t) or t == 0:
            return None
        return (row["abs_diff"] / abs(t)) * 100
    
    merged["pct_diff"] = merged.apply(safe_pct, axis=1)
    
    # Determine on_track status
    def is_on_track(row):
        target = row[target_col]
        forecast = row[forecast_col]
        
        if pd.isna(target) or target == 0:
            return None
        
        # For positive targets: on_track if forecast >= target * (1 - threshold)
        if target > 0:
            return forecast >= target * (1 - threshold)
        # For negative targets: on_track if forecast <= target * (1 + threshold)
        else:
            return forecast <= target * (1 + threshold)
    
    merged["on_track"] = merged.apply(is_on_track, axis=1)
    
    return merged


def calculate_forecast_accuracy(
    actual_df: pd.DataFrame, 
    forecast_df: pd.DataFrame, 
    date_col: str = "date", 
    actual_col: str = "actual", 
    forecast_col: str = "forecast"
) -> Dict[str, Any]:
    """
    Calculate accuracy metrics by comparing forecast with actual values.
    
    Parameters
    ----------
    actual_df : pd.DataFrame
        DataFrame containing actual values
    forecast_df : pd.DataFrame
        DataFrame containing forecast values
    date_col : str, default="date"
        Column containing dates for joining
    actual_col : str, default="actual"
        Column containing actual values
    forecast_col : str, default="forecast"
        Column containing forecast values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy metrics:
        - 'rmse': Root Mean Square Error
        - 'mae': Mean Absolute Error
        - 'mape': Mean Absolute Percentage Error
        - 'n': Number of observations used
        - 'mean_abs_error': Mean absolute error
        - 'max_abs_error': Maximum absolute error
        - 'bias': Mean forecast error (negative = under-forecast)
    """
    # Validate inputs
    if date_col not in actual_df.columns:
        raise ValueError(f"date_col '{date_col}' not found in actual_df")
    if date_col not in forecast_df.columns:
        raise ValueError(f"date_col '{date_col}' not found in forecast_df")
    if actual_col not in actual_df.columns:
        raise ValueError(f"actual_col '{actual_col}' not found in actual_df")
    if forecast_col not in forecast_df.columns:
        raise ValueError(f"forecast_col '{forecast_col}' not found in forecast_df")
    
    # Convert date columns to datetime for proper joining
    dfa = actual_df.copy()
    dff = forecast_df.copy()
    
    dfa[date_col] = pd.to_datetime(dfa[date_col])
    dff[date_col] = pd.to_datetime(dff[date_col])
    
    # Merge actual and forecast
    merged = pd.merge(
        dfa[[date_col, actual_col]], 
        dff[[date_col, forecast_col]], 
        on=date_col, 
        how="inner"
    ).dropna(subset=[actual_col, forecast_col])
    
    if merged.empty:
        return {
            "rmse": None, 
            "mae": None, 
            "mape": None, 
            "n": 0,
            "mean_abs_error": None,
            "max_abs_error": None,
            "bias": None
        }
    
    # Calculate error metrics
    errors = merged[forecast_col] - merged[actual_col]
    abs_errors = errors.abs()
    
    # Root Mean Square Error
    rmse = float(np.sqrt((errors**2).mean()))
    
    # Mean Absolute Error
    mae = float(abs_errors.mean())
    
    # Maximum Absolute Error
    max_abs_error = float(abs_errors.max())
    
    # Bias (mean forecast error)
    bias = float(errors.mean())
    
    # Mean Absolute Percentage Error
    def safe_ape(row):
        if row[actual_col] == 0:
            return None
        return abs(row[forecast_col] - row[actual_col]) / abs(row[actual_col]) * 100.0
    
    merged["ape"] = merged.apply(safe_ape, axis=1)
    mape = merged["ape"].mean(skipna=True)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": float(mape) if pd.notna(mape) else None,
        "n": len(merged),
        "mean_abs_error": mae,
        "max_abs_error": max_abs_error,
        "bias": bias
    }


def assess_forecast_uncertainty(
    forecast_df: pd.DataFrame, 
    lower_col: str = "lower", 
    upper_col: str = "upper", 
    forecast_col: str = "forecast"
) -> Dict[str, Any]:
    """
    Assess the uncertainty in a forecast based on prediction intervals.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing forecast and prediction intervals
    lower_col : str, default="lower"
        Column containing lower bound of prediction interval
    upper_col : str, default="upper"
        Column containing upper bound of prediction interval
    forecast_col : str, default="forecast"
        Column containing forecast values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing uncertainty metrics:
        - 'mean_interval_width': Average width of prediction intervals
        - 'max_interval_width': Maximum width of prediction intervals
        - 'mean_relative_width': Average width relative to forecast value
        - 'interval_growth': Growth in interval width over the forecast horizon
    """
    # Validate inputs
    required_cols = [lower_col, upper_col, forecast_col]
    missing_cols = [col for col in required_cols if col not in forecast_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Create a copy and drop rows with missing values
    dff = forecast_df.dropna(subset=required_cols).copy()
    
    if dff.empty:
        return {
            "mean_interval_width": None, 
            "max_interval_width": None, 
            "mean_relative_width": None,
            "interval_growth": None
        }
    
    # Calculate interval widths
    dff["interval_width"] = dff[upper_col] - dff[lower_col]
    mean_width = float(dff["interval_width"].mean())
    max_width = float(dff["interval_width"].max())
    
    # Calculate relative widths
    def safe_rel(row):
        denom = abs(row[forecast_col])
        return row["interval_width"] / denom if denom > 1e-9 else None
    
    dff["relative_width"] = dff.apply(safe_rel, axis=1)
    rel_width = dff["relative_width"].mean(skipna=True)
    
    # Calculate growth in interval width
    if len(dff) >= 2:
        first_width = dff["interval_width"].iloc[0]
        last_width = dff["interval_width"].iloc[-1]
        if first_width > 0:
            interval_growth = (last_width / first_width - 1) * 100
        else:
            interval_growth = None
    else:
        interval_growth = None
# Calculate growth in interval width
    if len(dff) >= 2:
        first_width = dff["interval_width"].iloc[0]
        last_width = dff["interval_width"].iloc[-1]
        if first_width > 0:
            interval_growth = (last_width / first_width - 1) * 100
        else:
            interval_growth = None
    else:
        interval_growth = None
    
    return {
        "mean_interval_width": float(mean_width),
        "max_interval_width": float(max_width),
        "mean_relative_width": float(rel_width) if pd.notna(rel_width) else None,
        "interval_growth": float(interval_growth) if interval_growth is not None else None
    }


def decompose_forecast_drivers(
    driver_forecasts: Dict[str, pd.DataFrame], 
    driver_coefs: Dict[str, float], 
    base_intercept: float = 0.0,
    date_col: str = "date",
    forecast_col: str = "forecast"
) -> pd.DataFrame:
    """
    Decompose a forecast into contributions from each driver.
    
    Parameters
    ----------
    driver_forecasts : Dict[str, pd.DataFrame]
        Dictionary mapping driver name to forecast DataFrame
    driver_coefs : Dict[str, float]
        Dictionary mapping driver name to coefficient
    base_intercept : float, default=0.0
        Intercept term in the model
    date_col : str, default="date"
        Column containing dates
    forecast_col : str, default="forecast"
        Column containing forecast values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'date': Date of forecast
        - 'total_forecast': Total forecast value
        - '{driver}_contribution': Contribution from each driver
    """
    # Validate inputs
    if not driver_forecasts:
        raise ValueError("driver_forecasts dictionary is empty")
    if not driver_coefs:
        raise ValueError("driver_coefs dictionary is empty")
    
    # Collect all dates from all driver forecasts
    all_dates = set()
    for driver, df_fc in driver_forecasts.items():
        if date_col not in df_fc.columns:
            raise ValueError(f"date_col '{date_col}' not found in forecast for driver '{driver}'")
        if forecast_col not in df_fc.columns:
            raise ValueError(f"forecast_col '{forecast_col}' not found in forecast for driver '{driver}'")
        
        all_dates.update(df_fc[date_col].unique())
    
    # Sort dates for consistent output
    all_dates = sorted(all_dates)
    
    if not all_dates:
        return pd.DataFrame(columns=["date", "total_forecast"])
    
    # Build rows with driver contributions
    rows = []
    for d in all_dates:
        row_data = {"date": d}
        total = base_intercept
        
        # Calculate contribution from each driver
        for driver, df_fc in driver_forecasts.items():
            coef = driver_coefs.get(driver, 0.0)
            
            # Find forecast value for this date
            matching_rows = df_fc[df_fc[date_col] == d]
            if not matching_rows.empty:
                forecast_val = matching_rows[forecast_col].iloc[0]
                if pd.isna(forecast_val):
                    forecast_val = 0.0
            else:
                forecast_val = 0.0
            
            # Calculate contribution
            contrib = forecast_val * coef
            row_data[f"{driver}_contribution"] = contrib
            total += contrib
        
        row_data["total_forecast"] = total
        rows.append(row_data)
    
    return pd.DataFrame(rows)


def aggregate_dimensional_forecast(
    dimensional_forecasts: Dict[str, pd.DataFrame],
    date_col: str = "date",
    forecast_col: str = "forecast",
    agg_method: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate forecasts from multiple dimension slices.
    
    Parameters
    ----------
    dimensional_forecasts : Dict[str, pd.DataFrame]
        Dictionary mapping slice value to forecast DataFrame
    date_col : str, default="date"
        Column containing dates
    forecast_col : str, default="forecast"
        Column containing forecast values
    agg_method : str, default="sum"
        Method to aggregate forecasts: 'sum', 'mean', 'min', 'max'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated forecast values
    """
    # Validate inputs
    if not dimensional_forecasts:
        return pd.DataFrame(columns=[date_col, forecast_col])
    
    valid_methods = ["sum", "mean", "min", "max"]
    if agg_method not in valid_methods:
        raise ValueError(f"agg_method '{agg_method}' not recognized. Use one of {valid_methods}")
    
    # Collect all dates from all dimension forecasts
    all_dates = set()
    for slice_val, df_fc in dimensional_forecasts.items():
        if date_col not in df_fc.columns:
            raise ValueError(f"date_col '{date_col}' not found in forecast for slice '{slice_val}'")
        if forecast_col not in df_fc.columns:
            raise ValueError(f"forecast_col '{forecast_col}' not found in forecast for slice '{slice_val}'")
        
        all_dates.update(df_fc[date_col].unique())
    
    # Sort dates for consistent output
    all_dates = sorted(all_dates)
    
    if not all_dates:
        return pd.DataFrame(columns=[date_col, forecast_col])
    
    # Create a DataFrame with one row per date
    result_df = pd.DataFrame({"date": all_dates})
    
    # Add a column for each dimension slice
    for slice_val, df_fc in dimensional_forecasts.items():
        slice_forecast = pd.DataFrame({
            date_col: df_fc[date_col],
            f"{slice_val}_forecast": df_fc[forecast_col]
        })
        
        # Merge with result DataFrame
        result_df = pd.merge(result_df, slice_forecast, on=date_col, how="left")
    
    # Get all slice forecast columns
    slice_cols = [col for col in result_df.columns if col.endswith("_forecast")]
    
    # Aggregate across slices
    if agg_method == "sum":
        result_df[forecast_col] = result_df[slice_cols].sum(axis=1)
    elif agg_method == "mean":
        result_df[forecast_col] = result_df[slice_cols].mean(axis=1)
    elif agg_method == "min":
        result_df[forecast_col] = result_df[slice_cols].min(axis=1)
    elif agg_method == "max":
        result_df[forecast_col] = result_df[slice_cols].max(axis=1)
    
    # Return only date and forecast columns
    return result_df[[date_col, forecast_col]]