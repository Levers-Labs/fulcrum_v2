"""
seasonality_analysis.py

A comprehensive set of analytical functions for seasonality analysis, including:
  - Statsmodels-based methods for classifying seasonal patterns, detecting seasonal breaks, 
    and evaluating seasonality effects.
  - Funnel conversion seasonality analysis that aggregates funnel stage data and applies seasonal 
    decomposition to both raw counts and conversion rates.
  - Prophet-based seasonality forecasting implemented in a purely functional style (no classes). 
    Two Prophet models (additive and multiplicative) are fitted and compared by RMSE, and the best 
    model is used for forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import logging

from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import median_abs_deviation

# Attempt to import Prophet for forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# =============================================================================
# Statsmodels-based Seasonality Functions
# =============================================================================

def classify_seasonal_pattern(
    df: pd.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    candidate_periods: Optional[Dict[str, int]] = None,
    ac_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Identify a seasonal pattern (weekly, monthly, quarterly, yearly) by checking
    autocorrelations at candidate lags. If none exceed ac_threshold, returns "none".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series.
    value_col : str, default "value"
        Column name containing the numerical values.
    date_col : str, default "date"
        Column name containing the dates.
    candidate_periods : Optional[Dict[str, int]]
        Mapping of pattern names to lag periods. Defaults to:
          {"weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365}
    ac_threshold : float, default 0.5
        Minimum absolute autocorrelation required to consider a pattern.

    Returns
    -------
    Dict[str, Any]
        {
          "seasonal_pattern": str in ["weekly", "monthly", "quarterly", "yearly", "none"],
          "autocorrelations": { pattern_name: autocorr_value, ... }
        }
    """
    if candidate_periods is None:
        candidate_periods = {
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90,
            "yearly": 365
        }

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='raise')
    df_copy.sort_values(date_col, inplace=True)
    ts = df_copy.set_index(date_col)[value_col].asfreq('D')
    autocorrelations = {}
    best_pattern = "none"
    best_ac = 0.0

    for pattern, lag in candidate_periods.items():
        ac = ts.autocorr(lag=lag)
        autocorrelations[pattern] = ac
        if ac is not None and abs(ac) > abs(best_ac) and abs(ac) >= ac_threshold:
            best_ac = ac
            best_pattern = pattern

    return {
        "seasonal_pattern": best_pattern,
        "autocorrelations": autocorrelations
    }

def detect_seasonal_pattern_break(
    df: pd.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    seasonal_pattern: Optional[str] = None,
    deviation_threshold: float = 0.3,
    min_history: int = 30
) -> pd.DataFrame:
    """
    Flag observations that deviate significantly from historical seasonal averages.
    If seasonal_pattern is None, it is determined automatically using classify_seasonal_pattern.

    The function assigns each date a seasonal slot (e.g., weekday for weekly) and compares
    the value against the historical average for that slot. A boolean column "seasonal_break"
    is added to flag large deviations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data.
    value_col : str, default "value"
        Column name containing numerical values.
    date_col : str, default "date"
        Column name containing dates.
    seasonal_pattern : Optional[str], default None
        One of "weekly", "monthly", "quarterly", "yearly". If None, it is determined automatically.
    deviation_threshold : float, default 0.3
        Relative deviation threshold to flag a seasonal break.
    min_history : int, default 30
        Minimum number of observations required for analysis.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an added boolean column "seasonal_break".
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy.sort_values(date_col, inplace=True)

    if seasonal_pattern is None:
        info = classify_seasonal_pattern(df_copy, value_col=value_col, date_col=date_col)
        seasonal_pattern = info["seasonal_pattern"]
    if seasonal_pattern == "none":
        df_copy["seasonal_break"] = False
        return df_copy

    # Define seasonal slot based on identified pattern
    if seasonal_pattern == "weekly":
        df_copy["seasonal_slot"] = df_copy[date_col].dt.weekday
    elif seasonal_pattern == "monthly":
        df_copy["seasonal_slot"] = df_copy[date_col].dt.day
    elif seasonal_pattern == "quarterly":
        df_copy["seasonal_slot"] = (df_copy[date_col].dt.month - 1) // 3
    elif seasonal_pattern == "yearly":
        df_copy["seasonal_slot"] = df_copy[date_col].dt.dayofyear
    else:
        df_copy["seasonal_break"] = False
        return df_copy

    if len(df_copy) < min_history:
        df_copy["seasonal_break"] = False
        return df_copy

    # Compute historical average for each seasonal slot
    df_copy["seasonal_avg"] = df_copy.groupby("seasonal_slot")[value_col].transform("mean")

    def relative_deviation(row):
        if abs(row["seasonal_avg"]) < 1e-9:
            return 0.0
        return abs(row[value_col] - row["seasonal_avg"]) / abs(row["seasonal_avg"])

    df_copy["rel_deviation"] = df_copy.apply(relative_deviation, axis=1)
    df_copy["seasonal_break"] = df_copy["rel_deviation"] > deviation_threshold

    # Remove temporary columns
    df_copy.drop(columns=["seasonal_slot", "seasonal_avg", "rel_deviation"], inplace=True)
    return df_copy

def evaluate_seasonality_effect(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    period: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate the impact of seasonality by performing an STL decomposition on the series and
    comparing the seasonal component at the beginning vs. the end of the series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data.
    date_col : str, default "date"
        Column name for dates.
    value_col : str, default "value"
        Column name for the values.
    period : Optional[int]
        The period (e.g., 7 for weekly seasonality). Must be provided.

    Returns
    -------
    dict
        {
          "seasonal_diff": float,
          "total_diff": float,
          "fraction_of_total_diff": float
        }
        If period is not provided or data is insufficient, returns a dict with an "error" key.
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy.sort_values(date_col, inplace=True)

    if period is None:
        return {"error": "Must specify period (e.g. 7 for weekly)"}

    y = df_copy[value_col].to_numpy()
    if len(y) < period * 2:
        return {"error": "Not enough data for the specified period."}

    decomposition = seasonal_decompose(y, period=period, model='additive', extrapolate_trend='freq')
    seasonal = decomposition.seasonal
    val_t0 = seasonal[0]
    val_t1 = seasonal[-1]
    total_diff = y[-1] - y[0]
    seasonal_diff = val_t1 - val_t0
    frac = seasonal_diff / total_diff if abs(total_diff) > 1e-9 else 0.0

    return {
        "seasonal_diff": float(seasonal_diff),
        "total_diff": float(total_diff),
        "fraction_of_total_diff": float(frac)
    }

# =============================================================================
# Funnel Conversion Seasonality Analysis
# =============================================================================

def analyze_funnel_conversion_seasonality(
    df: pd.DataFrame,
    date_col: str = "date",
    funnel_stage_col: str = "stage",
    count_col: str = "count",
    baseline_stage: Optional[str] = None,
    frequency: str = "D",
    min_history: int = 30,
    decomposition_model: str = "additive"
) -> Dict[str, Any]:
    """
    Analyze seasonality in funnel conversion rates or raw counts.

    The function assumes that `df` contains events for various funnel stages (e.g., 'visit',
    'signup', 'purchase'). It aggregates counts per time unit (default daily) for each stage.

    If `baseline_stage` is provided (and exists in the data), conversion rates are computed for
    every other stage as: rate = count(stage) / count(baseline_stage). Otherwise, raw counts for
    each stage are analyzed.

    For each resulting time series, seasonal decomposition is attempted (if there is enough history)
    and the seasonal pattern is classified.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the date and funnel stage columns.
    date_col : str, default "date"
        Column name representing dates.
    funnel_stage_col : str, default "stage"
        Column name representing funnel stages.
    count_col : str, default "count"
        Column representing the count of events. If missing, each row is counted as 1.
    baseline_stage : Optional[str], default None
        Stage to use as the baseline for conversion rate calculation.
    frequency : str, default "D"
        Frequency for aggregating the time series (e.g., 'D' for daily).
    min_history : int, default 30
        Minimum number of observations required for seasonal decomposition.
    decomposition_model : str, default "additive"
        Type of seasonal decomposition to use ("additive" or "multiplicative").

    Returns
    -------
    Dict[str, Any]
        A dictionary keyed by funnel stage (or conversion rate series) where each value is a dict:
            - "ts": the aggregated time series (pd.Series)
            - "decomposition": the seasonal decomposition result (or None if not possible)
            - "seasonal_pattern": pattern classification (e.g., "weekly", "none", etc.)
            - "autocorrelations": autocorrelation values from the classification
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy.sort_values(date_col, inplace=True)

    # If count_col is missing, assign 1 to each row
    if count_col not in df_copy.columns:
        df_copy[count_col] = 1

    # Aggregate counts by date and funnel stage
    agg = df_copy.groupby([date_col, funnel_stage_col])[count_col].sum().reset_index()
    pivot = agg.pivot(index=date_col, columns=funnel_stage_col, values=count_col).fillna(0)
    pivot = pivot.asfreq(frequency).fillna(0)

    results = {}

    if baseline_stage and baseline_stage in pivot.columns:
        baseline_series = pivot[baseline_stage].replace(0, np.nan)
        for stage in pivot.columns:
            if stage == baseline_stage:
                continue
            conversion_rate = (pivot[stage] / baseline_series).fillna(0)
            if len(conversion_rate) >= min_history:
                try:
                    decomp = seasonal_decompose(
                        conversion_rate, period=7, model=decomposition_model, extrapolate_trend='freq'
                    )
                except Exception as e:
                    logger.warning(f"Decomposition failed for stage '{stage}': {e}")
                    decomp = None
            else:
                decomp = None

            classif = classify_seasonal_pattern(
                pd.DataFrame({"value": conversion_rate, "date": conversion_rate.index}),
                value_col="value", date_col="date"
            )
            results[stage] = {
                "ts": conversion_rate,
                "decomposition": decomp,
                "seasonal_pattern": classif["seasonal_pattern"],
                "autocorrelations": classif["autocorrelations"]
            }
    else:
        # Process each funnel stage separately (raw counts)
        for stage in pivot.columns:
            ts = pivot[stage]
            if len(ts) >= min_history:
                try:
                    decomp = seasonal_decompose(
                        ts, period=7, model=decomposition_model, extrapolate_trend='freq'
                    )
                except Exception as e:
                    logger.warning(f"Decomposition failed for stage '{stage}': {e}")
                    decomp = None
            else:
                decomp = None

            classif = classify_seasonal_pattern(
                pd.DataFrame({"value": ts, "date": ts.index}),
                value_col="value", date_col="date"
            )
            results[stage] = {
                "ts": ts,
                "decomposition": decomp,
                "seasonal_pattern": classif["seasonal_pattern"],
                "autocorrelations": classif["autocorrelations"]
            }

    return results


# =============================================================================
# Prophet-based Seasonality Analysis
# =============================================================================

def _compute_rmse(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Compute the root mean squared error between two arrays.
    """
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))

def _fit_prophet_model(
    df: pd.DataFrame,
    seasonality_mode: str,
    yearly_seasonality: bool,
    weekly_seasonality: bool,
    prophet_kwargs: Dict[str, Any]
) -> (Prophet, float):
    """
    Helper function to fit a Prophet model with the specified seasonality mode and compute its RMSE.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns "ds" and "y".
    seasonality_mode : str
        Either "additive" or "multiplicative".
    yearly_seasonality : bool
        Whether to include yearly seasonality.
    weekly_seasonality : bool
        Whether to include weekly seasonality.
    prophet_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to Prophet.

    Returns
    -------
    (Prophet, float)
        A tuple of the fitted Prophet model and its in-sample RMSE.
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        seasonality_mode=seasonality_mode,
        **prophet_kwargs
    )
    model.fit(df)
    forecast = model.predict(df)
    rmse = _compute_rmse(df["y"], forecast["yhat"])
    return model, rmse

def analyze_seasonality_prophet(
    df: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    periods: int = 0,
    **prophet_kwargs
) -> Dict[str, Any]:
    """
    Convenience function that fits Prophet models (additive and multiplicative) in a functional style,
    compares their in-sample RMSE, selects the best model, and produces a forecast.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns "date" and "value".
    yearly_seasonality : bool, default True
        Whether to include yearly seasonality.
    weekly_seasonality : bool, default True
        Whether to include weekly seasonality.
    periods : int, default 0
        Number of future periods to forecast. If 0, returns an in-sample forecast.
    prophet_kwargs : dict
        Additional keyword arguments to pass to Prophet.

    Returns
    -------
    Dict[str, Any]
        {
          "forecast": pd.DataFrame (the Prophet forecast),
          "is_additive": bool,
          "rmse_additive": float,
          "rmse_multiplicative": float
        }
    """
    if not PROPHET_AVAILABLE:
        return {"error": "Prophet not installed. Cannot perform prophet-based seasonality analysis."}

    # Prepare the DataFrame for Prophet: rename "date" -> "ds", "value" -> "y"
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy["value"] = df_copy["value"].astype(float)
    df_copy.rename(columns={"date": "ds", "value": "y"}, inplace=True)

    # Fit additive and multiplicative models
    model_add, rmse_add = _fit_prophet_model(df_copy, "additive", yearly_seasonality, weekly_seasonality, prophet_kwargs)
    model_mult, rmse_mult = _fit_prophet_model(df_copy, "multiplicative", yearly_seasonality, weekly_seasonality, prophet_kwargs)

    if rmse_add <= rmse_mult:
        best_model = model_add
        is_additive = True
    else:
        best_model = model_mult
        is_additive = False

    # Forecast: in-sample if periods == 0, or extend for the specified number of periods
    if periods == 0:
        forecast = best_model.predict(df_copy)
    else:
        future = best_model.make_future_dataframe(periods=periods)
        forecast = best_model.predict(future)

    return {
        "forecast": forecast,
        "is_additive": is_additive,
        "rmse_additive": rmse_add,
        "rmse_multiplicative": rmse_mult
    }