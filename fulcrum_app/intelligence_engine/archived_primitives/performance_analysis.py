"""
performance_analysis.py

A comprehensive library for KPI and funnel performance analysis.
Includes functions to calculate Goal-vs-Actual differences, historical GvA, 
status classifications and changes, run durations, threshold proximity checks, 
required growth rates, and detailed funnel conversion analyses.
"""

import pandas as pd
import numpy as np
import math
from typing import Optional, List, Dict, Any

def calculate_metric_gva(
    actual: float, 
    target: float, 
    allow_negative_target: bool = False
) -> Dict[str, Optional[float]]:
    """
    Compute the difference between an actual metric value and its target.

    Parameters
    ----------
    actual : float
        The actual observed value.
    target : float
        The target value. Typically positive, but can be zero or negative
        if allow_negative_target is True.
    allow_negative_target : bool, default=False
        If True, negative or zero targets are allowed. When allowed,
        percentage difference is computed as:
            pct_diff = (actual - target) / abs(target) * 100.
        Otherwise, if target <= 0, pct_diff is returned as None.

    Returns
    -------
    dict
        {
            'abs_diff': actual - target,
            'pct_diff': percentage difference or None
        }
    """
    abs_diff = actual - target

    # When target is not valid for percentage calculations:
    if (target <= 0) and not allow_negative_target:
        return {"abs_diff": abs_diff, "pct_diff": None}

    # Handle target == 0 (allowed) separately to avoid division by zero.
    if target == 0:
        pct = 0.0 if actual == 0 else (float('inf') if actual > 0 else float('-inf'))
    else:
        pct = (abs_diff / abs(target)) * 100.0

    return {"abs_diff": abs_diff, "pct_diff": pct}

def calculate_historical_gva(
    df_actual: pd.DataFrame,
    df_target: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    allow_negative_target: bool = False
) -> pd.DataFrame:
    """
    Compute historical GvA (Goal vs. Actual) over a time series.
    Merges two DataFrames on a date column and computes for each date:
      - abs_gva = actual - target
      - pct_gva = (abs_gva / abs(target)) * 100 (if target > 0 or allowed negative)

    Parameters
    ----------
    df_actual : pd.DataFrame
        DataFrame with columns [date_col, value_col] for actuals.
    df_target : pd.DataFrame
        DataFrame with columns [date_col, value_col] for targets.
    date_col : str, default='date'
        Name of the datetime column used for merging.
    value_col : str, default='value'
        Name of the numeric column representing metric values.
    allow_negative_target : bool, default=False
        If True, negative or zero targets are allowed and pct_gva is computed
        relative to abs(target). Otherwise, pct_gva is set to None when target <= 0.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        [date, value_actual, value_target, abs_gva, pct_gva].
    """
    merged = pd.merge(
        df_actual[[date_col, value_col]],
        df_target[[date_col, value_col]],
        on=date_col,
        how="left",
        suffixes=("_actual", "_target")
    )

    # Compute absolute difference
    merged["abs_gva"] = merged[f"{value_col}_actual"] - merged[f"{value_col}_target"]

    # Use vectorized computation for pct_gva:
    tgt = merged[f"{value_col}_target"]
    act = merged[f"{value_col}_actual"]

    # Define a mask where pct_gva is undefined:
    invalid = (tgt <= 0) & (~allow_negative_target)
    # Avoid division by zero if allowed:
    safe_tgt = tgt.replace({0: np.nan})

    merged["pct_gva"] = np.where(
        invalid,
        np.nan,
        np.where(
            tgt == 0,
            np.where(act == 0, 0.0, np.where(act > 0, float('inf'), float('-inf'))),
            (merged["abs_gva"] / safe_tgt.abs()) * 100.0
        )
    )

    return merged

def classify_metric_status(
    row_val: float, 
    row_target: float, 
    threshold: float = 0.05, 
    allow_negative_target: bool = False, 
    status_if_no_target: str = "no_target"
) -> str:
    """
    Classify a metric as "on_track" or "off_track" based on a threshold.

    For positive targets:
        on_track if row_val >= row_target * (1 - threshold).
    For negative targets (when allowed):
        on_track if row_val <= row_target * (1 + threshold)
        (i.e. less deviation in the negative direction).

    Parameters
    ----------
    row_val : float
        The actual observed value.
    row_target : float
        The target value.
    threshold : float, default=0.05
        Allowable deviation fraction (e.g., 5%).
    allow_negative_target : bool, default=False
        Whether to handle negative targets.
    status_if_no_target : str, default="no_target"
        Status to return when no valid target exists.

    Returns
    -------
    str
        One of "on_track", "off_track", or status_if_no_target.
    """
    if row_target is None or pd.isna(row_target):
        return status_if_no_target

    if (row_target <= 0) and not allow_negative_target:
        return status_if_no_target

    # Positive target logic.
    if row_target > 0:
        cutoff = row_target * (1 - threshold)
        return "on_track" if row_val >= cutoff else "off_track"
    else:
        # Negative target logic (allow_negative_target is True here).
        # For negative targets, being "on track" means not exceeding the negative threshold.
        cutoff = row_target * (1 + threshold)
        return "on_track" if row_val <= cutoff else "off_track"

def detect_status_changes(
    df: pd.DataFrame, 
    status_col: str = "status", 
    sort_by_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Identify rows where the status value changes from the previous row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column with status values.
    status_col : str, default='status'
        Name of the status column.
    sort_by_date : str, optional
        If provided, the DataFrame is sorted by this column before detection.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with two new columns:
          - 'prev_status': the previous row's status
          - 'status_flip': boolean flag indicating a status change.
    """
    out_df = df.copy()
    if sort_by_date:
        out_df.sort_values(sort_by_date, inplace=True)
    out_df["prev_status"] = out_df[status_col].shift(1)
    out_df["status_flip"] = (out_df[status_col] != out_df["prev_status"]) & out_df["prev_status"].notna()
    return out_df

def track_status_durations(
    df: pd.DataFrame, 
    status_col: str = "status",
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute consecutive runs of identical statuses. When a date column is provided,
    the actual duration (in days) is computed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a status column and optionally a date column.
    status_col : str, default='status'
        Column name holding status values.
    date_col : str or None, default=None
        If provided, must be datetime-like; durations are computed in days.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per consecutive status run. Columns include:
          - 'status': the status value
          - 'start_index', 'end_index': row indices for the run
          - 'run_length': number of rows in the run
          - 'start_date', 'end_date': boundary dates (if date_col provided)
          - 'duration_days': duration in days (if date_col provided)
    """
    # Ensure a clean, zero-indexed DataFrame
    df_clean = df.reset_index(drop=True).copy()
    # Identify runs by grouping on changes in status.
    df_clean["group"] = (df_clean[status_col] != df_clean[status_col].shift(1)).cumsum()
    runs = df_clean.groupby("group", as_index=False).agg(
        status=(status_col, "first"),
        start_index=("group", lambda x: x.index[0]),
        end_index=("group", lambda x: x.index[-1]),
        run_length=("group", "count")
    )
    if date_col:
        # Ensure date_col is datetime
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        runs["start_date"] = df_clean.groupby("group")[date_col].first().values
        runs["end_date"] = df_clean.groupby("group")[date_col].last().values
        runs["duration_days"] = (runs["end_date"] - runs["start_date"]).dt.days + 1
    else:
        runs["start_date"] = None
        runs["end_date"] = None
        runs["duration_days"] = None
    return runs.drop(columns="group")

def monitor_threshold_proximity(
    val: float, 
    target: float, 
    margin: float = 0.05,
    allow_negative_target: bool = False
) -> bool:
    """
    Check if 'val' is within +/- margin fraction of 'target'.

    Parameters
    ----------
    val : float
        The observed value.
    target : float
        The target value.
    margin : float, default=0.05
        Fractional margin (e.g., 0.05 for 5%).
    allow_negative_target : bool, default=False
        If False and target <= 0, returns False. Otherwise, computes using abs(target).

    Returns
    -------
    bool
        True if |val - target|/|target| <= margin; False otherwise.
    """
    if (target <= 0) and not allow_negative_target:
        return False

    # Special-case when target is zero and allowed.
    if target == 0:
        return val == 0

    return abs(val - target) / abs(target) <= margin

def calculate_required_growth(
    current_value: float, 
    target_value: float, 
    periods_left: int,
    allow_negative: bool = False
) -> Optional[float]:
    """
    Determine the compound per-period growth rate needed to reach target_value from current_value
    over a specified number of periods.

    For positive current and target values:
        rate = (target_value / current_value)^(1/periods_left) - 1.
    When negative values are allowed, the function attempts a ratio-based approach.

    Parameters
    ----------
    current_value : float
        Current metric value.
    target_value : float
        Desired metric value after periods_left.
    periods_left : int
        Number of periods over which growth occurs.
    allow_negative : bool, default=False
        Whether to allow negative or zero values in the computation.

    Returns
    -------
    float or None
        The per-period compound growth rate (e.g., 0.02 for 2% growth), or None if not feasible.
    """
    if periods_left <= 0:
        return None

    # Standard domain checks for positive values.
    if not allow_negative:
        if current_value <= 0 or target_value <= 0:
            return None
    else:
        # For negative-to-negative growth, work in absolute terms.
        if current_value < 0 and target_value < 0:
            current_value, target_value = abs(current_value), abs(target_value)
        # If signs differ or one value is zero, the calculation is undefined.
        elif current_value == 0 or target_value == 0 or (current_value * target_value < 0):
            return None

    ratio = target_value / current_value
    if ratio <= 0:
        return None

    rate = ratio ** (1.0 / periods_left) - 1.0
    return rate