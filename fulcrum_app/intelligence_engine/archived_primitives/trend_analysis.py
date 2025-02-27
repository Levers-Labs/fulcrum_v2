"""
trend_analysis.py

This module provides functions for trend analysis using Statistical Process Control (SPC)
methods. It includes analysis routines that combine traditional process control with Wheeler-like
rules, as well as additional utilities to detect anomalies, record highs/lows, performance
plateaus, and trend changes.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

def process_control_analysis(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    half_average_point: int = 9,
    consecutive_signal_threshold: int = 5,
    min_data_points: int = 10,
    moving_range_multiplier: float = 2.66,
    consecutive_run_length: int = 7,
    long_run_total_length: int = 12,
    long_run_min_length: int = 10,
    short_run_total_length: int = 4,
    short_run_min_length: int = 3,
) -> pd.DataFrame:
    """
    Perform advanced SPC analysis on a time series by calculating center lines,
    control limits, slopes, and detecting signals based on multiple rules.

    Steps:
      1. Sort data by `date_col` and check if there is enough data.
      2. Iteratively compute half-averages over segments to derive a center line and slope.
      3. Calculate control limits (UCL/LCL) using the average moving range.
      4. Detect signals using SPC rules:
         - Points outside control limits.
         - A run of `consecutive_run_length` points all above or below the center line.
         - In a window of `long_run_total_length` points, at least `long_run_min_length` are above or below center.
         - In a window of `short_run_total_length` points, at least `short_run_min_length` are near UCL or LCL.
      5. If at least `consecutive_signal_threshold` consecutive signals are detected, recalc from that point.

    Returns a DataFrame with columns:
      - 'central_line'
      - 'ucl'
      - 'lcl'
      - 'slope'
      - 'slope_change'
      - 'trend_signal_detected': bool
    """
    dff = df.copy()
    if date_col not in dff.columns or value_col not in dff.columns:
        raise ValueError("DataFrame must contain both date_col and value_col.")

    # Convert and sort date column
    dff[date_col] = pd.to_datetime(dff[date_col])
    dff.sort_values(date_col, inplace=True, ascending=True)
    dff.reset_index(drop=True, inplace=True)

    if len(dff) < min_data_points:
        dff["central_line"] = float("nan")
        dff["ucl"] = float("nan")
        dff["lcl"] = float("nan")
        dff["slope"] = float("nan")
        dff["slope_change"] = float("nan")
        dff["trend_signal_detected"] = False
        return dff

    n_points = len(dff)
    # Initialize arrays for computed metrics
    central_line_array: List[Optional[float]] = [None] * n_points
    slope_array: List[Optional[float]] = [None] * n_points
    ucl_array: List[Optional[float]] = [None] * n_points
    lcl_array: List[Optional[float]] = [None] * n_points
    signal_array: List[bool] = [False] * n_points

    start_idx = 0
    while start_idx < n_points:
        # Define segment: use approximately half_average_point * 2 data points
        end_idx = min(start_idx + half_average_point * 2, n_points)
        seg_length = end_idx - start_idx
        if seg_length < 2:
            break

        # Compute center line and slope for the segment
        segment_center, segment_slope = _compute_segment_center_line(
            dff, start_idx, end_idx, half_average_point, value_col
        )

        # Populate center line and slope arrays for the segment
        for i in range(seg_length):
            idx = start_idx + i
            central_line_array[idx] = segment_center[i]
            slope_array[idx] = segment_slope

        # Compute UCL and LCL using the average moving range of the segment
        segment_values = dff[value_col].iloc[start_idx:end_idx].reset_index(drop=True)
        avgrange = _average_moving_range(segment_values)
        for i in range(seg_length):
            idx = start_idx + i
            cl_val = central_line_array[idx]
            if cl_val is not None and not math.isnan(cl_val):
                ucl_array[idx] = cl_val + avgrange * moving_range_multiplier
                lcl_array[idx] = cl_val - avgrange * moving_range_multiplier
            else:
                ucl_array[idx] = float("nan")
                lcl_array[idx] = float("nan")

        # Detect signals in the current segment using SPC rules
        seg_signals = _detect_spc_signals(
            df_segment=dff.iloc[start_idx:end_idx],
            offset=start_idx,
            central_line_array=central_line_array,
            ucl_array=ucl_array,
            lcl_array=lcl_array,
            value_col=value_col,
            consecutive_run_length=consecutive_run_length,
            long_run_total_length=long_run_total_length,
            long_run_min_length=long_run_min_length,
            short_run_total_length=short_run_total_length,
            short_run_min_length=short_run_min_length,
        )

        for sig_idx in seg_signals:
            signal_array[sig_idx] = True

        # Check if we have enough consecutive signals to trigger recalculation
        recalc_idx = _check_consecutive_signals(seg_signals, threshold=consecutive_signal_threshold)
        if recalc_idx is not None and recalc_idx < n_points:
            if recalc_idx >= n_points - 1:
                break
            start_idx = recalc_idx
        else:
            start_idx = end_idx

    # Compute slope_change: percentage change between consecutive slopes
    slope_change_array: List[Optional[float]] = [None] * n_points
    for i in range(1, n_points):
        s_now = slope_array[i]
        s_prev = slope_array[i - 1]
        if s_now is not None and s_prev is not None:
            if abs(s_prev) < 1e-9:
                slope_change_array[i] = None
            else:
                slope_change_array[i] = (s_now - s_prev) / abs(s_prev) * 100.0
        else:
            slope_change_array[i] = None

    # Assign computed arrays to the DataFrame
    dff["central_line"] = central_line_array
    dff["ucl"] = ucl_array
    dff["lcl"] = lcl_array
    dff["slope"] = slope_array
    dff["slope_change"] = slope_change_array
    dff["trend_signal_detected"] = signal_array

    return dff

def _compute_segment_center_line(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    half_average_point: int,
    value_col: str,
) -> Tuple[List[Optional[float]], float]:
    """
    Compute the center line of a segment using a half-average approach:
      - Average of the first `half_average_point` values.
      - Average of the last `half_average_point` values.
      - Slope is computed as the difference divided by half_average_point.
      - The center line is interpolated based on the computed slope.
      
    Returns a tuple (center_line_list, slope).
    """
    seg = df[value_col].iloc[start_idx:end_idx].reset_index(drop=True)
    n = len(seg)
    if n < 2:
        return ([None] * n, 0.0)
    half_pt = min(half_average_point, n // 2)
    first_avg = seg.iloc[:half_pt].mean()
    second_avg = seg.iloc[-half_pt:].mean()
    slope = (second_avg - first_avg) / float(half_pt) if half_pt > 0 else 0.0

    center_line: List[Optional[float]] = [None] * n
    # Use an interior point as the anchor
    mid_idx = half_pt // 2 if half_pt > 0 else 0
    if mid_idx >= n:
        center_line = [seg.mean()] * n
        slope = 0.0
        return (center_line, slope)

    center_line[mid_idx] = first_avg
    # Forward fill from the midpoint
    for i in range(mid_idx + 1, n):
        center_line[i] = center_line[i - 1] + slope
    # Backward fill from the midpoint
    for i in range(mid_idx - 1, -1, -1):
        center_line[i] = center_line[i + 1] - slope

    return (center_line, slope)

def _average_moving_range(values: pd.Series) -> float:
    """
    Compute the average moving range of consecutive points.
    Uses the typical approach: the mean of the absolute differences of consecutive values.
    """
    diffs = values.diff().abs().dropna()
    if len(diffs) == 0:
        return 0.0
    return diffs.mean()

def _detect_spc_signals(
    df_segment: pd.DataFrame,
    offset: int,
    central_line_array: List[Optional[float]],
    ucl_array: List[Optional[float]],
    lcl_array: List[Optional[float]],
    value_col: str,
    consecutive_run_length: int,
    long_run_total_length: int,
    long_run_min_length: int,
    short_run_total_length: int,
    short_run_min_length: int,
) -> List[int]:
    """
    Detect SPC signals within the subrange [offset, offset + len(df_segment)) using
    the following rules:
      1. Points outside control limits.
      2. A run of `consecutive_run_length` points all above or all below the center line.
      3. In a window of `long_run_total_length` points, at least `long_run_min_length` above or below center.
      4. In a window of `short_run_total_length` points, at least `short_run_min_length` points near the UCL or LCL.
    
    Returns a list of global indices where a signal is detected.
    """
    n = len(df_segment)
    idx_start = offset

    # Build a local DataFrame with the relevant metrics
    local_df = df_segment.reset_index(drop=True).copy()
    local_df["central_line"] = [central_line_array[idx_start + i] for i in range(n)]
    local_df["ucl"] = [ucl_array[idx_start + i] for i in range(n)]
    local_df["lcl"] = [lcl_array[idx_start + i] for i in range(n)]

    # Rule 1: Points outside control limits
    rule1 = (local_df[value_col] > local_df["ucl"]) | (local_df[value_col] < local_df["lcl"])

    # Rule 2: Run of consecutive_run_length points above or below center
    above_center = local_df[value_col] > local_df["central_line"]
    below_center = local_df[value_col] < local_df["central_line"]
    rule2_above = above_center.rolling(window=consecutive_run_length, min_periods=consecutive_run_length).sum() == consecutive_run_length
    rule2_below = below_center.rolling(window=consecutive_run_length, min_periods=consecutive_run_length).sum() == consecutive_run_length
    rule2 = rule2_above | rule2_below

    # Rule 3: In a long_run_total_length window, at least long_run_min_length above or below center
    rolling_up = above_center.rolling(window=long_run_total_length, min_periods=long_run_total_length).sum()
    rolling_down = below_center.rolling(window=long_run_total_length, min_periods=long_run_total_length).sum()
    rule3 = (rolling_up >= long_run_min_length) | (rolling_down >= long_run_min_length)

    # Rule 4: In a short_run_total_length window, at least short_run_min_length near UCL or LCL
    local_df["one_sigma_up"] = local_df["central_line"] + (local_df["ucl"] - local_df["central_line"]) / 3
    local_df["one_sigma_down"] = local_df["central_line"] - (local_df["central_line"] - local_df["lcl"]) / 3
    near_ucl = local_df[value_col] > local_df["one_sigma_up"]
    near_lcl = local_df[value_col] < local_df["one_sigma_down"]
    rule4_up = near_ucl.rolling(window=short_run_total_length, min_periods=short_run_total_length).sum() >= short_run_min_length
    rule4_down = near_lcl.rolling(window=short_run_total_length, min_periods=short_run_total_length).sum() >= short_run_min_length
    rule4 = rule4_up | rule4_down

    # Combine all rules: a point is flagged if any rule triggers
    combined_rule = rule1 | rule2 | rule3 | rule4

    local_signal_idx = combined_rule[combined_rule.fillna(False)].index
    global_signal_idx = [idx_start + int(i) for i in local_signal_idx]
    return global_signal_idx

def _check_consecutive_signals(signal_idxes: List[int], threshold: int) -> Optional[int]:
    """
    Check if there are >= threshold consecutive signals.
    Returns the global index of the start of the run if found; otherwise, returns None.
    """
    if not signal_idxes:
        return None
    signal_idxes = sorted(signal_idxes)
    consecutive_count = 1
    for i in range(1, len(signal_idxes)):
        if signal_idxes[i] == signal_idxes[i - 1] + 1:
            consecutive_count += 1
            if consecutive_count >= threshold:
                return signal_idxes[i - threshold + 1]
        else:
            consecutive_count = 1
    return None

def detect_spc_anomalies(
    df: pd.DataFrame, 
    value_col: str = "value",
    window: int = 7
) -> pd.DataFrame:
    """
    Flag points as anomalies if they fall outside ±3 sigma of the rolling mean.
    Adds columns: 'rolling_mean', 'rolling_std', 'ucl', 'lcl', 'spc_anomaly'.
    """
    dff = df.copy()
    dff["rolling_mean"] = dff[value_col].rolling(window).mean()
    dff["rolling_std"] = dff[value_col].rolling(window).std()
    dff["ucl"] = dff["rolling_mean"] + 3 * dff["rolling_std"]
    dff["lcl"] = dff["rolling_mean"] - 3 * dff["rolling_std"]

    dff["spc_anomaly"] = (
        (dff[value_col] > dff["ucl"]) | (dff[value_col] < dff["lcl"])
    ) & dff["rolling_mean"].notna()
    return dff

def detect_record_high(df: pd.DataFrame, value_col: str = "value") -> bool:
    """
    Check if the latest value is a record high (ties included).
    Returns True if the most recent value equals the maximum in the series.
    """
    if df.empty:
        return False
    latest = df[value_col].iloc[-1]
    highest = df[value_col].max()
    return np.isclose(latest, highest) or (latest == highest)

def detect_record_low(df: pd.DataFrame, value_col: str = "value") -> bool:
    """
    Check if the latest value is a record low.
    Returns True if the most recent value equals the minimum in the series.
    """
    if df.empty:
        return False
    latest = df[value_col].iloc[-1]
    lowest = df[value_col].min()
    return np.isclose(latest, lowest) or (latest == lowest)

def detect_performance_plateau(
    df: pd.DataFrame, 
    value_col: str = "value", 
    tolerance: float = 0.01, 
    window: int = 7
) -> bool:
    """
    Determine if the latest `window` data points are in a plateau (minimal variation).
    Defined as (max-min)/|mean| < tolerance.
    """
    if len(df) < window:
        return False
    sub = df[value_col].tail(window).dropna()
    if len(sub) < window:
        return False
    avg = sub.mean()
    if abs(avg) < 1e-12:
        return False
    return ((sub.max() - sub.min()) / abs(avg)) < tolerance

def analyze_metric_trend(
    df: pd.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    slope_threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Analyze the overall trend by computing the slope using SPC analysis.
    Interprets the most recent slope:
      - "up" if slope > slope_threshold,
      - "down" if slope < -slope_threshold,
      - "stable" otherwise.
    
    Returns a dict with keys "trend" and "slope".
    """
    spc_df = process_control_analysis(df, date_col=date_col, value_col=value_col)
    valid_slopes = spc_df["slope"].dropna()
    if valid_slopes.empty:
        return {"trend": "no_data", "slope": 0.0}
    recent_slope = valid_slopes.iloc[-1]
    if recent_slope > slope_threshold:
        trend = "up"
    elif recent_slope < -slope_threshold:
        trend = "down"
    else:
        trend = "stable"
    return {"trend": trend, "slope": recent_slope}

def detect_trend_changes(
    df: pd.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    slope_change_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Identify trend changes by calculating slope changes using SPC analysis.
    A trend change is flagged if the slope flips sign or if the absolute change in slope
    exceeds the `slope_change_threshold`.
    """
    spc_df = process_control_analysis(df, date_col=date_col, value_col=value_col).copy()
    spc_df["prev_slope"] = spc_df["slope"].shift(1)

    def has_trend_change(row: pd.Series) -> bool:
        if pd.isna(row["slope"]) or pd.isna(row["prev_slope"]):
            return False
        diff = row["slope"] - row["prev_slope"]
        sign_flip = (row["slope"] * row["prev_slope"] < 0)
        big_diff = abs(diff) > slope_change_threshold
        return sign_flip or big_diff

    spc_df["trend_change"] = spc_df.apply(has_trend_change, axis=1)
    return spc_df

def detect_new_trend_direction(
    df: pd.DataFrame,
    slope_col: str = "slope"
) -> str:
    """
    Determine if there is a new trend direction by comparing the last two slopes.
      - "new_upward" if slope flips from negative to positive.
      - "new_downward" if slope flips from positive to negative.
      - "no_change" otherwise.
    """
    if df.empty or slope_col not in df.columns or len(df) < 2:
        return "no_change"
    last_slope = df[slope_col].iloc[-1]
    prev_slope = df[slope_col].iloc[-2]
    if pd.isna(last_slope) or pd.isna(prev_slope):
        return "no_change"
    if prev_slope < 0 and last_slope > 0:
        return "new_upward"
    elif prev_slope > 0 and last_slope < 0:
        return "new_downward"
    else:
        return "no_change"