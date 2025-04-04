# =============================================================================
# TrendAnalysis
#
# This module provides functions for analyzing trends in time series data:
# - Process control analysis (SPC)
# - Trend detection
# - Record high/low detection
# - Performance plateau detection
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - scipy.stats for linregress
# =============================================================================

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any

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
    Perform statistical process control (SPC) analysis on time series data.
    
    This function implements SPC by:
    1. Calculating a central line for each segment
    2. Computing control limits based on moving ranges
    3. Detecting signal patterns (points outside limits, runs, etc.)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing date and value columns
    date_col : str, default="date"
        Name of the date column
    value_col : str, default="value"
        Name of the value column
    half_average_point : int, default=9
        Half-width of window used for central line calculation
    consecutive_signal_threshold : int, default=5
        Number of consecutive signals that triggers recalculation
    min_data_points : int, default=10
        Minimum number of data points required for analysis
    moving_range_multiplier : float, default=2.66
        Multiplier for control limits (traditional SPC uses 2.66)
    consecutive_run_length : int, default=7
        Number of consecutive points in same direction to detect a trend
    long_run_total_length : int, default=12
        Window size for detecting long runs
    long_run_min_length : int, default=10
        Minimum count in long_run_total_length to trigger signal
    short_run_total_length : int, default=4
        Window size for detecting short runs
    short_run_min_length : int, default=3
        Minimum count in short_run_total_length to trigger signal
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - central_line: Expected process mean at each point
        - ucl: Upper control limit
        - lcl: Lower control limit
        - slope: Local trend slope
        - slope_change: Percentage change in slope
        - trend_signal_detected: Boolean flag for signals
    """
    # Input validation
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"DataFrame must contain {date_col} and {value_col}")

    # Create a copy to avoid modifying the original
    dff = df.copy()
    dff[date_col] = pd.to_datetime(dff[date_col])
    dff.sort_values(date_col, inplace=True)
    dff.reset_index(drop=True, inplace=True)

    # For small datasets, return DataFrame with empty control columns
    if len(dff) < min_data_points:
        dff["central_line"] = np.nan
        dff["ucl"] = np.nan
        dff["lcl"] = np.nan
        dff["slope"] = np.nan
        dff["slope_change"] = np.nan
        dff["trend_signal_detected"] = False
        return dff

    n_points = len(dff)
    central_line_array = [None] * n_points
    slope_array = [None] * n_points
    ucl_array = [None] * n_points
    lcl_array = [None] * n_points
    signal_array = [False] * n_points

    # Process data in segments
    start_idx = 0
    while start_idx < n_points:
        # Define segment end
        end_idx = min(start_idx + half_average_point * 2, n_points)
        seg_length = end_idx - start_idx
        
        if seg_length < 2:
            break
            
        # Compute center line and slope for this segment
        segment_center, segment_slope = _compute_segment_center_line(
            dff, start_idx, end_idx, half_average_point, value_col
        )
        
        # Store center line and slope values
        for i in range(seg_length):
            idx = start_idx + i
            central_line_array[idx] = segment_center[i]
            slope_array[idx] = segment_slope
        
        # Calculate control limits from moving ranges
        segment_values = dff[value_col].iloc[start_idx:end_idx].reset_index(drop=True)
        avg_range = _average_moving_range(segment_values)
        
        for i in range(seg_length):
            idx = start_idx + i
            cl_val = central_line_array[idx]
            if cl_val is not None and not math.isnan(cl_val):
                ucl_array[idx] = cl_val + avg_range * moving_range_multiplier
                lcl_array[idx] = cl_val - avg_range * moving_range_multiplier
            else:
                ucl_array[idx] = np.nan
                lcl_array[idx] = np.nan

        # Detect signals in this segment
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
            short_run_min_length=short_run_min_length
        )
        
        # Mark signals
        for sidx in seg_signals:
            signal_array[sidx] = True
        
        # Check if we need to recalculate due to consecutive signals
        recalc_idx = _check_consecutive_signals(seg_signals, consecutive_signal_threshold)
        if recalc_idx is not None and recalc_idx < n_points:
            if recalc_idx >= n_points - 1:
                break
            start_idx = recalc_idx
        else:
            start_idx = end_idx

    # Calculate slope changes between consecutive points
    slope_change_array = [None] * n_points
    for i in range(1, n_points):
        s_now = slope_array[i]
        s_prev = slope_array[i-1]
        if s_now is not None and s_prev is not None:
            if abs(s_prev) < 1e-9:  # Avoid division by zero
                slope_change_array[i] = None
            else:
                slope_change_array[i] = (s_now - s_prev) / abs(s_prev) * 100.0
        else:
            slope_change_array[i] = None

    # Add results to the DataFrame
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
    value_col: str
) -> Tuple[List[float], float]:
    """
    Calculate the center line and slope for a segment of data.
    
    This helper function creates a local trend line by:
    1. Computing averages of first and last sections in the segment
    2. Computing the slope between these averages
    3. Generating a complete center line based on this slope
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    start_idx : int
        Starting index for this segment
    end_idx : int
        Ending index for this segment
    half_average_point : int
        Number of points to use for averaging at each end
    value_col : str
        Name of the value column
        
    Returns
    -------
    Tuple[List[float], float]
        (center_line array, slope value)
    """
    # Extract segment values
    seg = df[value_col].iloc[start_idx:end_idx].reset_index(drop=True)
    n = len(seg)
    
    if n < 2:
        return ([None] * n, 0.0)
    
    # Adjust half point based on available data
    half_pt = min(half_average_point, n//2)
    
    # Calculate averages of first and last sections
    first_avg = seg.iloc[:half_pt].mean()
    second_avg = seg.iloc[-half_pt:].mean()
    
    # Calculate slope between these averages
    slope = (second_avg - first_avg) / float(half_pt) if half_pt > 0 else 0.0
    
    # Generate center line based on slope
    center_line = [None] * n
    mid_idx = half_pt // 2 if half_pt > 0 else 0
    
    if mid_idx >= n:
        # If segment is too small, use flat center line
        center_line = [seg.mean()] * n
        slope = 0.0
        return (center_line, slope)
    
    # Set middle point and extend in both directions
    center_line[mid_idx] = first_avg
    
    # Forward projection
    for i in range(mid_idx + 1, n):
        center_line[i] = center_line[i-1] + slope
    
    # Backward projection
    for i in range(mid_idx - 1, -1, -1):
        center_line[i] = center_line[i+1] - slope
    
    return (center_line, slope)


def _average_moving_range(values: pd.Series) -> float:
    """
    Calculate the average moving range for a series.
    
    The moving range is the absolute difference between consecutive points.
    This is used in SPC to calculate control limits.
    
    Parameters
    ----------
    values : pd.Series
        Series of values
        
    Returns
    -------
    float
        Average moving range or 0.0 if not enough data
    """
    # Calculate absolute differences between consecutive points
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
    short_run_min_length: int
) -> List[int]:
    """
    Detect SPC rule violations indicating a process signal.
    
    This function checks for:
    1. Points beyond control limits
    2. Consecutive points above/below center line
    3. Long runs with most points above/below center
    4. Short runs near control limits
    
    Parameters
    ----------
    df_segment : pd.DataFrame
        Segment of data to check
    offset : int
        Index offset for global array position
    central_line_array, ucl_array, lcl_array : List[Optional[float]]
        Arrays containing control values
    value_col : str
        Name of the value column
    consecutive_run_length, long_run_total_length, etc. : int
        SPC rule parameters
        
    Returns
    -------
    List[int]
        List of indices where signals were detected
    """
    n = len(df_segment)
    idx_start = offset
    
    # Create a local copy with control limits
    local_df = df_segment.reset_index(drop=True).copy()
    local_df["central_line"] = [central_line_array[idx_start+i] for i in range(n)]
    local_df["ucl"] = [ucl_array[idx_start+i] for i in range(n)]
    local_df["lcl"] = [lcl_array[idx_start+i] for i in range(n)]

    # Rule 1: Points outside control limits
    rule1 = (local_df[value_col] > local_df["ucl"]) | (local_df[value_col] < local_df["lcl"])

    # Rule 2: Consecutive points above/below center line
    above_center = local_df[value_col] > local_df["central_line"]
    below_center = local_df[value_col] < local_df["central_line"]
    
    # Rolling window to detect consecutive runs
    rule2_above = above_center.rolling(
        window=consecutive_run_length, 
        min_periods=consecutive_run_length
    ).sum() == consecutive_run_length
    
    rule2_below = below_center.rolling(
        window=consecutive_run_length, 
        min_periods=consecutive_run_length
    ).sum() == consecutive_run_length
    
    rule2 = rule2_above | rule2_below

    # Rule 3: Long runs with most points above/below center
    rolling_up = above_center.rolling(
        window=long_run_total_length, 
        min_periods=long_run_total_length
    ).sum()
    
    rolling_down = below_center.rolling(
        window=long_run_total_length, 
        min_periods=long_run_total_length
    ).sum()
    
    rule3 = (rolling_up >= long_run_min_length) | (rolling_down >= long_run_min_length)

    # Rule 4: Short runs near control limits (within 1 sigma)
    local_df["one_sigma_up"] = local_df["central_line"] + (local_df["ucl"] - local_df["central_line"]) / 3
    local_df["one_sigma_down"] = local_df["central_line"] - (local_df["central_line"] - local_df["lcl"]) / 3
    
    near_ucl = local_df[value_col] > local_df["one_sigma_up"]
    near_lcl = local_df[value_col] < local_df["one_sigma_down"]
    
    rule4_up = near_ucl.rolling(
        window=short_run_total_length, 
        min_periods=short_run_total_length
    ).sum() >= short_run_min_length
    
    rule4_down = near_lcl.rolling(
        window=short_run_total_length, 
        min_periods=short_run_total_length
    ).sum() >= short_run_min_length
    
    rule4 = rule4_up | rule4_down

    # Combine all rules
    combined = rule1 | rule2 | rule3 | rule4
    
    # Get indices of signals in global coordinates
    local_signal_idx = combined[combined.fillna(False)].index
    global_signal_idx = [idx_start + int(i) for i in local_signal_idx]
    
    return global_signal_idx


def _check_consecutive_signals(signal_idxes: List[int], threshold: int) -> Optional[int]:
    """
    Check if there are enough consecutive signals to trigger recalculation.
    
    Parameters
    ----------
    signal_idxes : List[int]
        List of indices where signals were detected
    threshold : int
        Number of consecutive signals required to trigger recalculation
        
    Returns
    -------
    Optional[int]
        Starting index for recalculation or None if not needed
    """
    if not signal_idxes:
        return None
        
    s = sorted(signal_idxes)
    consecutive_count = 1
    
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            consecutive_count += 1
            if consecutive_count >= threshold:
                return s[i - threshold + 1]
        else:
            consecutive_count = 1
            
    return None


def detect_record_high(df: pd.DataFrame, value_col: str = "value") -> bool:
    """
    Determine if the most recent value is the highest in the series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series
    value_col : str, default="value"
        Name of the value column
        
    Returns
    -------
    bool
        True if latest value is the highest, False otherwise
    """
    if df.empty:
        return False
        
    latest = df[value_col].iloc[-1]
    highest = df[value_col].max()
    
    # Use numpy.isclose to handle floating point comparison
    return (latest == highest or np.isclose(latest, highest))


def detect_record_low(df: pd.DataFrame, value_col: str = "value") -> bool:
    """
    Determine if the most recent value is the lowest in the series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series
    value_col : str, default="value"
        Name of the value column
        
    Returns
    -------
    bool
        True if latest value is the lowest, False otherwise
    """
    if df.empty:
        return False
        
    latest = df[value_col].iloc[-1]
    lowest = df[value_col].min()
    
    # Use numpy.isclose to handle floating point comparison
    return (latest == lowest or np.isclose(latest, lowest))


def detect_performance_plateau(
    df: pd.DataFrame, 
    value_col: str = "value", 
    tolerance: float = 0.01, 
    window: int = 7
) -> bool:
    """
    Detect if a series has plateaued (minimal variation) over a window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series
    value_col : str, default="value"
        Name of the value column
    tolerance : float, default=0.01
        Maximum relative variation allowed (max-min)/mean
    window : int, default=7
        Number of periods to check for plateau
        
    Returns
    -------
    bool
        True if series has plateaued, False otherwise
    """
    if len(df) < window:
        return False
        
    # Get the tail of the series
    sub = df[value_col].tail(window).dropna()
    
    if len(sub) < window:
        return False
        
    avg = sub.mean()
    
    # Avoid division by zero
    if abs(avg) < 1e-12:
        return False
        
    # Calculate relative range
    relative_range = (sub.max() - sub.min()) / abs(avg)
    
    return relative_range < tolerance


def analyze_metric_trend(
    df: pd.DataFrame, 
    value_col: str = "value", 
    date_col: str = "date", 
    window_size: int = 7
) -> Dict[str, Any]:
    """
    Analyze the overall trend of a metric time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    value_col : str, default="value"
        Name of the value column
    date_col : str, default="date"
        Name of the date column
    window_size : int, default=7
        Size of the window for rolling calculations
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'trend_direction': 'up', 'down', or 'stable'
        - 'trend_slope': Overall slope value
        - 'trend_confidence': R-squared value
        - 'recent_direction': Direction in most recent window
        - 'is_accelerating': Boolean indicating acceleration
        - 'is_plateaued': Boolean indicating plateau
    """
    if len(df) < 2:
        return {
            'trend_direction': 'insufficient_data',
            'trend_slope': None,
            'trend_confidence': None,
            'recent_direction': None,
            'is_accelerating': False,
            'is_plateaued': False
        }
    
    # Ensure data is properly sorted
    dff = df.copy()
    if date_col in dff.columns:
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff = dff.sort_values(date_col)
    
    # Calculate overall trend using linear regression
    y = dff[value_col].values
    x = np.arange(len(y))
    
    # Handle missing values
    mask = ~np.isnan(y)
    if sum(mask) < 2:
        return {
            'trend_direction': 'insufficient_data',
            'trend_slope': None,
            'trend_confidence': None,
            'recent_direction': None,
            'is_accelerating': False,
            'is_plateaued': False
        }
    
    slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
    
    # Determine overall trend direction
    if abs(slope) < 1e-6:
        trend_direction = 'stable'
    elif slope > 0:
        trend_direction = 'up'
    else:
        trend_direction = 'down'
    
    # Check if recent trend (last window_size points) is different
    if len(dff) >= window_size:
        recent = dff.tail(window_size)
        y_recent = recent[value_col].values
        x_recent = np.arange(len(y_recent))
        
        mask_recent = ~np.isnan(y_recent)
        if sum(mask_recent) >= 2:
            recent_slope, _, _, _, _ = linregress(x_recent[mask_recent], y_recent[mask_recent])
            
            if abs(recent_slope) < 1e-6:
                recent_direction = 'stable'
            elif recent_slope > 0:
                recent_direction = 'up'
            else:
                recent_direction = 'down'
                
            # Check for acceleration
            is_accelerating = abs(recent_slope) > abs(slope)
        else:
            recent_direction = None
            is_accelerating = False
    else:
        recent_direction = None
        is_accelerating = False
    
    # Check for plateau
    is_plateaued = detect_performance_plateau(
        dff, 
        value_col=value_col, 
        tolerance=0.01, 
        window=min(window_size, len(dff))
    )
    
    return {
        'trend_direction': trend_direction,
        'trend_slope': slope,
        'trend_confidence': r_value**2,  # R-squared
        'recent_direction': recent_direction,
        'is_accelerating': is_accelerating,
        'is_plateaued': is_plateaued
    }


def detect_anomalies(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    window_size: int = 7,
    z_threshold: float = 3.0,
    method: str = "combined"
) -> pd.DataFrame:
    """
    Detect anomalies in a time series using multiple methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    date_col : str, default="date"
        Name of the date column
    value_col : str, default="value"
        Name of the value column
    window_size : int, default=7
        Size of the rolling window for SPC methods
    z_threshold : float, default=3.0
        Z-score threshold for the variance method
    method : str, default="combined"
        Detection method: 'variance', 'spc', or 'combined'
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - 'rolling_mean': Rolling average
        - 'rolling_std': Rolling standard deviation
        - 'ucl', 'lcl': Control limits (if method includes 'spc')
        - 'is_anomaly_variance': Boolean flags for variance-based anomalies
        - 'is_anomaly_spc': Boolean flags for SPC-based anomalies
        - 'is_anomaly': Combined anomaly flag
    """
    if df.empty or len(df) < 2:
        return df.copy()
    
    # Ensure data is properly sorted
    dff = df.copy()
    if date_col in dff.columns:
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff = dff.sort_values(date_col)
    
    # Calculate rolling statistics
    dff['rolling_mean'] = dff[value_col].rolling(window=window_size, min_periods=2).mean()
    dff['rolling_std'] = dff[value_col].rolling(window=window_size, min_periods=2).std()
    
    # Variance method: z-score approach
    if method in ['variance', 'combined']:
        # Calculate z-scores
        dff['z_score'] = np.nan
        mask = ~dff['rolling_std'].isna() & (dff['rolling_std'] > 0)
        if mask.any():
            dff.loc[mask, 'z_score'] = (
                (dff.loc[mask, value_col] - dff.loc[mask, 'rolling_mean']) / 
                dff.loc[mask, 'rolling_std']
            )
        
        # Flag anomalies based on z-score threshold
        dff['is_anomaly_variance'] = np.abs(dff['z_score']) > z_threshold
        dff['is_anomaly_variance'] = dff['is_anomaly_variance'].fillna(False)
    else:
        dff['is_anomaly_variance'] = False
    
    # SPC method: control limits approach
    if method in ['spc', 'combined']:
        # Add SPC control limits
        dff['ucl'] = dff['rolling_mean'] + z_threshold * dff['rolling_std']
        dff['lcl'] = dff['rolling_mean'] - z_threshold * dff['rolling_std']
        
        # Flag points outside control limits
        dff['is_anomaly_spc'] = (
            (dff[value_col] > dff['ucl']) | 
            (dff[value_col] < dff['lcl'])
        )
        dff['is_anomaly_spc'] = dff['is_anomaly_spc'].fillna(False)
    else:
        dff['is_anomaly_spc'] = False
    
    # Combined anomaly flag
    dff['is_anomaly'] = dff['is_anomaly_variance'] | dff['is_anomaly_spc']
    
    return dff


from scipy.stats import linregress  # Import for analyze_metric_trend function