# =============================================================================
# DescriptiveStats
#
# This module provides functions for calculating descriptive statistics
# on numeric data, including central tendency, dispersion, shape metrics,
# and outlier detection.
#
# Dependencies:
#   - numpy as np
#   - pandas as pd
#   - scipy.stats for skew, kurtosis, median_abs_deviation
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from scipy.stats import skew, kurtosis, median_abs_deviation

def calculate_descriptive_stats(
    data: Union[pd.Series, pd.DataFrame, List, np.ndarray],
    value_col: Optional[str] = None,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    percentiles: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics for a dataset.
    
    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame, List, np.ndarray]
        The data to analyze. If DataFrame, must specify value_col.
    value_col : Optional[str], default=None
        Column name to analyze if data is a DataFrame.
    zscore_threshold : float, default=3.0
        Threshold for z-score outlier detection.
    iqr_multiplier : float, default=1.5
        Multiplier for IQR-based outlier detection.
    percentiles : Optional[List[float]], default=None
        Additional percentiles to compute, e.g. [10, 25, 50, 75, 90].
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing various statistics:
        - count: Total number of non-null observations
        - mean, median: Central tendency measures
        - min, max: Minimum and maximum values
        - std, variance: Dispersion measures
        - skew, kurtosis: Shape measures
        - iqr: Interquartile range
        - mad: Median absolute deviation
        - cv: Coefficient of variation (std/mean)
        - outlier_count_z: Number of outliers by z-score method
        - outlier_count_iqr: Number of outliers by IQR method
        - mode: List of mode values
        - p{x}: Percentile values if requested
        
    Examples
    --------
    >>> calculate_descriptive_stats([10, 15, 20, 25, 30])
    {'count': 5, 'mean': 20.0, 'median': 20.0, ...}
    
    >>> df = pd.DataFrame({'values': [10, 15, 20, 25, 30]})
    >>> calculate_descriptive_stats(df, value_col='values')
    {'count': 5, 'mean': 20.0, 'median': 20.0, ...}
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        if value_col is None:
            raise ValueError("Must specify 'value_col' when input is a DataFrame")
        series = pd.Series(data[value_col])
    elif isinstance(data, pd.Series):
        series = data
    else:
        series = pd.Series(data)
    
    # Remove NaNs for calculations
    s = series.dropna()
    total_count = len(s)
    
    if total_count == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "variance": None,
            "skew": None,
            "kurtosis": None,
            "iqr": None,
            "mad": None,
            "cv": None,
            "outlier_count_z": 0,
            "outlier_count_iqr": 0,
            "mode": [],
        }

    # Set default percentiles if none provided
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    arr = s.to_numpy()
    
    # Basic statistics
    min_val = float(s.min())
    max_val = float(s.max())
    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std(ddof=1))  # ddof=1 for sample standard deviation
    variance_val = std_val**2

    # Shape statistics
    skew_val = float(skew(arr, bias=False))
    kurt_val = float(kurtosis(arr, bias=False))

    # Interquartile range
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    iqr_val = q75 - q25

    # Median absolute deviation
    mad_val = float(median_abs_deviation(arr, scale='normal'))
    
    # Coefficient of variation
    cv_val = None if mean_val == 0 else std_val / abs(mean_val)

    # Z-score based outlier detection
    outlier_count_z = 0
    if std_val > 1e-12:
        z_scores = np.abs((arr - mean_val) / std_val)
        outlier_count_z = int((z_scores > zscore_threshold).sum())

    # IQR-based outlier detection
    lower_fence = q25 - iqr_multiplier * iqr_val
    upper_fence = q75 + iqr_multiplier * iqr_val
    outlier_count_iqr = int(((arr < lower_fence) | (arr > upper_fence)).sum())

    # Mode calculation
    mode_series = s.mode()
    mode_list = mode_series.tolist()

    # Build result dictionary
    stats = {
        "count": total_count,
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "std": std_val,
        "variance": variance_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "iqr": iqr_val,
        "mad": mad_val,
        "cv": cv_val,
        "outlier_count_z": outlier_count_z,
        "outlier_count_iqr": outlier_count_iqr,
        "mode": mode_list,
    }

    # Add requested percentiles
    for p in percentiles:
        key = f"p{int(p)}"
        if key in stats:
            continue
        val = np.percentile(arr, p)
        stats[key] = float(val)

    return stats

def calculate_rolling_statistics(
    df: pd.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    window_size: int = 7,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate rolling statistics over a time window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data.
    value_col : str, default="value"
        Column name containing the numeric values.
    date_col : str, default="date"
        Column name containing dates, used for sorting.
    window_size : int, default=7
        Size of the rolling window.
    min_periods : Optional[int], default=None
        Minimum number of observations required to calculate statistics.
        If None, defaults to window_size.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - rolling_mean: Rolling mean over the specified window
        - rolling_std: Rolling standard deviation
        - rolling_median: Rolling median
        - rolling_min: Rolling minimum
        - rolling_max: Rolling maximum
        
    Notes
    -----
    Data is sorted by date_col before calculations.
    """
    # Validate inputs
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    
    # Work with a copy of the DataFrame
    result_df = df.copy()
    
    # Sort by date
    result_df[date_col] = pd.to_datetime(result_df[date_col])
    result_df.sort_values(by=date_col, inplace=True)
    
    # Set min_periods if not provided
    if min_periods is None:
        min_periods = window_size
    
    # Calculate rolling statistics
    rolling = result_df[value_col].rolling(window=window_size, min_periods=min_periods)
    
    result_df['rolling_mean'] = rolling.mean()
    result_df['rolling_std'] = rolling.std()
    result_df['rolling_median'] = rolling.median()
    result_df['rolling_min'] = rolling.min()
    result_df['rolling_max'] = rolling.max()
    
    return result_df

def detect_outliers(
    data: Union[pd.Series, pd.DataFrame, List, np.ndarray],
    value_col: Optional[str] = None,
    method: str = "zscore",
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in a dataset using different methods.
    
    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame, List, np.ndarray]
        The data to analyze. If DataFrame, must specify value_col.
    value_col : Optional[str], default=None
        Column name to analyze if data is a DataFrame.
    method : str, default="zscore"
        Method to use for outlier detection:
        - "zscore": Z-score method
        - "iqr": Interquartile Range method
        - "mad": Median Absolute Deviation method
    threshold : float, default=3.0
        Threshold for outlier detection. For:
        - "zscore": Number of standard deviations
        - "iqr": Multiplier for IQR
        - "mad": Number of MAD units
        
    Returns
    -------
    np.ndarray
        Boolean array where True indicates an outlier
        
    Examples
    --------
    >>> detect_outliers([1, 2, 3, 100])
    array([False, False, False,  True])
    
    >>> detect_outliers([1, 2, 3, 100], method="iqr", threshold=1.5)
    array([False, False, False,  True])
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        if value_col is None:
            raise ValueError("Must specify 'value_col' when input is a DataFrame")
        values = data[value_col].dropna().values
    elif isinstance(data, pd.Series):
        values = data.dropna().values
    else:
        values = np.array(data)
        values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return np.array([])
    
    method = method.lower()
    
    if method == "zscore":
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        if std_val < 1e-12:  # Avoid division by zero
            return np.zeros(len(values), dtype=bool)
        z_scores = np.abs((values - mean_val) / std_val)
        return z_scores > threshold
        
    elif method == "iqr":
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        return (values < lower_bound) | (values > upper_bound)
        
    elif method == "mad":
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        mad = mad * 1.4826  # Scale factor for normal distribution
        return np.abs(values - median_val) / mad > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of 'zscore', 'iqr', or 'mad'.")