# =============================================================================
# DataQuality
#
# This file includes primitives for checking data quality issues:
# missing data detection, suspicious spikes/outliers,
# overall data quality scoring, and other quality checks.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _pct_missing(series: pd.Series) -> float:
    """
    Calculate percentage of missing values in a series.
    
    Parameters
    ----------
    series : pd.Series
        Series to check for missing values
        
    Returns
    -------
    float
        Percentage of missing values (0-100)
    """
    return series.isna().mean() * 100

def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using z-score method.
    
    Parameters
    ----------
    series : pd.Series
        Series to check for outliers
    threshold : float, default=3.0
        Z-score threshold for outlier detection
        
    Returns
    -------
    pd.Series
        Boolean series where True indicates an outlier
    """
    if len(series) < 3:
        return pd.Series([False] * len(series), index=series.index)
        
    mean = series.mean()
    std = series.std()
    
    if std == 0 or pd.isna(std):
        return pd.Series([False] * len(series), index=series.index)
        
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold

# -----------------------------------------------------------------------------
# Main Analysis Functions
# -----------------------------------------------------------------------------

def check_missing_data(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None
) -> pd.Series:
    """
    Calculate percentage of missing or null observations in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    cols : Optional[List[str]], default=None
        Specific columns to check. If None, checks all columns.
        
    Returns
    -------
    pd.Series
        Each column's percent missing, indexed by column name
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4, 5],
    ...     'B': [10, None, None, 40, 50]
    ... })
    >>> check_missing_data(df)
    A    20.0
    B    40.0
    dtype: float64
    """
    if cols is not None:
        df = df[cols].copy()
        
    return df.isna().mean() * 100


def detect_data_spikes(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    value_col: str = 'value', 
    threshold: float = 200.0,
    use_pct_change: bool = True
) -> pd.DataFrame:
    """
    Identify suspicious spikes that may be data-ingestion errors.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    date_col : str, default='date'
        Column containing dates
    value_col : str, default='value'
        Column containing values to check
    threshold : float, default=200.0
        Percentage threshold for spike detection (e.g., 200% jump)
    use_pct_change : bool, default=True
        If True, uses pandas pct_change method; otherwise calculates manually
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'prev_value': Previous period's value
        - 'pct_change': Percentage change from previous value
        - 'suspicious_spike': Boolean flag for suspicious spikes
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=5),
    ...     'value': [100, 110, 500, 120, 130]
    ... })
    >>> result = detect_data_spikes(df)
    >>> # The row with value=500 should be flagged as a suspicious spike
    """
    # Input validation
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
        
    # Sort by date and create a copy
    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')
    df_sorted = df_sorted.sort_values(by=date_col)
    
    # Convert value column to numeric if needed
    if not pd.api.types.is_numeric_dtype(df_sorted[value_col]):
        df_sorted[value_col] = pd.to_numeric(df_sorted[value_col], errors='coerce')
    
    # Calculate previous values and percentage change
    df_sorted['prev_value'] = df_sorted[value_col].shift(1)
    
    if use_pct_change:
        df_sorted['pct_change'] = df_sorted[value_col].pct_change() * 100
    else:
        df_sorted['pct_change'] = np.where(
            df_sorted['prev_value'] != 0,
            (df_sorted[value_col] - df_sorted['prev_value']) / np.abs(df_sorted['prev_value']) * 100,
            np.nan
        )
    
    # Flag suspicious spikes
    df_sorted['suspicious_spike'] = np.abs(df_sorted['pct_change']) > threshold
    df_sorted['suspicious_spike'] = df_sorted['suspicious_spike'].fillna(False)
    
    return df_sorted


def score_data_quality(
    df: pd.DataFrame, 
    value_col: str = 'value',
    date_col: Optional[str] = None,
    check_outliers: bool = True,
    check_missing: bool = True,
    check_spikes: bool = True,
    spike_threshold: float = 200.0
) -> Dict[str, float]:
    """
    Compute an overall data quality score based on coverage, outliers, spikes, etc.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    value_col : str, default='value'
        Column containing values to check
    date_col : Optional[str], default=None
        Column containing dates, required if check_spikes=True
    check_outliers : bool, default=True
        Whether to check for outliers
    check_missing : bool, default=True
        Whether to check for missing values
    check_spikes : bool, default=True
        Whether to check for suspicious spikes
    spike_threshold : float, default=200.0
        Threshold for spike detection
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'quality_score': Overall quality score (0-1, higher is better)
        - 'missing_percent': Percentage of missing values
        - 'outlier_percent': Percentage of outliers
        - 'spike_percent': Percentage of suspicious spikes
        - 'row_count': Number of rows analyzed
        
    Notes
    -----
    The quality score is based on weighted penalties for missing values,
    outliers, and suspicious spikes. The formula is:
    quality_score = 1.0 - (missing_weight * missing_pct/100 + 
                           outlier_weight * outlier_pct/100 + 
                           spike_weight * spike_pct/100)
    """
    # Input validation
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    if check_spikes and date_col is None:
        raise ValueError("date_col must be provided when check_spikes=True")
    if check_spikes and date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    
    result = {
        'quality_score': 1.0,
        'missing_percent': 0.0,
        'outlier_percent': 0.0,
        'spike_percent': 0.0,
        'row_count': len(df)
    }
    
    # Weights for quality score calculation
    missing_weight = 0.4
    outlier_weight = 0.3
    spike_weight = 0.3
    
    # Check for missing values
    if check_missing:
        missing_pct = _pct_missing(df[value_col])
        result['missing_percent'] = missing_pct
    
    # Check for outliers
    if check_outliers and len(df) > 0:
        # Convert value column to numeric if needed
        value_series = pd.to_numeric(df[value_col], errors='coerce')
        outliers = _detect_outliers_zscore(value_series.dropna())
        outlier_count = outliers.sum() if len(outliers) > 0 else 0
        outlier_pct = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
        result['outlier_percent'] = outlier_pct
    
    # Check for suspicious spikes
    if check_spikes and len(df) > 1:
        spike_df = detect_data_spikes(df, date_col, value_col, spike_threshold)
        spike_count = spike_df['suspicious_spike'].sum()
        spike_pct = (spike_count / len(df)) * 100 if len(df) > 0 else 0
        result['spike_percent'] = spike_pct
    
    # Calculate overall quality score
    penalties = (
        missing_weight * result['missing_percent'] / 100 +
        outlier_weight * result['outlier_percent'] / 100 +
        spike_weight * result['spike_percent'] / 100
    )
    result['quality_score'] = max(0.0, min(1.0, 1.0 - penalties))
    
    return result


def diagnose_numeric_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    include_only: Optional[List[str]] = None,
    non_negative_cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Perform comprehensive data quality diagnostics on numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    exclude_cols : Optional[List[str]], default=None
        Columns to exclude from analysis
    include_only : Optional[List[str]], default=None
        Only analyze these columns (supersedes exclude_cols)
    non_negative_cols : Optional[List[str]], default=None
        Columns that should only contain non-negative values
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping column names to diagnostic results, including:
        - 'missing_percent': Percentage of missing values
        - 'non_numeric_percent': Percentage of non-numeric values
        - 'negative_percent': Percentage of negative values (for non_negative_cols)
        - 'zero_percent': Percentage of zero values
        - 'min': Minimum value
        - 'max': Maximum value
        - 'mean': Mean value
        - 'median': Median value
        - 'issues': List of detected issues
    """
    # Determine columns to analyze
    if include_only is not None:
        # Only analyze specified columns
        cols_to_analyze = [col for col in include_only if col in df.columns]
    else:
        # Analyze all columns except excluded ones
        exclude_cols = exclude_cols or []
        cols_to_analyze = [col for col in df.columns if col not in exclude_cols]
    
    # Columns that should be non-negative
    non_negative_cols = non_negative_cols or []
    
    results = {}
    
    for col in cols_to_analyze:
        col_result = {
            'missing_percent': 0.0,
            'non_numeric_percent': 0.0,
            'negative_percent': 0.0,
            'zero_percent': 0.0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'issues': []
        }
        
        # Check missing values
        missing_pct = _pct_missing(df[col])
        col_result['missing_percent'] = missing_pct
        if missing_pct > 0:
            col_result['issues'].append(f"{missing_pct:.2f}% missing values")
        
        # Convert to numeric and check non-numeric values
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        original_len = len(df)
        non_numeric_count = numeric_series.isna().sum() - df[col].isna().sum()
        non_numeric_pct = (non_numeric_count / original_len) * 100 if original_len > 0 else 0
        col_result['non_numeric_percent'] = non_numeric_pct
        if non_numeric_pct > 0:
            col_result['issues'].append(f"{non_numeric_pct:.2f}% non-numeric values")
        
        # Calculate statistics on non-missing numeric values
        valid_values = numeric_series.dropna()
        if len(valid_values) > 0:
            col_result['min'] = float(valid_values.min())
            col_result['max'] = float(valid_values.max())
            col_result['mean'] = float(valid_values.mean())
            col_result['median'] = float(valid_values.median())
            
            # Check for negative values if column should be non-negative
            if col in non_negative_cols:
                negative_count = (valid_values < 0).sum()
                negative_pct = (negative_count / original_len) * 100 if original_len > 0 else 0
                col_result['negative_percent'] = negative_pct
                if negative_pct > 0:
                    col_result['issues'].append(f"{negative_pct:.2f}% negative values")
            
            # Check for zero values
            zero_count = (valid_values == 0).sum()
            zero_pct = (zero_count / original_len) * 100 if original_len > 0 else 0
            col_result['zero_percent'] = zero_pct
            if zero_pct > 80:
                col_result['issues'].append(f"{zero_pct:.2f}% zero values")
        
        results[col] = col_result
    
    return results