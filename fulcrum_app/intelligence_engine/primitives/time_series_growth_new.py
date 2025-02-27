# =============================================================================
# TimeSeriesGrowth
#
# This module provides functions for analyzing growth patterns in time series data:
# - Period-over-period growth calculations
# - Partial-to-date comparisons
# - Rolling averages and smoothing
# - Trend measurement via slope calculation
# - Cumulative growth and indexing
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - scipy.stats for linear regression
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Optional, Union, List, Dict, Tuple, Callable, Any

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _validate_date_sorted(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensure DataFrame is sorted by date and dates are datetime objects.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate and sort
    date_col : str
        Column name containing dates
        
    Returns
    -------
    pd.DataFrame
        Date-sorted DataFrame with datetime column
    
    Raises
    ------
    ValueError
        If date_col doesn't exist in the DataFrame
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    return result.sort_values(by=date_col)

# -----------------------------------------------------------------------------
# Main Analysis Functions
# -----------------------------------------------------------------------------

def calculate_pop_growth(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    value_col: str = 'value',
    periods: int = 1,
    fill_method: Optional[str] = None,
    annualize: bool = False,
    growth_col_name: str = 'pop_growth'
) -> pd.DataFrame:
    """
    Calculate period-over-period growth rates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    date_col : str, default='date'
        Column name containing dates
    value_col : str, default='value'
        Column name containing values to calculate growth for
    periods : int, default=1
        Number of periods to shift for growth calculation
    fill_method : Optional[str], default=None
        Method to fill NA values: None, 'ffill', 'bfill', or 'interpolate'
    annualize : bool, default=False
        Whether to annualize growth rates
    growth_col_name : str, default='pop_growth'
        Name for the growth rate column in output
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added growth rate column
        
    Notes
    -----
    - Growth rate is calculated as ((current - previous) / previous) * 100
    - When annualize=True, the formula is ((current/previous)^(365/days_diff) - 1) * 100
    - NaN values will appear for the first `periods` rows
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=5),
    ...     'value': [100, 110, 121, 133.1, 146.41]
    ... })
    >>> calculate_pop_growth(df)
       date       value  pop_growth
    0  2023-01-01  100.00        NaN
    1  2023-01-02  110.00      10.00
    2  2023-01-03  121.00      10.00
    3  2023-01-04  133.10      10.00
    4  2023-01-05  146.41      10.00
    """
    # Input validation
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    if periods <= 0:
        raise ValueError("periods must be a positive integer")
    
    # Ensure data is sorted by date
    df_sorted = _validate_date_sorted(df, date_col)
    
    # Calculate previous values
    df_sorted['prev_value'] = df_sorted[value_col].shift(periods)
    
    # Calculate growth rate
    if annualize and date_col in df_sorted.columns:
        # Annualized growth rate based on days between observations
        df_sorted['days_diff'] = (df_sorted[date_col] - df_sorted[date_col].shift(periods)).dt.days
        
        # Calculate annualized growth 
        df_sorted[growth_col_name] = np.where(
            (df_sorted['prev_value'] > 0) & (df_sorted['days_diff'] > 0),
            (np.power(df_sorted[value_col] / df_sorted['prev_value'], 365 / df_sorted['days_diff']) - 1) * 100,
            np.nan
        )
        df_sorted.drop('days_diff', axis=1, inplace=True)
    else:
        # Standard period-over-period growth rate
        df_sorted[growth_col_name] = np.where(
            df_sorted['prev_value'] != 0,
            ((df_sorted[value_col] - df_sorted['prev_value']) / df_sorted['prev_value']) * 100,
            np.nan
        )
    
    # Handle infinities
    df_sorted[growth_col_name] = df_sorted[growth_col_name].replace([np.inf, -np.inf], np.nan)
    
    # Fill NAs if requested
    if fill_method:
        if fill_method in ['ffill', 'bfill']:
            df_sorted[growth_col_name] = df_sorted[growth_col_name].fillna(method=fill_method)
        elif fill_method == 'interpolate':
            df_sorted[growth_col_name] = df_sorted[growth_col_name].interpolate()
        else:
            raise ValueError(f"Unsupported fill_method: {fill_method}. Use 'ffill', 'bfill', or 'interpolate'")
    
    # Drop temporary column
    df_sorted.drop('prev_value', axis=1, inplace=True)
    
    return df_sorted


def calculate_to_date_growth_rates(
    current_df: pd.DataFrame, 
    prior_df: pd.DataFrame,
    date_col: str = 'date',
    value_col: str = 'value',
    aggregator: str = 'sum',
    partial_interval: Optional[str] = None
) -> Dict[str, float]:
    """
    Compare partial-to-date periods (e.g., MTD, YTD) between current and prior periods.
    
    Parameters
    ----------
    current_df : pd.DataFrame
        DataFrame containing current period data
    prior_df : pd.DataFrame
        DataFrame containing prior period data
    date_col : str, default='date'
        Column name containing dates
    value_col : str, default='value'
        Column name containing values 
    aggregator : str, default='sum'
        How to aggregate values: 'sum', 'mean', 'median', 'min', 'max'
    partial_interval : Optional[str], default=None
        Type of partial interval: 'MTD' (month-to-date), 'QTD' (quarter-to-date),
        'YTD' (year-to-date), or 'WTD' (week-to-date)
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'current_value': Aggregated value in current period
        - 'prior_value': Aggregated value in prior period
        - 'abs_diff': Absolute difference
        - 'growth_rate': Percentage growth rate
        
    Notes
    -----
    When partial_interval is specified, date filtering is applied to match
    the pattern (e.g., MTD will filter to same day of month in both periods)
    """
    # Input validation
    for df, name in [(current_df, 'current_df'), (prior_df, 'prior_df')]:
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in {name}")
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in {name}")
            
    # Check aggregator is valid
    valid_aggregators = {'sum', 'mean', 'median', 'min', 'max'}
    if aggregator not in valid_aggregators:
        raise ValueError(f"Invalid aggregator: {aggregator}. Must be one of {valid_aggregators}")
    
    # Create copies with datetime indices
    curr = current_df.copy()
    prior = prior_df.copy()
    curr[date_col] = pd.to_datetime(curr[date_col])
    prior[date_col] = pd.to_datetime(prior[date_col])
    
    # Apply partial interval filtering if specified
    if partial_interval:
        # Get latest date from current period
        latest_date = curr[date_col].max()
        
        if partial_interval == 'MTD':
            # Month-to-date: Filter both dataframes to same day of month
            curr = curr[curr[date_col].dt.day <= latest_date.day]
            prior = prior[prior[date_col].dt.day <= latest_date.day]
            
        elif partial_interval == 'QTD':
            # Quarter-to-date: Filter to same day within quarter
            quarter_start = pd.Timestamp(year=latest_date.year, 
                                        month=((latest_date.month-1)//3)*3+1, 
                                        day=1)
            days_into_quarter = (latest_date - quarter_start).days
            
            prior_dates = prior[date_col].unique()
            if len(prior_dates) > 0:
                prior_latest = max(prior_dates)
                prior_quarter_start = pd.Timestamp(year=prior_latest.year, 
                                                month=((prior_latest.month-1)//3)*3+1, 
                                                day=1)
                prior = prior[prior[date_col] <= (prior_quarter_start + pd.Timedelta(days=days_into_quarter))]
            
        elif partial_interval == 'YTD':
            # Year-to-date: Filter to same day of year
            curr = curr[curr[date_col].dt.dayofyear <= latest_date.dayofyear]
            prior = prior[prior[date_col].dt.dayofyear <= latest_date.dayofyear]
            
        elif partial_interval == 'WTD':
            # Week-to-date: Filter to same day of week
            curr = curr[curr[date_col].dt.dayofweek <= latest_date.dayofweek]
            prior = prior[prior[date_col].dt.dayofweek <= latest_date.dayofweek]
            
        else:
            raise ValueError(f"Unsupported partial_interval: {partial_interval}. "
                            "Must be one of 'MTD', 'QTD', 'YTD', or 'WTD'")
    
    # Apply aggregation
    agg_func = getattr(pd.Series, aggregator)
    curr_val = agg_func(curr[value_col])
    prior_val = agg_func(prior[value_col])
    
    # Calculate growth
    abs_diff = curr_val - prior_val
    
    if prior_val == 0:
        growth_rate = np.nan
    else:
        growth_rate = (abs_diff / prior_val) * 100
        
    return {
        'current_value': curr_val,
        'prior_value': prior_val,
        'abs_diff': abs_diff,
        'growth_rate': growth_rate
    }


def calculate_average_growth(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    value_col: str = 'value',
    method: str = 'arithmetic',
    periods: int = 1
) -> Dict[str, float]:
    """
    Compute the average growth rate over multiple consecutive periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    date_col : str, default='date'
        Column name containing dates
    value_col : str, default='value'
        Column name containing values
    method : str, default='arithmetic'
        Method for averaging: 'arithmetic', 'geometric', or 'cagr'
    periods : int, default=1
        Number of periods to shift for growth calculation
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'avg_growth_rate': Average growth rate (percentage)
        - 'start_value': First value in series
        - 'end_value': Last value in series
        - 'periods_count': Number of periods used
        
    Notes
    -----
    - 'arithmetic': Simple average of period-over-period growth rates
    - 'geometric': Compound average growth rate (CAGR)
    - 'cagr': Specialized CAGR that accounts for exact time differences
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=5),
    ...     'value': [100, 110, 121, 133.1, 146.41]
    ... })
    >>> calculate_average_growth(df)
    {'avg_growth_rate': 10.0, 'start_value': 100.0, 'end_value': 146.41, 'periods_count': 4}
    
    >>> calculate_average_growth(df, method='geometric')
    {'avg_growth_rate': 10.0, 'start_value': 100.0, 'end_value': 146.41, 'periods_count': 4}
    """
    # Input validation
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    if method not in ['arithmetic', 'geometric', 'cagr']:
        raise ValueError(f"Invalid method: {method}. Must be 'arithmetic', 'geometric', or 'cagr'")
    
    # Ensure data is sorted by date
    df_sorted = _validate_date_sorted(df, date_col)
    
    # Calculate growth rates
    if len(df_sorted) <= periods:
        return {
            'avg_growth_rate': np.nan,
            'start_value': df_sorted[value_col].iloc[0] if not df_sorted.empty else np.nan,
            'end_value': df_sorted[value_col].iloc[-1] if not df_sorted.empty else np.nan,
            'periods_count': max(0, len(df_sorted) - periods)
        }
    
    start_value = df_sorted[value_col].iloc[0]
    end_value = df_sorted[value_col].iloc[-1]
    
    if method == 'arithmetic':
        # Calculate period-over-period growth rates
        df_sorted['prev_value'] = df_sorted[value_col].shift(periods)
        df_sorted['pct_change'] = np.where(
            df_sorted['prev_value'] != 0,
            ((df_sorted[value_col] - df_sorted['prev_value']) / df_sorted['prev_value']) * 100,
            np.nan
        )
        
        # Average of growth rates (excluding NaNs)
        avg_growth = df_sorted['pct_change'].mean(skipna=True)
        
    elif method == 'geometric':
        # Compound Annual Growth Rate (CAGR) formula
        if start_value <= 0:
            avg_growth = np.nan
        else:
            n_periods = len(df_sorted) - 1
            avg_growth = ((end_value / start_value) ** (1 / n_periods) - 1) * 100
            
    elif method == 'cagr':
        # CAGR accounting for actual time differences
        if start_value <= 0:
            avg_growth = np.nan
        else:
            start_date = df_sorted[date_col].iloc[0]
            end_date = df_sorted[date_col].iloc[-1]
            years_diff = (end_date - start_date).days / 365.25
            
            if years_diff <= 0:
                avg_growth = np.nan
            else:
                avg_growth = ((end_value / start_value) ** (1 / years_diff) - 1) * 100
    
    return {
        'avg_growth_rate': avg_growth,
        'start_value': start_value,
        'end_value': end_value,
        'periods_count': len(df_sorted) - periods
    }


def calculate_rolling_averages(
    df: pd.DataFrame, 
    value_col: str = 'value', 
    windows: List[int] = [7, 28],
    min_periods: Optional[Dict[int, int]] = None,
    center: bool = False
) -> pd.DataFrame:
    """
    Create rolling means for smoothing out fluctuations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    value_col : str, default='value'
        Column name containing values
    windows : List[int], default=[7, 28]
        List of window sizes for rolling calculations
    min_periods : Optional[Dict[int, int]], default=None
        Dictionary mapping window size to minimum periods required
        If None, uses window size as minimum
    center : bool, default=False
        Whether to center the window
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added rolling average columns
        
    Examples
    --------
    >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> calculate_rolling_averages(df, windows=[3])
       value  rolling_avg_3
    0      1            NaN
    1      2            NaN
    2      3           2.00
    3      4           3.00
    4      5           4.00
    5      6           5.00
    6      7           6.00
    7      8           7.00
    8      9           8.00
    9     10           9.00
    """
    # Input validation
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    if not windows:
        raise ValueError("No window sizes specified")
    
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Set default min_periods if not provided
    if min_periods is None:
        min_periods = {w: w for w in windows}
    else:
        # Ensure all windows have min_periods
        for w in windows:
            if w not in min_periods:
                min_periods[w] = w
    
    # Calculate rolling averages for each window size
    for w in windows:
        col_name = f'rolling_avg_{w}'
        result_df[col_name] = result_df[value_col].rolling(
            window=w, 
            min_periods=min_periods[w], 
            center=center
        ).mean()
    
    return result_df


def calculate_slope_of_time_series(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    value_col: str = 'value',
    normalize: bool = False
) -> Dict[str, float]:
    """
    Fit a linear regression to find the overall trend slope.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    date_col : str, default='date'
        Column name containing dates
    value_col : str, default='value'
        Column name containing values
    normalize : bool, default=False
        Whether to normalize the slope as a percentage of the mean value
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing regression results:
        - 'slope': Slope coefficient (units per day or percentage per day if normalized)
        - 'intercept': Y-intercept value
        - 'r_value': Correlation coefficient
        - 'p_value': P-value for hypothesis test
        - 'std_err': Standard error
        - 'slope_per_day': Slope in units per day
        - 'slope_per_week': Slope in units per week
        - 'slope_per_month': Slope in units per month (30 days)
        - 'slope_per_year': Slope in units per year (365 days)
        
    Notes
    -----
    When normalize=True, the slope is expressed as percentage change per day
    relative to the mean value of the series.
    """
    # Input validation
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    
    # Ensure df is sorted by date
    df_sorted = _validate_date_sorted(df, date_col)
    
    # Create time index (days from first observation)
    first_date = df_sorted[date_col].min()
    df_sorted['time_index'] = (df_sorted[date_col] - first_date).dt.days
    
    # Get data without NaNs
    mask = ~df_sorted[['time_index', value_col]].isna().any(axis=1)
    if not mask.any() or len(df_sorted[mask]) < 2:
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r_value': np.nan,
            'p_value': np.nan,
            'std_err': np.nan,
            'slope_per_day': np.nan,
            'slope_per_week': np.nan,
            'slope_per_month': np.nan,
            'slope_per_year': np.nan
        }
    
    # Run linear regression
    x = df_sorted.loc[mask, 'time_index'].values
    y = df_sorted.loc[mask, value_col].values
    
    result = linregress(x, y)
    
    # Calculate normalized slope if requested
    slope_per_day = result.slope
    if normalize and np.mean(y) != 0:
        slope_per_day = (slope_per_day / np.mean(y)) * 100
    
    # Calculate different time scales
    slope_per_week = slope_per_day * 7
    slope_per_month = slope_per_day * 30
    slope_per_year = slope_per_day * 365
    
    return {
        'slope': result.slope,
        'intercept': result.intercept,
        'r_value': result.rvalue,
        'p_value': result.pvalue,
        'std_err': result.stderr,
        'slope_per_day': slope_per_day,
        'slope_per_week': slope_per_week,
        'slope_per_month': slope_per_month,
        'slope_per_year': slope_per_year
    }


def calculate_cumulative_growth(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    value_col: str = 'value',
    method: str = 'index',
    base_index: float = 100.0,
    starting_date: Optional[Union[str, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Transform a series into a cumulative index from a baseline.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    date_col : str, default='date'
        Column name containing dates
    value_col : str, default='value'
        Column name containing values
    method : str, default='index'
        Method to calculate growth:
        - 'index': Each value becomes an index relative to the first value
        - 'cumsum': Running sum of values
        - 'cumprod': Running product of (1 + growth rates)
    base_index : float, default=100.0
        Starting index value when method='index'
    starting_date : Optional[Union[str, pd.Timestamp]], default=None
        Date to use as baseline; if None, uses the first date
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with an added 'cumulative_growth' column
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=5),
    ...     'value': [100, 110, 121, 133.1, 146.41]
    ... })
    >>> calculate_cumulative_growth(df, method='index')
           date     value  cumulative_growth
    0 2023-01-01  100.00             100.00
    1 2023-01-02  110.00             110.00
    2 2023-01-03  121.00             121.00
    3 2023-01-04  133.10             133.10
    4 2023-01-05  146.41             146.41
    
    >>> calculate_cumulative_growth(df, method='cumsum')
           date     value  cumulative_growth
    0 2023-01-01  100.00             100.00
    1 2023-01-02  110.00             210.00
    2 2023-01-03  121.00             331.00
    3 2023-01-04  133.10             464.10
    4 2023-01-05  146.41             610.51
    """
    # Input validation
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    if method not in ['index', 'cumsum', 'cumprod']:
        raise ValueError(f"Invalid method: {method}. Must be 'index', 'cumsum', or 'cumprod'")
    
    # Ensure data is sorted by date
    df_sorted = _validate_date_sorted(df, date_col)
    
    # Filter to starting date if provided
    if starting_date is not None:
        starting_date = pd.to_datetime(starting_date)
        if starting_date not in df_sorted[date_col].values:
            raise ValueError(f"Starting date {starting_date} not found in data")
        df_sorted = df_sorted[df_sorted[date_col] >= starting_date].copy()
    
    if df_sorted.empty:
        return df_sorted.copy()
    
    # Get base value for indexing
    base_value = df_sorted[value_col].iloc[0]
    
    if method == 'index':
        # Calculate index relative to first value
        if base_value == 0:
            df_sorted['cumulative_growth'] = np.nan
        else:
            df_sorted['cumulative_growth'] = df_sorted[value_col] / base_value * base_index
            
    elif method == 'cumsum':
        # Running sum
        df_sorted['cumulative_growth'] = df_sorted[value_col].cumsum()
        
    elif method == 'cumprod':
        # Calculate growth rates
        df_sorted['growth_rate'] = df_sorted[value_col].pct_change().fillna(0) + 1
        
        # Calculate cumulative product of growth rates
        df_sorted['cumulative_growth'] = df_sorted['growth_rate'].cumprod() * base_value
        
        # Remove temporary column
        df_sorted.drop('growth_rate', axis=1, inplace=True)
    
    return df_sorted