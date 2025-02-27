# =============================================================================
# DimensionalAnalysis
#
# This module provides functions for analyzing metrics across different dimensions:
# - Slice aggregation and share calculations
# - Ranking and comparison of dimension slices
# - Composition changes and impact analysis
# - Concentration and distribution metrics
# - Key driver identification and attribution
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import pandas as pd
import numpy as np
import math
import itertools
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _safe_share(numerator: float, denominator: float) -> float:
    """
    Calculate share percentage safely, handling zero denominator.
    
    Parameters
    ----------
    numerator : float
        Value to calculate share for
    denominator : float
        Total value
        
    Returns
    -------
    float
        Percentage share (0.0 if denominator is zero)
    """
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return (numerator / denominator) * 100.0

def _safe_pct_diff(row: pd.Series, base_col: str) -> Optional[float]:
    """
    Calculate percentage difference safely given a row and a base value column.
    
    Parameters
    ----------
    row : pd.Series
        Row containing values to compare
    base_col : str
        Column name for the base value
        
    Returns
    -------
    Optional[float]
        Percentage difference or None if base value is zero
    """
    if row[base_col] == 0 or pd.isna(row[base_col]):
        return None
    return (row["abs_diff"] / abs(row[base_col])) * 100.0

def _safe_relative_change(eval_val: float, comp_val: float) -> Optional[float]:
    """
    Compute relative change safely, handling zero values.
    
    Parameters
    ----------
    eval_val : float
        Evaluation value (typically current period)
    comp_val : float
        Comparison value (typically prior period)
        
    Returns
    -------
    Optional[float]
        Relative change or None if comparison value is zero
    """
    if comp_val == 0 or pd.isna(comp_val):
        return None
    return (eval_val - comp_val) / abs(comp_val)

# =============================================================================
# Main Functions: Slice Metrics & Shares
# =============================================================================

def calculate_slice_metrics(
    df: pd.DataFrame, 
    slice_col: str, 
    value_col: str, 
    agg_func: Union[str, Callable] = 'sum',
    top_n: Optional[int] = None,
    other_label: str = "Other",
    dropna_slices: bool = True
) -> pd.DataFrame:
    """
    Group data by slice_col and aggregate value_col with the specified function.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing slice and value columns.
    slice_col : str
        Column name used for slicing/grouping.
    value_col : str
        Column name containing the metric values to aggregate.
    agg_func : Union[str, Callable], default='sum'
        Aggregation function: 'sum', 'mean', 'min', 'max', 'count', or a custom function.
    top_n : Optional[int], default=None
        If provided, keep only the top_n slices and combine the rest into 'Other'.
    other_label : str, default="Other"
        Label to use for the combined smaller slices when top_n is used.
    dropna_slices : bool, default=True
        Whether to exclude rows with NaN in slice_col.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [slice_col, "aggregated_value"] sorted by aggregated_value.
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "region": ["North", "South", "East", "West", "North"],
    ...     "sales": [100, 200, 50, 75, 150]
    ... })
    >>> calculate_slice_metrics(df, "region", "sales")
       region  aggregated_value
    0   North              250
    1   South              200
    2    West               75
    3    East               50
    
    >>> calculate_slice_metrics(df, "region", "sales", top_n=2)
       region  aggregated_value
    0   North              250
    1   South              200
    2   Other              125
    """
    # Input validation
    if slice_col not in df.columns:
        raise ValueError(f"slice_col '{slice_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in DataFrame")

    # Create a copy of the DataFrame
    dff = df.copy()
    
    # Drop rows with NaN in slice_col if requested
    if dropna_slices:
        dff = dff.dropna(subset=[slice_col])
    
    # Group by slice column and aggregate values
    if isinstance(agg_func, str):
        valid_agg_funcs = {'sum', 'mean', 'min', 'max', 'count', 'median'}
        if agg_func.lower() not in valid_agg_funcs:
            raise ValueError(f"agg_func '{agg_func}' not recognized. Use one of {valid_agg_funcs} or a callable")
        
        grouped = dff.groupby(slice_col)[value_col]
        try:
            result = grouped.agg(agg_func.lower()).reset_index(name='aggregated_value')
        except Exception as e:
            raise ValueError(f"Error applying aggregation function '{agg_func}': {e}")
    else:
        # Custom aggregation function
        try:
            result = dff.groupby(slice_col)[value_col].apply(agg_func).reset_index(name='aggregated_value')
        except Exception as e:
            raise ValueError(f"Error applying custom aggregation function: {e}")
    
    # Sort by aggregated value (descending)
    result.sort_values('aggregated_value', ascending=False, inplace=True, ignore_index=True)
    
    # Handle top_n if specified
    if top_n is not None and len(result) > top_n:
        top_slices = result.iloc[:top_n].copy()
        other_slices = result.iloc[top_n:]
        
        # Combine smaller slices into "Other"
        other_value = other_slices['aggregated_value'].sum()
        other_row = pd.DataFrame({
            slice_col: [other_label], 
            'aggregated_value': [other_value]
        })
        
        # Combine and sort
        result = pd.concat([top_slices, other_row], ignore_index=True)
        result.sort_values('aggregated_value', ascending=False, inplace=True, ignore_index=True)
    
    return result


def compute_slice_shares(
    agg_df: pd.DataFrame,
    slice_col: str,
    val_col: str = "aggregated_value",
    share_col_name: str = "share_pct"
) -> pd.DataFrame:
    """
    Calculate each slice's percentage share of the total.
    
    Parameters
    ----------
    agg_df : pd.DataFrame
        DataFrame containing aggregated values by slice.
    slice_col : str
        Column name for the slice/dimension.
    val_col : str, default="aggregated_value"
        Column containing the aggregated metric values.
    share_col_name : str, default="share_pct"
        Name of the resulting share percentage column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column for the percentage share.
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "region": ["North", "South", "East"],
    ...     "aggregated_value": [50, 30, 20]
    ... })
    >>> compute_slice_shares(df, "region")
       region  aggregated_value  share_pct
    0  North                50       50.0
    1  South                30       30.0
    2   East                20       20.0
    """
    # Input validation
    if slice_col not in agg_df.columns:
        raise ValueError(f"slice_col '{slice_col}' not found in DataFrame")
    if val_col not in agg_df.columns:
        raise ValueError(f"val_col '{val_col}' not found in DataFrame")
    
    # Create a copy of the DataFrame
    dff = agg_df.copy()
    
    # Calculate total
    total = dff[val_col].sum()
    
    # Calculate share percentages
    if total == 0:
        dff[share_col_name] = 0.0
    else:
        dff[share_col_name] = (dff[val_col] / total) * 100.0
        
    return dff


def rank_metric_slices(
    agg_df: pd.DataFrame, 
    val_col: str = "aggregated_value", 
    top_n: int = 5, 
    ascending: bool = False
) -> pd.DataFrame:
    """
    Return the top or bottom slices sorted by value.
    
    Parameters
    ----------
    agg_df : pd.DataFrame
        DataFrame with aggregated values.
    val_col : str, default="aggregated_value"
        Column containing the metric values.
    top_n : int, default=5
        Number of slices to return.
    ascending : bool, default=False
        If True, returns the lowest slices; if False, returns the highest.

    Returns
    -------
    pd.DataFrame
        DataFrame with the ranked slices.
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "region": ["North", "South", "East", "West"],
    ...     "aggregated_value": [100, 200, 50, 75]
    ... })
    >>> rank_metric_slices(df, top_n=2)
       region  aggregated_value
    0  South               200
    1  North               100
    
    >>> rank_metric_slices(df, top_n=2, ascending=True)
       region  aggregated_value
    0   East                50
    1   West                75
    """
    # Input validation
    if val_col not in agg_df.columns:
        raise ValueError(f"val_col '{val_col}' not found in DataFrame")
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    
    # Create a copy and sort
    dff = agg_df.copy()
    dff.sort_values(val_col, ascending=ascending, inplace=True)
    
    # Return top_n rows
    return dff.head(top_n)

# =============================================================================
# Main Functions: Composition & Impact Analysis
# =============================================================================

def analyze_composition_changes(
    df_t0: pd.DataFrame, 
    df_t1: pd.DataFrame, 
    slice_col: str = "segment", 
    val_col: str = "aggregated_value"
) -> pd.DataFrame:
    """
    Compare values and percentage shares between two time periods (T0 and T1).
    
    Parameters
    ----------
    df_t0 : pd.DataFrame
        Aggregated data for time T0 (earlier period).
    df_t1 : pd.DataFrame
        Aggregated data for time T1 (later period).
    slice_col : str, default="segment"
        Column representing the slice/dimension.
    val_col : str, default="aggregated_value"
        Column with the aggregated metric values.

    Returns
    -------
    pd.DataFrame
        DataFrame comparing T0 and T1 with differences in values and shares.
        
    Notes
    -----
    The output contains:
    - slice_col: Dimension slice
    - {val_col}_t0: Value in period T0
    - {val_col}_t1: Value in period T1
    - share_pct_t0: Share percentage in T0
    - share_pct_t1: Share percentage in T1
    - abs_diff: Absolute difference (T1 - T0)
    - share_diff: Share difference (share_pct_t1 - share_pct_t0)
    """
    # Input validation
    if slice_col not in df_t0.columns or slice_col not in df_t1.columns:
        raise ValueError(f"slice_col '{slice_col}' not found in one or both DataFrames")
    if val_col not in df_t0.columns or val_col not in df_t1.columns:
        raise ValueError(f"val_col '{val_col}' not found in one or both DataFrames")

    # Ensure share percentages exist in both dataframes
    def ensure_share(df_in: pd.DataFrame) -> pd.DataFrame:
        if "share_pct" not in df_in.columns:
            return compute_slice_shares(df_in, slice_col, val_col=val_col)
        return df_in

    t0 = ensure_share(df_t0.copy())
    t1 = ensure_share(df_t1.copy())

    # Merge T0 and T1 data
    merged = pd.merge(
        t0[[slice_col, val_col, "share_pct"]],
        t1[[slice_col, val_col, "share_pct"]],
        on=slice_col,
        how="outer",
        suffixes=("_t0", "_t1")
    ).fillna(0)

    # Calculate differences
    merged["abs_diff"] = merged[f"{val_col}_t1"] - merged[f"{val_col}_t0"]
    merged["share_diff"] = merged["share_pct_t1"] - merged["share_pct_t0"]
    
    # Sort by absolute difference (descending)
    merged.sort_values("abs_diff", ascending=False, inplace=True, ignore_index=True)
    
    return merged


def analyze_dimension_impact(
    df_t0: pd.DataFrame, 
    df_t1: pd.DataFrame, 
    dimension_col: str = "segment",
    value_col: str = "value"
) -> pd.DataFrame:
    """
    Analyze how each dimension slice contributed to the overall metric change.
    
    Parameters
    ----------
    df_t0 : pd.DataFrame
        DataFrame for the comparison period (T0).
    df_t1 : pd.DataFrame
        DataFrame for the evaluation period (T1).
    dimension_col : str, default="segment"
        Column representing the dimension slice.
    value_col : str, default="value"
        Column with metric values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - dimension_col: The dimension slice
        - {value_col}_t0: Value in period T0 
        - {value_col}_t1: Value in period T1
        - delta: Absolute change (T1 - T0)
        - share_of_total_delta: Percentage contribution to overall change
        
    Notes
    -----
    The rows are sorted by delta (descending), showing which dimension
    slices had the largest impact on the overall metric change.
    """
    # Input validation
    if dimension_col not in df_t0.columns or dimension_col not in df_t1.columns:
        raise ValueError(f"dimension_col '{dimension_col}' not found in one or both DataFrames")
    if value_col not in df_t0.columns or value_col not in df_t1.columns:
        raise ValueError(f"value_col '{value_col}' not found in one or both DataFrames")

    # Prepare DataFrames
    t0 = df_t0.copy()
    t1 = df_t1.copy()
    
    # Rename columns for clarity
    t0.rename(columns={value_col: f"{value_col}_t0"}, inplace=True)
    t1.rename(columns={value_col: f"{value_col}_t1"}, inplace=True)

    # Merge T0 and T1 data
    merged = pd.merge(
        t0[[dimension_col, f"{value_col}_t0"]],
        t1[[dimension_col, f"{value_col}_t1"]],
        on=dimension_col,
        how="outer"
    ).fillna(0)

    # Calculate delta and share of total delta
    merged["delta"] = merged[f"{value_col}_t1"] - merged[f"{value_col}_t0"]
    total_delta = merged["delta"].sum()

    # Calculate each slice's contribution to total delta
    merged["share_of_total_delta"] = merged["delta"].apply(
        lambda d: _safe_share(d, total_delta)
    )
    
    # Sort by delta (descending)
    merged.sort_values("delta", ascending=False, inplace=True, ignore_index=True)
    
    return merged


def calculate_concentration_index(
    df: pd.DataFrame, 
    val_col: str = "aggregated_value",
    method: str = "HHI"
) -> float:
    """
    Calculate a concentration index to measure distribution inequality.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated metric values.
    val_col : str, default="aggregated_value"
        Column containing the values.
    method : str, default="HHI"
        Method to use: "HHI" (Herfindahl-Hirschman Index) or "gini" (Gini coefficient).

    Returns
    -------
    float
        The concentration index value (0 to 1 scale).
        
    Notes
    -----
    - HHI is the sum of squared market shares (0=perfect equality, 1=monopoly)
    - Gini coefficient measures inequality (0=perfect equality, 1=perfect inequality)
    
    Examples
    --------
    >>> df = pd.DataFrame({"aggregated_value": [50, 30, 20]})
    >>> calculate_concentration_index(df, method="HHI")
    0.38
    
    >>> calculate_concentration_index(df, method="gini")
    0.2
    """
    # Input validation
    if val_col not in df.columns:
        raise ValueError(f"val_col '{val_col}' not found in DataFrame")
    
    # Handle empty dataframe
    if df.empty or df[val_col].sum() <= 0:
        return 0.0
    
    method = method.upper()
    
    if method == "HHI":
        # HHI = sum of squared market shares
        total = df[val_col].sum()
        if total <= 0:
            return 0.0
        shares = df[val_col] / total
        return float(np.sum(shares**2))
    
    elif method.lower() == "gini":
        # Gini coefficient calculation
        values = df[val_col].to_numpy()
        if np.any(values < 0):
            raise ValueError("Negative values not allowed for Gini coefficient.")
        
        # Sort values
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        if n <= 1:
            return 0.0
            
        # Cumulative proportion of the population (x-axis)
        cum_people = np.arange(1, n + 1) / n
        
        # Cumulative proportion of income (y-axis)
        cum_income = np.cumsum(sorted_vals) / np.sum(sorted_vals)
        
        # Gini coefficient using area under Lorenz curve
        B = np.trapz(cum_income, cum_people)  # Area under the Lorenz curve
        gini = 1 - 2 * B  # Gini = 1 - 2B
        
        return float(gini)
    
    else:
        raise ValueError(f"Unknown concentration method: {method}. Use 'HHI' or 'gini'.")


def compare_dimension_slices_over_time(
    df: pd.DataFrame,
    slice_col: str,
    date_col: str = "date",
    value_col: str = "value",
    t0: Optional[Union[str, pd.Timestamp]] = None,
    t1: Optional[Union[str, pd.Timestamp]] = None,
    agg: str = "sum"
) -> pd.DataFrame:
    """
    Compare each dimension slice between two time points.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data.
    slice_col : str
        Column representing the dimension slice.
    date_col : str, default="date"
        Column containing date values.
    value_col : str, default="value"
        Column with metric values.
    t0 : Optional[Union[str, pd.Timestamp]], default=None
        First time point (defaults to the earliest date).
    t1 : Optional[Union[str, pd.Timestamp]], default=None
        Second time point (defaults to the latest date).
    agg : str, default="sum"
        Aggregation function to apply ('sum', 'mean', 'count', etc.)

    Returns
    -------
    pd.DataFrame
        DataFrame comparing values at T0 and T1 with columns:
        - slice_col: Dimension slice
        - val_t0: Value at time T0
        - val_t1: Value at time T1
        - abs_diff: Absolute difference (T1 - T0) 
        - pct_diff: Percentage difference
    """
    # Input validation
    if slice_col not in df.columns:
        raise ValueError(f"slice_col '{slice_col}' not found in DataFrame")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in DataFrame")

    # Create a copy and convert date column to datetime
    dff = df.copy()
    dff[date_col] = pd.to_datetime(dff[date_col])

    # Determine t0 and t1 if not provided
    if t0 is None:
        t0 = dff[date_col].min()
    else:
        t0 = pd.to_datetime(t0)
        
    if t1 is None:
        t1 = dff[date_col].max()
    else:
        t1 = pd.to_datetime(t1)

    # Filter data for T0 and T1
    df_t0 = dff[dff[date_col] == t0]
    df_t1 = dff[dff[date_col] == t1]

    # Aggregate by slice for T0 and T1
    if agg not in ["sum", "mean", "min", "max", "count", "median"]:
        raise ValueError(f"Unsupported aggregation method: {agg}")
        
    g_t0 = df_t0.groupby(slice_col)[value_col].agg(agg).reset_index().rename(columns={value_col: "val_t0"})
    g_t1 = df_t1.groupby(slice_col)[value_col].agg(agg).reset_index().rename(columns={value_col: "val_t1"})

    # Merge T0 and T1 data
    merged = pd.merge(g_t0, g_t1, on=slice_col, how="outer").fillna(0)
    
    # Calculate differences
    merged["abs_diff"] = merged["val_t1"] - merged["val_t0"]
    merged["pct_diff"] = merged.apply(lambda r: _safe_pct_diff(r, "val_t0"), axis=1)
    
    # Sort by absolute difference (descending)
    merged.sort_values("abs_diff", ascending=False, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    return merged