# =============================================================================
# DimensionalAnalysis
#
# This file includes primitives for dimension and driver analysis:
# - Slice metrics, shares, and ranking
# - Composition changes and impact analysis
# - Concentration and distribution analysis
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
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _safe_share(numerator: float, denominator: float) -> float:
    """Calculate share percentage safely."""
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0

def _safe_pct_diff(row: pd.Series, base_col: str) -> Optional[float]:
    """Calculate percentage difference safely given a row and a base value column."""
    if row[base_col] == 0:
        return None
    return (row["abs_diff"] / abs(row[base_col])) * 100.0

def _safe_relative_change(eval_val: float, comp_val: float) -> Optional[float]:
    """Compute relative change safely."""
    if comp_val == 0:
        return None
    return (eval_val - comp_val) / abs(comp_val)

# =============================================================================
# Main Functions: Slice Metrics & Shares
# =============================================================================

def calculate_slice_metrics(
    df: pd.DataFrame, 
    slice_col: str, 
    value_col: str, 
    agg_func: str = 'sum',
    top_n: Optional[int] = None,
    other_label: str = "Other",
    dropna_slices: bool = True
) -> pd.DataFrame:
    """
    Group by slice_col and aggregate value_col with the specified aggregation function.
    Optionally, keep only the top_n slices and lump the remaining ones into 'Other'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    slice_col : str
        Column name used for slicing.
    value_col : str
        Column name containing the metric values.
    agg_func : str, default "sum"
        Aggregation function to apply (e.g., "sum", "mean").
    top_n : Optional[int], default None
        Number of top slices to retain. Remaining slices are combined into one labeled as 'Other'.
    other_label : str, default "Other"
        Label for lumped slices.
    dropna_slices : bool, default True
        Whether to drop rows with NaN in slice_col.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [slice_col, "aggregated_value"].
    """
    dff = df.copy()
    if dropna_slices:
        dff = dff[~dff[slice_col].isna()]

    # Group by slice column and aggregate
    grouped = dff.groupby(slice_col)[value_col]
    
    # Apply appropriate aggregation function
    if agg_func not in ['sum', 'mean', 'min', 'max', 'count']:
        raise ValueError(f"agg_func '{agg_func}' not supported. Use 'sum', 'mean', 'min', 'max', or 'count'.")

    if agg_func == 'sum':
        result = grouped.sum().reset_index(name='aggregated_value')
    elif agg_func == 'mean':
        result = grouped.mean().reset_index(name='aggregated_value')
    elif agg_func == 'min':
        result = grouped.min().reset_index(name='aggregated_value')
    elif agg_func == 'max':
        result = grouped.max().reset_index(name='aggregated_value')
    elif agg_func == 'count':
        result = grouped.count().reset_index(name='aggregated_value')

    # Sort by aggregated value descending
    result.sort_values('aggregated_value', ascending=False, inplace=True, ignore_index=True)

    # Handle top_n and "Other" aggregation if requested
    if top_n is not None and len(result) > top_n:
        top_part = result.iloc[:top_n].copy()
        other_part = result.iloc[top_n:]
        other_val = other_part['aggregated_value'].sum()
        other_row = pd.DataFrame({slice_col: [other_label], 'aggregated_value': [other_val]})
        result = pd.concat([top_part, other_row], ignore_index=True)

    # Ensure final sorting
    result.sort_values('aggregated_value', ascending=False, inplace=True, ignore_index=True)
    return result


def compute_slice_shares(
    agg_df: pd.DataFrame,
    slice_col: str,
    val_col: str = "aggregated_value",
    share_col_name: str = "share_pct"
) -> pd.DataFrame:
    """
    Calculate each slice's percentage share of the total aggregated value.

    Parameters
    ----------
    agg_df : pd.DataFrame
        DataFrame containing aggregated values.
    slice_col : str
        Column name for the slice.
    val_col : str, default "aggregated_value"
        Column containing the aggregated metric values.
    share_col_name : str, default "share_pct"
        Name of the resulting share percentage column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column for the percentage share.
    """
    dff = agg_df.copy()
    total = dff[val_col].sum()
    
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
    Return the top_n (or bottom_n if ascending=True) slices sorted by val_col.

    Parameters
    ----------
    agg_df : pd.DataFrame
        DataFrame with aggregated values.
    val_col : str, default "aggregated_value"
        Column containing the metric values.
    top_n : int, default 5
        Number of slices to return.
    ascending : bool, default False
        If True, returns the lowest slices; otherwise, the highest.

    Returns
    -------
    pd.DataFrame
        DataFrame with the ranked slices.
    """
    dff = agg_df.copy()
    dff.sort_values(val_col, ascending=ascending, inplace=True)
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
    Compare absolute values and percentage shares of each slice between two time periods (T0 and T1).

    Parameters
    ----------
    df_t0 : pd.DataFrame
        Aggregated data for time T0.
    df_t1 : pd.DataFrame
        Aggregated data for time T1.
    slice_col : str, default "segment"
        Column representing the slice.
    val_col : str, default "aggregated_value"
        Column with the aggregated metric values.

    Returns
    -------
    pd.DataFrame
        DataFrame comparing T0 and T1 with differences in values and shares.
    """
    def ensure_share(df_in: pd.DataFrame) -> pd.DataFrame:
        if "share_pct" not in df_in.columns:
            return compute_slice_shares(df_in, slice_col, val_col=val_col)
        return df_in

    t0 = ensure_share(df_t0.copy())
    t1 = ensure_share(df_t1.copy())

    merged = pd.merge(
        t0[[slice_col, val_col, "share_pct"]],
        t1[[slice_col, val_col, "share_pct"]],
        on=slice_col,
        how="outer",
        suffixes=("_t0", "_t1")
    ).fillna(0)

    merged["abs_diff"] = merged[f"{val_col}_t1"] - merged[f"{val_col}_t0"]
    merged["share_diff"] = merged["share_pct_t1"] - merged["share_pct_t0"]
    merged.sort_values("abs_diff", ascending=False, inplace=True, ignore_index=True)
    return merged


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
    Compare each slice's aggregated value between two distinct time points (t0 and t1).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a date column.
    slice_col : str
        Column representing the dimension slice.
    date_col : str, default "date"
        Column containing date values.
    value_col : str, default "value"
        Column with metric values.
    t0 : Optional[Union[str, pd.Timestamp]], default None
        First time point (defaults to the earliest date).
    t1 : Optional[Union[str, pd.Timestamp]], default None
        Second time point (defaults to the latest date).
    agg : str, default "sum"
        Aggregation function to apply.

    Returns
    -------
    pd.DataFrame
        DataFrame comparing aggregated values and their differences.
    """
    dff = df.copy()
    dff[date_col] = pd.to_datetime(dff[date_col])

    t0 = pd.to_datetime(t0) if t0 is not None else dff[date_col].min()
    t1 = pd.to_datetime(t1) if t1 is not None else dff[date_col].max()

    df_t0 = dff[dff[date_col] == t0]
    df_t1 = dff[dff[date_col] == t1]

    g_t0 = df_t0.groupby(slice_col)[value_col].agg(agg).reset_index().rename(columns={value_col: "val_t0"})
    g_t1 = df_t1.groupby(slice_col)[value_col].agg(agg).reset_index().rename(columns={value_col: "val_t1"})

    merged = pd.merge(g_t0, g_t1, on=slice_col, how="outer").fillna(0)
    merged["abs_diff"] = merged["val_t1"] - merged["val_t0"]
    merged["pct_diff"] = merged.apply(lambda r: _safe_pct_diff(r, "val_t0"), axis=1)
    merged.sort_values("abs_diff", ascending=False, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def analyze_dimension_impact(
    df_t0: pd.DataFrame, 
    df_t1: pd.DataFrame, 
    dimension_col: str = "segment",
    value_col: str = "value"
) -> pd.DataFrame:
    """
    Analyze slice-level changes from T0 to T1—including absolute differences,
    relative changes, and percentage shares.

    Parameters
    ----------
    df_t0 : pd.DataFrame
        DataFrame for the comparison period (T0).
    df_t1 : pd.DataFrame
        DataFrame for the evaluation period (T1).
    dimension_col : str, default "segment"
        Column representing the slice.
    value_col : str, default "value"
        Column with metric values.

    Returns
    -------
    pd.DataFrame
      [dimension_col, valT0, valT1, delta, share_of_total_delta]
    """
    t0 = df_t0.copy()
    t1 = df_t1.copy()
    t0.rename(columns={value_col: f"{value_col}_t0"}, inplace=True)
    t1.rename(columns={value_col: f"{value_col}_t1"}, inplace=True)

    merged = pd.merge(
        t0[[dimension_col, f"{value_col}_t0"]],
        t1[[dimension_col, f"{value_col}_t1"]],
        on=dimension_col,
        how="outer"
    ).fillna(0)

    merged["delta"] = merged[f"{value_col}_t1"] - merged[f"{value_col}_t0"]
    total_delta = merged["delta"].sum()

    merged["share_of_total_delta"] = merged["delta"].apply(
        lambda d: _safe_share(d, total_delta) if total_delta != 0 else 0.0
    )
    merged.sort_values("delta", ascending=False, inplace=True, ignore_index=True)
    return merged


def calculate_concentration_index(
    df: pd.DataFrame, 
    val_col: str = "aggregated_value",
    method: str = "HHI"
) -> float:
    """
    Calculate a concentration index to measure distribution inequality.
    
    Currently supports:
    - HHI (Herfindahl-Hirschman Index): Sum of squared market shares (0 to 1)
    - Gini coefficient: Measures inequality (0=perfect equality, 1=perfect inequality)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated metric values.
    val_col : str, default "aggregated_value"
        Column containing the values.
    method : str, default "HHI"
        Method to use: "HHI" or "gini".

    Returns
    -------
    float
        The concentration index value.
    """
    if method.upper() == "HHI":
        # HHI = sum of squared market shares (from 0 to 1)
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
        total = np.sum(values)
        if total == 0:
            return 0.0
            
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals)) / (n * total) - (n + 1) / n
        return float(gini)
    
    else:
        raise ValueError(f"Unknown concentration method: {method}")

# =============================================================================
# Driver Analysis and Key Dimension Identification
# =============================================================================

@dataclass
class DimensionValuePair:
    """Dimension and value pair for segment identification."""
    dimension: str
    value: str


@dataclass
class Dimension:
    """Dimension information with statistical significance."""
    name: str
    score: float = 0
    is_key_dimension: bool = False
    values: Set[str] = field(default_factory=set)


def find_key_dimensions(
    df: pd.DataFrame, 
    dimensions: List[str], 
    metric_col: str, 
    baseline_df: pd.DataFrame, 
    comparison_df: pd.DataFrame, 
    agg_method: str = "sum"
) -> Tuple[List[str], Dict[str, float]]:
    """
    Identify the key dimensions driving changes in metrics between two time periods.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset
    dimensions : List[str]
        List of dimension column names to analyze
    metric_col : str
        Column containing the numeric metric
    baseline_df : pd.DataFrame
        The baseline period data (T0)
    comparison_df : pd.DataFrame
        The comparison period data (T1)
    agg_method : str, default="sum"
        Aggregation method to use ("sum", "mean", etc.)

    Returns
    -------
    Tuple[List[str], Dict[str, float]]
        A tuple containing (key_dimensions, dimension_scores)
    """
    key_dimensions = []
    dimension_scores = {}

    # Calculate the weights for each dimension
    for dimension in dimensions:
        # Skip if the dimension has too many values (> 100)
        unique_values = df[dimension].nunique()
        if unique_values > 100:
            continue
            
        # Get aggregated metrics by dimension for baseline and comparison
        if agg_method == "sum":
            baseline_metric_by_dim = baseline_df.groupby(dimension)[metric_col].sum()
            comparison_metric_by_dim = comparison_df.groupby(dimension)[metric_col].sum()
        elif agg_method == "mean":
            baseline_metric_by_dim = baseline_df.groupby(dimension)[metric_col].mean()
            comparison_metric_by_dim = comparison_df.groupby(dimension)[metric_col].mean()
        elif agg_method == "count":
            baseline_metric_by_dim = baseline_df.groupby(dimension)[metric_col].count()
            comparison_metric_by_dim = comparison_df.groupby(dimension)[metric_col].count()
        else:
            raise ValueError(f"Unsupported agg_method: {agg_method}")
        
        # Join the two series
        combined = pd.DataFrame({
            'baseline': baseline_metric_by_dim,
            'comparison': comparison_metric_by_dim
        }).fillna(0)
        
        # Calculate the total metrics
        baseline_total = baseline_metric_by_dim.sum()
        comparison_total = comparison_metric_by_dim.sum()
        
        # Calculate weights and changes
        combined['weight'] = (combined['baseline'] + combined['comparison']) / (baseline_total + comparison_total)
        
        # Calculate percent change safely 
        combined['change'] = combined.apply(
            lambda row: _safe_relative_change(row['comparison'], row['baseline']) * 100 
                        if not pd.isna(row['baseline']) and row['baseline'] != 0 
                        else 0, 
            axis=1
        )
        
        # Calculate weighted change
        combined['weighted_change'] = combined['weight'] * combined['change']
        weighted_change_mean = combined['weighted_change'].sum()
        
        # Calculate standard deviation of changes
        combined['squared_diff'] = combined['weight'] * (combined['change'] - weighted_change_mean)**2
        weighted_std = math.sqrt(combined['squared_diff'].sum())
        
        # Store the score for this dimension
        dimension_scores[dimension] = weighted_std
    
    # Determine key dimensions (high variance dimensions)
    if dimension_scores:
        mean_score = sum(dimension_scores.values()) / len(dimension_scores)
        for dim, score in dimension_scores.items():
            if score > mean_score or score > 0.02:
                key_dimensions.append(dim)
    
    return key_dimensions, dimension_scores


def create_dimension_objects(
    df: pd.DataFrame, 
    dimensions: List[str], 
    dimension_scores: Dict[str, float]
) -> Dict[str, Dimension]:
    """
    Create Dimension objects with metadata for each dimension.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the dimensions
    dimensions : List[str]
        List of dimension column names
    dimension_scores : Dict[str, float]
        Dictionary mapping dimension names to their scores

    Returns
    -------
    Dict[str, Dimension]
        Dictionary of dimension objects
    """
    dimension_objects = {}
    
    for dim in dimensions:
        values = set(df[dim].astype(str).unique())
        
        # Remove null or empty values
        if None in values:
            values.remove(None)
        if '' in values:
            values.remove('')
        if 'nan' in values:
            values.remove('nan')
        
        # Create dimension object
        score = dimension_scores.get(dim, 0)
        is_key = dim in dimension_scores and (
            score > sum(dimension_scores.values()) / len(dimension_scores) or score > 0.02
        )
        
        dimension_objects[dim] = Dimension(
            name=dim,
            score=score,
            is_key_dimension=is_key,
            values=values
        )
    
    return dimension_objects


def analyze_key_drivers(
    df: pd.DataFrame, 
    baseline_date_range: Tuple[str, str], 
    comparison_date_range: Tuple[str, str], 
    dimension_cols: List[str], 
    metric_col: str, 
    agg_method: str = 'sum',
    date_col: str = 'date', 
    expected_value: float = 0, 
    max_dimensions: int = 3
) -> Dict[str, Any]:
    """
    Analyze key drivers in data by comparing metrics between two time periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with dimensions and metrics
    baseline_date_range : Tuple[str, str]
        (start_date, end_date) for the baseline period
    comparison_date_range : Tuple[str, str]
        (start_date, end_date) for the comparison period
    dimension_cols : List[str]
        List of dimension columns to analyze
    metric_col : str
        Column name containing the metric to analyze
    agg_method : str, default='sum'
        Aggregation method ('sum', 'count', or 'mean')
    date_col : str, default='date'
        Column containing dates
    expected_value : float, default=0
        Expected change percentage
    max_dimensions : int, default=3
        Maximum number of dimensions to combine in segmentation
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with analysis results including:
        - key_dimensions: List[str]
        - dimension_scores: Dict[str, float]
        - segments: Dict[str, Dict]
    """
    # Convert string dates to datetime objects
    if date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Filter by date ranges
    baseline_start = pd.to_datetime(baseline_date_range[0])
    baseline_end = pd.to_datetime(baseline_date_range[1])
    comparison_start = pd.to_datetime(comparison_date_range[0])
    comparison_end = pd.to_datetime(comparison_date_range[1])
    
    baseline_df = df[(df[date_col] >= baseline_start) & (df[date_col] <= baseline_end)]
    comparison_df = df[(df[date_col] >= comparison_start) & (df[date_col] <= comparison_end)]
    
    if baseline_df.empty or comparison_df.empty:
        return {
            "error": "Insufficient data for analysis",
            "key_dimensions": [],
            "dimension_scores": {},
            "segments": {}
        }
    
    # Identify key dimensions
    key_dimensions, dimension_scores = find_key_dimensions(
        df, dimension_cols, metric_col, 
        baseline_df, comparison_df, agg_method
    )
    
    # Create segment analysis (analyzing combinations of dimensions)
    segments = {}
    dimension_objects = create_dimension_objects(df, dimension_cols, dimension_scores)
    
    # Calculate the baseline and comparison totals
    if agg_method == 'sum':
        baseline_total = baseline_df[metric_col].sum()
        comparison_total = comparison_df[metric_col].sum()
    elif agg_method == 'mean':
        baseline_total = baseline_df[metric_col].mean()
        comparison_total = comparison_df[metric_col].mean()
    else:
        # For count or other methods, default to sum
        baseline_total = baseline_df[metric_col].sum()
        comparison_total = comparison_df[metric_col].sum()
    
    # Generate dimension combinations up to max_dimensions
    all_combos = []
    for i in range(1, min(max_dimensions, len(dimension_cols)) + 1):
        all_combos.extend(itertools.combinations(dimension_cols, i))
    
    # Analyze each combination of dimensions
    for dim_combo in all_combos:
        # Skip if none of the dimensions are key dimensions (for efficiency)
        if not any(dim in key_dimensions for dim in dim_combo):
            continue
        
        # Group by the dimension combination
        baseline_grouped = baseline_df.groupby(list(dim_combo))[metric_col]
        comparison_grouped = comparison_df.groupby(list(dim_combo))[metric_col]
        
        if agg_method == 'sum':
            baseline_agg = baseline_grouped.sum()
            comparison_agg = comparison_grouped.sum()
        elif agg_method == 'mean':
            baseline_agg = baseline_grouped.mean()
            comparison_agg = comparison_grouped.mean()
        elif agg_method == 'count':
            baseline_agg = baseline_grouped.count()
            comparison_agg = comparison_grouped.count()
        
        # Convert to DataFrame and join
        baseline_df_agg = baseline_agg.reset_index()
        comparison_df_agg = comparison_agg.reset_index()
        
        # Merge the two DataFrames
        merged = pd.merge(
            baseline_df_agg, 
            comparison_df_agg, 
            on=list(dim_combo), 
            how='outer', 
            suffixes=('_baseline', '_comparison')
        ).fillna(0)
        
        # Calculate metrics for each segment
        for _, row in merged.iterrows():
            # Create dimension-value pairs
            dimension_pairs = [
                DimensionValuePair(dim=dim, value=str(row[dim])) 
                for dim in dim_combo
            ]
            
            # Generate serialized key
            key_parts = []
            for pair in sorted(dimension_pairs, key=lambda x: x.dimension):
                key_parts.append(f"{pair.dimension}:{pair.value}")
            serialized_key = "|".join(key_parts)
            
            # Calculate values and changes
            baseline_value = float(row[f'{metric_col}_baseline'])
            comparison_value = float(row[f'{metric_col}_comparison'])
            impact = comparison_value - baseline_value
            
            # Skip if the impact is negligible
            if abs(impact) < 1e-6:
                continue
                
            # Calculate relative change
            change_percentage = None
            if baseline_value != 0:
                change_percentage = (impact / baseline_value) * 100
            
            # Calculate contribution to overall change
            contribution = None
            if (comparison_total - baseline_total) != 0:
                contribution = impact / (comparison_total - baseline_total)
            
            # Calculate change deviation (z-score equivalent)
            # How unusual is this change compared to expected change
            change_dev = abs(change_percentage - expected_value) if change_percentage is not None else None
            
            # Store the segment info
            segments[serialized_key] = {
                "dimensions": [{"dimension": pair.dimension, "value": pair.value} for pair in dimension_pairs],
                "baseline_value": baseline_value,
                "comparison_value": comparison_value,
                "impact": impact,
                "change_percentage": change_percentage,
                "change_deviation": change_dev,
                "contribution": contribution,
                "sort_value": abs(impact)
            }
    
    # Sort segments by absolute impact
    sorted_segments = dict(sorted(
        segments.items(), 
        key=lambda item: item[1]['sort_value'], 
        reverse=True
    ))
    
    # Limit to top 1000 segments for practical purposes
    top_segments = dict(list(sorted_segments.items())[:1000])
    
    result = {
        "key_dimensions": key_dimensions,
        "dimension_scores": dimension_scores,
        "segments": top_segments,
        "baseline_total": baseline_total,
        "comparison_total": comparison_total,
        "total_change": comparison_total - baseline_total,
        "total_change_percentage": ((comparison_total - baseline_total) / baseline_total * 100) 
                                   if baseline_total != 0 else None
    }
    
    return result