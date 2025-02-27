"""
dimensional_analysis.py

This module provides a comprehensive suite of functions for dimensional analysis—decomposing metrics by their dimensions and slices. It includes methods to:
  - Aggregate and compute slice metrics and share distributions.
  - Rank slices and compare changes over time.
  - Detect anomalies within slices.
  - Evaluate concentration and inequality via indices.
  - Analyze composition and drift between time periods.
  - Compare frequency distributions using chi-square tests.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List
import logging
from scipy.stats import chisquare

# =============================================================================
# Helper Functions
# =============================================================================

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
# Main Functions: Aggregation & Slice Metrics
# =============================================================================

def calculate_slice_metrics(
    df: pd.DataFrame, 
    slice_col: str, 
    value_col: str, 
    agg: str = "sum",
    top_n: Optional[int] = None,
    other_label: str = "Other",
    dropna_slices: bool = True
) -> pd.DataFrame:
    """
    Group by `slice_col` and aggregate `value_col` with the specified aggregation function.
    Optionally, keep only the top_n slices and lump the remaining ones into 'Other'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    slice_col : str
        Column name used for slicing.
    value_col : str
        Column name containing the metric values.
    agg : str, default "sum"
        Aggregation function to apply (e.g., "sum", "mean").
    top_n : Optional[int], default None
        Number of top slices to retain. Remaining slices are combined into one labeled as 'Other'.
    other_label : str, default "Other"
        Label for lumped slices.
    dropna_slices : bool, default True
        Whether to drop rows with NaN in `slice_col`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [slice_col, "aggregated_value"].
    """
    dff = df.copy()
    if dropna_slices:
        dff = dff[~dff[slice_col].isna()]

    grouped = dff.groupby(slice_col)[value_col]
    try:
        result = grouped.agg(agg).reset_index()
    except Exception as e:
        raise ValueError(f"Invalid aggregation method '{agg}': {e}")

    result.rename(columns={value_col: "aggregated_value"}, inplace=True)
    result.sort_values("aggregated_value", ascending=False, inplace=True)
    
    if top_n is not None and len(result) > top_n:
        top_df = result.iloc[:top_n].copy()
        other_value = result.iloc[top_n:]["aggregated_value"].sum()
        other_df = pd.DataFrame({slice_col: [other_label], "aggregated_value": [other_value]})
        result = pd.concat([top_df, other_df], ignore_index=True)

    result.sort_values("aggregated_value", ascending=False, inplace=True, ignore_index=True)
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
    Return the top_n (or bottom_n if ascending=True) slices sorted by `val_col`.

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
    slice_col: str = "segment",
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
    slice_col : str, default "segment"
        Column representing the slice.
    value_col : str, default "value"
        Column with metric values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for T0 and T1 values, differences, and relative impact.
    """
    t0 = df_t0.copy()
    t1 = df_t1.copy()
    t0.rename(columns={value_col: f"{value_col}_t0"}, inplace=True)
    t1.rename(columns={value_col: f"{value_col}_t1"}, inplace=True)

    merged = pd.merge(
        t0[[slice_col, f"{value_col}_t0"]],
        t1[[slice_col, f"{value_col}_t1"]],
        on=slice_col,
        how="outer"
    ).fillna(0)

    merged["delta"] = merged[f"{value_col}_t1"] - merged[f"{value_col}_t0"]
    total_delta = merged["delta"].sum()

    merged["pct_of_total_delta"] = merged["delta"].apply(
        lambda d: _safe_share(d, total_delta) if total_delta != 0 else 0.0
    )
    merged.sort_values("delta", ascending=False, inplace=True, ignore_index=True)
    return merged

# =============================================================================
# Main Functions: Anomaly Detection & Inequality Metrics
# =============================================================================

def detect_anomalies_in_slices(
    df: pd.DataFrame, 
    slice_col: str, 
    value_col: str,
    date_col: Optional[str] = None,
    z_thresh: float = 3.0,
    min_points_per_slice: int = 5
) -> pd.DataFrame:
    """
    Detect anomalies in a numeric metric for each slice by comparing recent values to historical statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with metric data.
    slice_col : str
        Column representing the slice.
    value_col : str
        Column with numeric metric values.
    date_col : Optional[str], default None
        Column with date values; if provided, data is sorted by date.
    z_thresh : float, default 3.0
        Z-score threshold to flag anomalies.
    min_points_per_slice : int, default 5
        Minimum number of data points required per slice.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional boolean column "slice_anomaly" indicating anomalies.
    """
    dff = df.copy()
    if date_col:
        dff.sort_values(date_col, inplace=True)

    stats = dff.groupby(slice_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    stats.rename(columns={"mean": "slice_mean", "std": "slice_std", "count": "slice_count"}, inplace=True)
    merged = pd.merge(dff, stats, on=slice_col, how="left")

    def is_anomaly(row: pd.Series) -> bool:
        if row["slice_count"] < min_points_per_slice:
            return False
        if pd.isna(row["slice_std"]) or row["slice_std"] == 0:
            return False
        return abs(row[value_col] - row["slice_mean"]) > z_thresh * row["slice_std"]

    if date_col:
        latest_date_per_slice = merged.groupby(slice_col)[date_col].transform("max")
        merged["slice_anomaly"] = False
        mask = (merged[date_col] == latest_date_per_slice)
        merged.loc[mask, "slice_anomaly"] = merged.loc[mask].apply(is_anomaly, axis=1)
    else:
        merged["slice_anomaly"] = merged.apply(is_anomaly, axis=1)

    return merged

def calculate_share_square_concentration_index(
    df: pd.DataFrame, 
    val_col: str = "aggregated_value"
) -> float:
    """
    Calculate the share square concentration index (sum of squared shares).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated metric values.
    val_col : str, default "aggregated_value"
        Column containing the aggregated metric values.

    Returns
    -------
    float
        The share square concentration index.
    """
    total = df[val_col].sum()
    if total <= 0:
        return 0.0
    shares = df[val_col] / total
    return float(np.sum(shares**2))

def calculate_distribution_inequality_index(
    df: pd.DataFrame, 
    val_col: str = "aggregated_value"
) -> float:
    """
    Calculate the Gini coefficient (a measure of inequality) for the distribution of aggregated values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated values.
    val_col : str, default "aggregated_value"
        Column containing the aggregated metric values.

    Returns
    -------
    float
        The Gini coefficient.
    """
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

# =============================================================================
# Main Functions: Segment Drift & Frequency Distribution Analysis
# =============================================================================

def analyze_segment_drift(
    df: pd.DataFrame,
    date_col: str,
    value_col: str = "value",
    slice_cols: List[str] = [],
    evaluation_start_date: Optional[pd.Timestamp] = None,
    evaluation_end_date: Optional[pd.Timestamp] = None,
    comparison_start_date: Optional[pd.Timestamp] = None,
    comparison_end_date: Optional[pd.Timestamp] = None,
    agg_method: str = "sum",
    target_metric_direction: str = "INCREASING"
) -> Dict[str, Any]:
    """
    Compare a "comparison" period (T0) versus an "evaluation" period (T1) for dimension slices,
    reporting overall changes and slice-level metrics (e.g., absolute differences, relative change,
    pressure, and share changes).

    Parameters
    ----------
    df : pd.DataFrame
        Main dataset with at least [date_col, value_col] and slice columns.
    date_col : str
        Column name containing dates.
    value_col : str, default "value"
        Column with the numeric metric.
    slice_cols : List[str], default []
        List of dimension columns to group by. If empty, all data is treated as a single slice.
    evaluation_start_date : Optional[pd.Timestamp], default None
        Start date for the evaluation period (T1).
    evaluation_end_date : Optional[pd.Timestamp], default None
        End date for the evaluation period (T1).
    comparison_start_date : Optional[pd.Timestamp], default None
        Start date for the comparison period (T0).
    comparison_end_date : Optional[pd.Timestamp], default None
        End date for the comparison period (T0).
    agg_method : str, default "sum"
        Aggregation function for metric values.
    target_metric_direction : str, default "INCREASING"
        Indicates whether an increase is considered positive ("INCREASING") or negative ("DECREASING").

    Returns
    -------
    Dict[str, Any]
        {
          "evaluation_value": float,
          "comparison_value": float,
          "overall_change": float,
          "dimension_slices": [  # sorted by absolute impact descending
             {
               "slice_key": ...,
               "evaluation_value": ...,
               "comparison_value": ...,
               "abs_diff": ...,
               "relative_change": ...,
               "pressure": "UPWARD"/"DOWNWARD"/"UNCHANGED",
               "evaluation_share": ...,
               "comparison_share": ...,
               "slice_share_change_percentage": ...
             },
             ...
          ]
        }
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Define T0 (comparison) and T1 (evaluation) periods
    if comparison_start_date is None:
        comparison_start_date = df_copy[date_col].min()
    if comparison_end_date is None:
        comparison_end_date = df_copy[date_col].min()
    if evaluation_start_date is None:
        evaluation_start_date = df_copy[date_col].max()
    if evaluation_end_date is None:
        evaluation_end_date = df_copy[date_col].max()

    df_t0 = df_copy[(df_copy[date_col] >= pd.to_datetime(comparison_start_date)) & 
                    (df_copy[date_col] <= pd.to_datetime(comparison_end_date))]
    df_t1 = df_copy[(df_copy[date_col] >= pd.to_datetime(evaluation_start_date)) & 
                    (df_copy[date_col] <= pd.to_datetime(evaluation_end_date))]

    # If no slice_cols provided, create a dummy grouping
    if not slice_cols:
        df_t0["_slice_dummy_"] = "All"
        df_t1["_slice_dummy_"] = "All"
        slice_cols = ["_slice_dummy_"]

    g_t0 = df_t0.groupby(slice_cols)[value_col].agg(agg_method).reset_index().rename(columns={value_col: "comp_val"})
    g_t1 = df_t1.groupby(slice_cols)[value_col].agg(agg_method).reset_index().rename(columns={value_col: "eval_val"})

    merged = pd.merge(g_t0, g_t1, on=slice_cols, how="outer").fillna(0)
    merged["abs_diff"] = merged["eval_val"] - merged["comp_val"]

    total_comp = merged["comp_val"].sum()
    total_eval = merged["eval_val"].sum()
    overall_change = (total_eval - total_comp) / total_comp if total_comp != 0 else 0.0

    merged["eval_share"] = merged.apply(lambda r: _safe_share(r["eval_val"], total_eval), axis=1)
    merged["comp_share"] = merged.apply(lambda r: _safe_share(r["comp_val"], total_comp), axis=1)
    merged["slice_share_change_percentage"] = merged["eval_share"] - merged["comp_share"]

    def get_pressure(row: pd.Series) -> str:
        impact = row["abs_diff"]
        direction = target_metric_direction.upper()
        if impact > 0 and direction == "INCREASING":
            return "UPWARD"
        elif impact < 0 and direction == "INCREASING":
            return "DOWNWARD"
        elif impact > 0 and direction == "DECREASING":
            return "DOWNWARD"
        elif impact < 0 and direction == "DECREASING":
            return "UPWARD"
        return "UNCHANGED"

    merged["pressure"] = merged.apply(get_pressure, axis=1)
    merged["relative_change"] = merged.apply(lambda r: _safe_relative_change(r["eval_val"], r["comp_val"]), axis=1)

    dimension_slices = []
    for _, row in merged.iterrows():
        if len(slice_cols) > 1:
            slice_key = tuple(row[col] for col in slice_cols)
        else:
            slice_key = row[slice_cols[0]]
        dimension_slices.append({
            "slice_key": slice_key,
            "evaluation_value": row["eval_val"],
            "comparison_value": row["comp_val"],
            "abs_diff": row["abs_diff"],
            "relative_change": row["relative_change"],
            "pressure": row["pressure"],
            "evaluation_share": row["eval_share"],
            "comparison_share": row["comp_share"],
            "slice_share_change_percentage": row["slice_share_change_percentage"]
        })

    dimension_slices.sort(key=lambda x: abs(x["abs_diff"]), reverse=True)

    return {
        "evaluation_value": float(total_eval),
        "comparison_value": float(total_comp),
        "overall_change": float(overall_change),
        "dimension_slices": dimension_slices
    }

def compare_frequency_distributions(
    observed: List[float],
    expected: List[float]
) -> Dict[str, Any]:
    """
    Compare observed vs. expected frequency distributions using the chi-square test.

    Parameters
    ----------
    observed : List[float]
        Observed frequencies.
    expected : List[float]
        Expected frequencies.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
          - test_stat: Chi-square test statistic.
          - p_value: p-value from the test.
          - df: Degrees of freedom.
          - significant: Boolean indicating if the result is significant (p < 0.05).
    """
    obs = np.array(observed, dtype=float)
    exp = np.array(expected, dtype=float)
    obs = obs[~np.isnan(obs)]
    exp = exp[~np.isnan(exp)]
    if len(obs) < 2 or len(exp) < 2:
        return {"test_stat": None, "p_value": None, "df": None, "significant": False}
    try:
        stat, pval = chisquare(f_obs=obs, f_exp=exp)
    except Exception as e:
        raise ValueError(f"Error during chi-square test: {e}")
    dof = len(obs) - 1
    return {
        "test_stat": float(stat),
        "p_value": float(pval),
        "df": dof,
        "significant": (pval < 0.05)
    }

def compare_distributions_chisquare(
    observed: List[float],
    expected: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compare an observed frequency distribution to an expected distribution using the chi-square test.
    If `expected` is None, a uniform distribution is assumed.

    Parameters
    ----------
    observed : List[float]
        Observed frequencies.
    expected : Optional[List[float]], default None
        Expected frequencies.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
          - chi2_stat: The chi-square test statistic.
          - p_value: The p-value from the test.
    """
    if expected is None:
        total = sum(observed)
        expected = [total / len(observed)] * len(observed)
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    return {
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value)
    }
