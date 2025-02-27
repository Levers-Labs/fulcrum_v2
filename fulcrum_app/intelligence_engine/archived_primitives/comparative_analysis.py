"""
comparative_analysis.py

This module provides functions to:
    - Compute pairwise correlations among metrics.
    - Benchmark a primary metric against a peer metric over time.
    - Compare means between two groups via t-tests.
    - Test for Granger causality between time series.
    - Test for cointegration between time series.
    - Compare frequency distributions using the chi-square test.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from statsmodels.tsa.stattools import grangercausalitytests, coint
import statsmodels.api as sm
import patsy
from scipy.stats import ttest_ind, chisquare

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _pick_default_corr_method(df: pd.DataFrame, metrics: List[str]) -> str:
    """
    Heuristically pick a correlation method based on the uniqueness of each metric column.

    For each metric column:
      - If any column has fewer than 10 unique values, return 'kendall'.
      - Else if any column has fewer than 30 unique values, return 'spearman'.
      - Otherwise, return 'pearson'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    metrics : List[str]
        List of metric column names.

    Returns
    -------
    str
        One of 'pearson', 'spearman', or 'kendall'.
    """
    unique_counts = [
        pd.to_numeric(df[col], errors='coerce').dropna().nunique()
        for col in metrics if col in df.columns
    ]
    if not unique_counts:
        return "pearson"
    if min(unique_counts) < 10:
        return "kendall"
    elif min(unique_counts) < 30:
        return "spearman"
    else:
        return "pearson"

# -----------------------------------------------------------------------------
# Main Analysis Functions
# -----------------------------------------------------------------------------

def compare_metrics_correlation(
    df: pd.DataFrame, 
    metrics: List[str],
    method: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute a pairwise correlation matrix among specified metric columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric data.
    metrics : List[str]
        List of column names on which to compute pairwise correlations.
    method : Optional[str], default=None
        Correlation method to use ('pearson', 'spearman', or 'kendall'). If None, a default is chosen based on data uniqueness.

    Returns
    -------
    pd.DataFrame
        A correlation matrix with both rows and columns corresponding to the specified metrics.
        If insufficient non-null data (<2 rows), returns a DataFrame filled with NaN.
    """
    valid_methods = ["pearson", "spearman", "kendall"]
    if method is None:
        method = _pick_default_corr_method(df, metrics)
    elif method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    sub = df[metrics].copy()
    for col in metrics:
        sub[col] = pd.to_numeric(sub[col], errors='coerce')

    sub.dropna(axis=0, how="any", inplace=True)
    if len(sub) < 2:
        return pd.DataFrame(np.nan, index=metrics, columns=metrics)

    return sub.corr(method=method)

def benchmark_metrics_against_peers(
    df: pd.DataFrame,
    df_peer: pd.DataFrame,
    date_col: str = "date",
    my_val_col: str = "my_value",
    peer_val_col: str = "peer_value",
    method: str = "ratio"
) -> pd.DataFrame:
    """
    Compare a primary metric against a peer (benchmark) metric over time,
    returning either their ratio or difference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the primary metric with a date column.
    df_peer : pd.DataFrame
        DataFrame containing the peer metric with a date column.
    date_col : str, default "date"
        Column name representing the date.
    my_val_col : str, default "my_value"
        Column name for the primary metric.
    peer_val_col : str, default "peer_value"
        Column name for the peer metric.
    method : str, default "ratio"
        Comparison method: 'ratio' returns the ratio (with safe division), 'difference' returns the arithmetic difference.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with an additional column 'benchmark_comparison' containing the comparison metric.
    """
    try:
        d1 = df[[date_col, my_val_col]].copy()
        d2 = df_peer[[date_col, peer_val_col]].copy()
    except KeyError as e:
        raise ValueError(f"Required columns not found: {e}")

    merged = pd.merge(d1, d2, on=date_col, how="inner").dropna()
    if method == "ratio":
        def safe_ratio(row):
            if row[peer_val_col] == 0:
                return np.nan
            return row[my_val_col] / row[peer_val_col]
        merged["benchmark_comparison"] = merged.apply(safe_ratio, axis=1)
    elif method == "difference":
        merged["benchmark_comparison"] = merged[my_val_col] - merged[peer_val_col]
    else:
        raise ValueError(f"Unknown method: {method}")

    return merged

def compare_means(
    groupA: List[float],
    groupB: List[float],
    equal_var: bool = True
) -> Dict[str, Any]:
    """
    Compare the means of two independent groups using a two-sample t-test.

    Parameters
    ----------
    groupA : List[float]
        List of numerical values for group A.
    groupB : List[float]
        List of numerical values for group B.
    equal_var : bool, default True
        Assume equal variances if True.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
            - 'test_stat': The computed t-test statistic (or None if test cannot be performed).
            - 'p_value': The p-value from the test (or None).
            - 'df': Degrees of freedom (if equal_var is True, otherwise None).
            - 'significant': Boolean indicating whether the result is statistically significant (p < 0.05).
    """
    arrA = np.array(groupA, dtype=float)
    arrB = np.array(groupB, dtype=float)
    arrA = arrA[~np.isnan(arrA)]
    arrB = arrB[~np.isnan(arrB)]

    if len(arrA) < 2 or len(arrB) < 2:
        return {"test_stat": None, "p_value": None, "df": None, "significant": False}

    try:
        stat, pval = ttest_ind(arrA, arrB, equal_var=equal_var, nan_policy='omit')
    except Exception as e:
        raise ValueError(f"Error during t-test: {e}")

    dof = (len(arrA) + len(arrB) - 2) if equal_var else None
    return {
        "test_stat": float(stat),
        "p_value": float(pval),
        "df": dof,
        "significant": (pval < 0.05)
    }

