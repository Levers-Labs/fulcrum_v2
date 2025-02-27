# =============================================================================
# ComparativeAnalysis
#
# This file includes primitives for comparing multiple metrics:
# correlation, predictive significance (Granger causality), benchmarks,
# cointegration, and other statistical tests for relationships between metrics.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - scipy.stats for statistical tests
#   - statsmodels.tsa.stattools for time series analysis
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, grangercausalitytests
from typing import Dict, List, Any, Optional, Union, Tuple

def _pick_default_corr_method(df: pd.DataFrame, metrics: List[str]) -> str:
    """
    Select an appropriate correlation method based on the cardinality of the data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics to correlate
    metrics : List[str]
        List of column names for metrics to analyze
        
    Returns
    -------
    str
        The correlation method to use: 'pearson', 'kendall', or 'spearman'
    """
    unique_counts = [
        pd.to_numeric(df[m], errors='coerce').dropna().nunique()
        for m in metrics if m in df.columns
    ]
    if not unique_counts:
        return "pearson"
    if min(unique_counts) < 10:
        return "kendall"
    elif min(unique_counts) < 30:
        return "spearman"
    else:
        return "pearson"

def compare_metrics_correlation(
    df: pd.DataFrame, 
    metrics: List[str], 
    method: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate correlation coefficients between multiple metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics to correlate
    metrics : List[str]
        List of column names for metrics to analyze
    method : Optional[str], default=None
        Correlation method: 'pearson', 'spearman', or 'kendall'
        If None, a method is chosen based on data cardinality
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix with metrics as both rows and columns
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'metric1': [1, 2, 3, 4, 5],
    ...     'metric2': [2, 3, 5, 7, 11],
    ...     'metric3': [10, 9, 8, 7, 6]
    ... })
    >>> compare_metrics_correlation(df, ['metric1', 'metric2', 'metric3'])
    """
    valid_methods = ["pearson", "spearman", "kendall"]
    if method is None:
        method = _pick_default_corr_method(df, metrics)
    elif method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    # Filter metrics to only those present in the DataFrame
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        return pd.DataFrame()

    sub = df[valid_metrics].copy()
    for col in valid_metrics:
        sub[col] = pd.to_numeric(sub[col], errors='coerce')
    sub.dropna(axis=0, how='any', inplace=True)
    
    if len(sub) < 2:
        return pd.DataFrame(np.nan, index=metrics, columns=metrics)

    return sub.corr(method=method)

def benchmark_metrics_against_peers(
    df: pd.DataFrame,
    df_peer: pd.DataFrame,
    date_col: str = "date",
    primary_val_col: str = "primary_value",
    peer_val_col: str = "peer_value",
    method: str = "ratio"
) -> pd.DataFrame:
    """
    Compare a primary metric against a peer metric by ratio or difference.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the primary metric time series
    df_peer : pd.DataFrame
        DataFrame containing the peer metric time series
    date_col : str, default="date"
        Column name for date in both DataFrames
    primary_val_col : str, default="primary_value"
        Column name for the primary metric values
    peer_val_col : str, default="peer_value"
        Column name for the peer metric values
    method : str, default="ratio"
        Comparison method: "ratio" or "difference"
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus a 'benchmark_comparison' column
        
    Notes
    -----
    For ratio method, benchmark_comparison = primary_value / peer_value
    For difference method, benchmark_comparison = primary_value - peer_value
    """
    if date_col not in df.columns or primary_val_col not in df.columns:
        raise ValueError(f"Primary DataFrame must contain {date_col} and {primary_val_col}")
    
    if date_col not in df_peer.columns or peer_val_col not in df_peer.columns:
        raise ValueError(f"Peer DataFrame must contain {date_col} and {peer_val_col}")
        
    if method not in ["ratio", "difference"]:
        raise ValueError("Method must be 'ratio' or 'difference'")
        
    d1 = df[[date_col, primary_val_col]].copy()
    d2 = df_peer[[date_col, peer_val_col]].copy()
    
    # Convert date columns to datetime
    d1[date_col] = pd.to_datetime(d1[date_col])
    d2[date_col] = pd.to_datetime(d2[date_col])
    
    merged = pd.merge(d1, d2, on=date_col, how="inner").dropna()
    
    if method == "ratio":
        def safe_ratio(row):
            if row[peer_val_col] == 0:
                return np.nan
            return row[primary_val_col] / row[peer_val_col]
        merged["benchmark_comparison"] = merged.apply(safe_ratio, axis=1)
    elif method == "difference":
        merged["benchmark_comparison"] = merged[primary_val_col] - merged[peer_val_col]
    
    return merged

def compare_means(
    groupA: Union[List[float], np.ndarray, pd.Series],
    groupB: Union[List[float], np.ndarray, pd.Series],
    equal_var: bool = True
) -> Dict[str, Any]:
    """
    Perform a t-test to compare means between two groups.
    
    Parameters
    ----------
    groupA : Union[List[float], np.ndarray, pd.Series]
        First group of values
    groupB : Union[List[float], np.ndarray, pd.Series]
        Second group of values
    equal_var : bool, default=True
        Whether to assume equal variance (Student's t-test) or not (Welch's t-test)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - test_stat: t-statistic value
        - p_value: p-value of the test
        - df: degrees of freedom
        - significant: True if p_value < 0.05
    """
    arrA = np.array(groupA, dtype=float)
    arrB = np.array(groupB, dtype=float)
    arrA = arrA[~np.isnan(arrA)]
    arrB = arrB[~np.isnan(arrB)]
    
    if len(arrA) < 2 or len(arrB) < 2:
        return {"test_stat": None, "p_value": None, "df": None, "significant": False}
    
    stat, pval = stats.ttest_ind(arrA, arrB, equal_var=equal_var, nan_policy='omit')
    dof = (len(arrA) + len(arrB) - 2) if equal_var else None
    
    return {
        "test_stat": float(stat),
        "p_value": float(pval),
        "df": dof,
        "significant": (pval < 0.05)
    }

def perform_granger_causality_test(
    df: pd.DataFrame,
    target: str,
    exog: str,
    maxlag: int = 4,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Perform a Granger causality test to determine whether the time series in 'exog' helps predict 'target'.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    target : str
        The name of the target variable column
    exog : str
        The name of the exogenous variable column
    maxlag : int, default=4
        Maximum number of lags to test
    verbose : bool, default=False
        If True, prints detailed output from the test
        
    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the Granger causality test results for each lag with columns:
        'lag', 'F_stat', 'p_value', 'df_num', and 'df_denom'
    """
    if target not in df.columns or exog not in df.columns:
        raise ValueError("Both target and exog columns must be present in the DataFrame")
    
    # Convert to numeric and handle missing values
    test_data = df[[target, exog]].copy()
    for col in [target, exog]:
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    
    test_data = test_data.dropna()
    
    if len(test_data) <= maxlag + 1:
        return pd.DataFrame(columns=['lag', 'F_stat', 'p_value', 'df_num', 'df_denom'])
    
    try:
        results = grangercausalitytests(test_data, maxlag=maxlag, verbose=verbose)
        result_list = []
        
        for lag, res in results.items():
            # Extract the F-test result; it returns a tuple: (F, p-value, df_num, df_denom)
            f_test = res[0].get('ssr_ftest')
            if f_test is not None:
                F_stat, p_value, df_num, df_denom = f_test
                result_list.append({
                    "lag": lag,
                    "F_stat": F_stat,
                    "p_value": p_value,
                    "df_num": df_num,
                    "df_denom": df_denom
                })
        
        return pd.DataFrame(result_list)
    except Exception as e:
        print(f"Error in Granger causality test: {str(e)}")
        return pd.DataFrame(columns=['lag', 'F_stat', 'p_value', 'df_num', 'df_denom'])

def perform_cointegration_test(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> Dict[str, Any]:
    """
    Perform a cointegration test between two time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    col1 : str
        Name of the first time series column
    col2 : str
        Name of the second time series column
        
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - test_stat: The cointegration test statistic
        - p_value: The p-value from the cointegration test
        - critical_values: Critical values for the test at 1%, 5%, and 10% significance levels
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Both col1 and col2 must be present in the DataFrame")
    
    data = df[[col1, col2]].dropna()
    
    if len(data) < 20:  # Minimum data for meaningful cointegration test
        return {
            "test_stat": None,
            "p_value": None,
            "critical_values": None,
            "error": "Insufficient data for cointegration test (minimum 20 observations required)"
        }
    
    try:
        series1 = pd.to_numeric(data[col1], errors='coerce')
        series2 = pd.to_numeric(data[col2], errors='coerce')
        
        # Drop NaNs after conversion
        valid_mask = ~(np.isnan(series1) | np.isnan(series2))
        series1 = series1[valid_mask]
        series2 = series2[valid_mask]
        
        if len(series1) < 20:
            return {
                "test_stat": None,
                "p_value": None,
                "critical_values": None,
                "error": "Insufficient valid numeric data for cointegration test"
            }
        
        test_stat, p_value, critical_values = coint(series1, series2)
        return {
            "test_stat": float(test_stat),
            "p_value": float(p_value),
            "critical_values": critical_values.tolist()
        }
    except Exception as e:
        return {
            "test_stat": None,
            "p_value": None,
            "critical_values": None,
            "error": f"Error in cointegration test: {str(e)}"
        }