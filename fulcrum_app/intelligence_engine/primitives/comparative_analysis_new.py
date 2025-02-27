# =============================================================================
# ComparativeAnalysis
#
# This file includes primitives for comparing multiple metrics:
# correlation, predictive significance, synergy, benchmarks, t-tests, chi-square,
# cointegration, Granger-causality, etc.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - scipy, statsmodels for stats tests
# =============================================================================

# =============================================================================
# ComparativeAnalysis
#
# UPDATED:
#   - Added `_pick_default_corr_method` from sample code (#3).
#   - unify param naming in benchmark_metrics_against_peers (#4).
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - scipy.stats
#   - statsmodels.tsa.stattools
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, grangercausalitytests

def _pick_default_corr_method(df: pd.DataFrame, metrics):
    """
    Copied from your sample code. (#3)
    """
    unique_counts = [
        pd.to_numeric(df[m], errors='coerce').dropna().nunique()
        for m in metrics if m in df.columns
    ]
    if not unique_counts:
        return "pearson"
    if min(unique_counts)<10:
        return "kendall"
    elif min(unique_counts)<30:
        return "spearman"
    else:
        return "pearson"

def compare_metrics_correlation(df: pd.DataFrame, metrics, method=None):
    """
    Updated to use _pick_default_corr_method if method is None.
    """
    valid_methods=["pearson","spearman","kendall"]
    if method is None:
        method=_pick_default_corr_method(df, metrics)
    elif method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    sub=df[metrics].copy()
    for col in metrics:
        sub[col]=pd.to_numeric(sub[col], errors='coerce')
    sub.dropna(axis=0, how='any', inplace=True)
    if len(sub)<2:
        return pd.DataFrame(np.nan, index=metrics, columns=metrics)

    return sub.corr(method=method)

def benchmark_metrics_against_peers(
    df: pd.DataFrame,
    df_peer: pd.DataFrame,
    date_col: str="date",
    primary_val_col: str="primary_value",
    peer_val_col: str="peer_value",
    method: str="ratio"
) -> pd.DataFrame:
    """
    (Suggested Update #4): unify param naming. 
    Compares primary_val_col vs. peer_val_col by ratio or difference.
    """
    d1 = df[[date_col, primary_val_col]].copy()
    d2 = df_peer[[date_col, peer_val_col]].copy()
    merged = pd.merge(d1, d2, on=date_col, how="inner").dropna()
    if method=="ratio":
        def safe_ratio(row):
            if row[peer_val_col]==0:
                return np.nan
            return row[primary_val_col]/row[peer_val_col]
        merged["benchmark_comparison"]= merged.apply(safe_ratio, axis=1)
    elif method=="difference":
        merged["benchmark_comparison"]= merged[primary_val_col]-merged[peer_val_col]
    else:
        raise ValueError(f"Unknown method: {method}")
    return merged

def compare_means(groupA, groupB, equal_var=True):
    arrA=np.array(groupA, dtype=float)
    arrB=np.array(groupB, dtype=float)
    arrA=arrA[~np.isnan(arrA)]
    arrB=arrB[~np.isnan(arrB)]
    if len(arrA)<2 or len(arrB)<2:
        return {"test_stat":None,"p_value":None,"df":None,"significant":False}
    stat,pval=stats.ttest_ind(arrA,arrB,equal_var=equal_var,nan_policy='omit')
    dof=(len(arrA)+len(arrB)-2) if equal_var else None
    return {
        "test_stat": float(stat),
        "p_value": float(pval),
        "df": dof,
        "significant": (pval<0.05)
    }

def perform_granger_causality_test(
    df: pd.DataFrame,
    target: str,
    exog: str,
    maxlag: int = 4,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Perform a Granger causality test to determine whether the time series in 'exog' help predict 'target'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data.
    target : str
        The name of the target variable column.
    exog : str
        The name of the exogenous variable column.
    maxlag : int, default 4
        Maximum number of lags to test.
    verbose : bool, default False
        If True, prints detailed output from the test.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the Granger causality test results for each lag with columns:
        'lag', 'F_stat', 'p_value', 'df_num', and 'df_denom'.
    """
    if target not in df.columns or exog not in df.columns:
        raise ValueError("Both target and exog columns must be present in the DataFrame.")

    test_data = df[[target, exog]].dropna()
    results = grangercausalitytests(test_data, maxlag=maxlag, verbose=verbose)
    result_list = []
    for lag, res in results.items():
        # Extract the F-test result; it returns a tuple: (F, p-value, df_num, df_denom)
        f_test = res[0].get('ssr_ftest', None)
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
        DataFrame containing the time series data.
    col1 : str
        Name of the first time series column.
    col2 : str
        Name of the second time series column.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
            - 'test_stat': The cointegration test statistic.
            - 'p_value': The p-value from the cointegration test.
            - 'critical_values': Critical values for the test at 1%, 5%, and 10% significance levels.
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Both col1 and col2 must be present in the DataFrame.")
    data = df[[col1, col2]].dropna()
    series1 = pd.to_numeric(data[col1], errors='coerce')
    series2 = pd.to_numeric(data[col2], errors='coerce')
    test_stat, p_value, critical_values = coint(series1, series2)
    return {
        "test_stat": test_stat,
        "p_value": p_value,
        "critical_values": critical_values
    }