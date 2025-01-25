import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import patsy
from scipy.stats import ttest_ind, chisquare
from statsmodels.tsa.stattools import coint, adfullers

def compare_metrics_correlation(
    df: pd.DataFrame, 
    metrics: List[str],
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute the correlation matrix among multiple metric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the metric columns to analyze.
    metrics : List[str]
        A list of column names in df whose pairwise correlation we want.
    method : str, default='pearson'
        Correlation method. One of ['pearson','spearman','kendall'].

    Returns
    -------
    pd.DataFrame
        The correlation matrix (a square DataFrame) with rows/columns = metrics.

    Notes
    -----
    - If any metric is missing from df or not numeric, we handle it gracefully (return NaN or skip).
    - The correlation method can be 'pearson' (linear), 'spearman' (rank-based), or 'kendall'.
    - This function does a simple pairwise correlation, ignoring missing data.
    """
    valid_methods = ["pearson","spearman","kendall"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    # Filter columns
    sub = df[metrics].copy()
    # Possibly drop rows that have NaN in any metric
    sub.dropna(axis=0, how="any", inplace=True)
    # If sub is empty or only 1 row, correlation might be undefined
    if len(sub) < 2:
        # Return a DataFrame of NaN?
        return pd.DataFrame(np.nan, index=metrics, columns=metrics)

    corr_mat = sub.corr(method=method)
    return corr_mat

def detect_metric_predictive_significance(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str,
    max_lag: int = 5,
    add_const: bool = True
) -> dict:
    """
    Check if changes in x_col Granger-cause changes in y_col using a time-series approach.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [x_col, y_col] sorted by time (index or a separate date_col).
    x_col : str
        The potential leading metric.
    y_col : str
        The dependent metric we test if x_col can predict.
    max_lag : int, default=5
        The maximum lag to check in Granger causality test.
    add_const : bool, default=True
        Whether to add a constant in the regression.

    Returns
    -------
    dict
        A summary of p-values and significance for each lag up to max_lag. 
        Example:
        {
          'best_lag': 2,
          'p_value': 0.01,
          'all_lags': {
             lag: { 'ssr_chi2test': (stat, pval, df), ... },
             ...
          }
        }

    Notes
    -----
    - We rely on statsmodels.tsa.stattools.grangercausalitytests. 
      The df must be numeric and free of NaNs. Sort by time ascending first.
    - If p-value at a certain lag is below a threshold, we say "x_col Granger-causes y_col" at that lag.
    """
    data = df[[x_col, y_col]].dropna().copy()
    if len(data) < (max_lag+1):
        return {"error": "Not enough data to run Granger causality."}

    # Statsmodels expects the data shape: columns [y, x], so we reorder
    # Because it tests if x causes y => [y, x]
    arranged = data[[y_col, x_col]]

    results = grangercausalitytests(arranged, max_lag, addconst=add_const, verbose=False)
    # parse results
    all_lags_info = {}
    best_lag = None
    best_pval = 1.0
    for lag, test_res in results.items():
        # test_res is a dictionary of different tests, e.g. 'ssr_chi2test', 'ssr_ftest', etc.
        # We'll pick one, say 'ssr_chi2test'
        ssr_chi2test = test_res['ssr_chi2test']
        stat, pval, df_l, _ = ssr_chi2test
        all_lags_info[lag] = {
            "ssr_chi2test": (stat, pval, df_l)
        }
        if pval < best_pval:
            best_pval = pval
            best_lag = lag

    return {
        "best_lag": best_lag,
        "best_p_value": best_pval,
        "all_lags": all_lags_info
    }

def analyze_metric_interactions(
    df: pd.DataFrame,
    target_col: str,
    driver_cols: list,
    interaction_pairs: list
) -> dict:
    """
    Evaluate synergy between multiple drivers by including interaction terms in a regression.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [target_col] and [driver_cols].
    target_col : str
        The name of the dependent variable y.
    driver_cols : list of str
        The base driver columns (e.g. X1, X2).
    interaction_pairs : list of tuples
        e.g. [('X1','X2'), ('X1','X3')], indicating we want X1*X2, etc.

    Returns
    -------
    dict
        {
          'model_summary': str (text summary of statsmodels results),
          'interaction_terms': {
              'X1_X2': {
                 'coef': float,
                 'p_value': float
              }, ...
          }
        }

    Notes
    -----
    - We'll use a formula approach with patsy. For example y ~ X1 + X2 + X1:X2 for an interaction.
    - Real usage might do polynomial expansions or multiple interactions.
    """
    # Build a formula string: y ~ X1 + X2 + X1:X2
    # If we have multiple pairs, we can do something like X1:X2 + X1:X3 + ...
    base_terms = " + ".join(driver_cols)
    inter_terms = " + ".join([f"{a}:{b}" for (a,b) in interaction_pairs])
    formula = f"{target_col} ~ {base_terms}"
    if inter_terms:
        formula += " + " + inter_terms

    # Construct design matrices
    y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')
    model = sm.OLS(y, X).fit()

    # parse results
    summary_str = model.summary().as_text()
    interaction_info = {}
    for (a,b) in interaction_pairs:
        term = f"{a}:{b}"
        if term in model.params.index:
            coef = model.params[term]
            pval = model.pvalues[term]
            interaction_info[f"{a}_{b}"] = {
                "coef": float(coef),
                "p_value": float(pval)
            }
        else:
            interaction_info[f"{a}_{b}"] = {
                "coef": None,
                "p_value": None
            }

    return {
        "model_summary": summary_str,
        "interaction_terms": interaction_info
    }

def benchmark_metrics_against_peers(
    df: pd.DataFrame,
    df_peer: pd.DataFrame,
    date_col: str = "date",
    my_val_col: str = "my_value",
    peer_val_col: str = "peer_value",
    method: str = "ratio"
) -> pd.DataFrame:
    """
    Compare your metric vs. a peer/benchmark metric over the same time range.

    Parameters
    ----------
    df : pd.DataFrame
        e.g. columns=[date_col, my_val_col].
    df_peer : pd.DataFrame
        e.g. columns=[date_col, peer_val_col].
    date_col : str, default='date'
        The column to merge on.
    my_val_col : str, default='my_value'
    peer_val_col : str, default='peer_value'
    method : str, default='ratio'
        - 'ratio' => (my_value / peer_value)
        - 'difference' => (my_value - peer_value)

    Returns
    -------
    pd.DataFrame
        columns = [date_col, my_val_col, peer_val_col, 'benchmark_diff_or_ratio'].
    """
    d1 = df[[date_col, my_val_col]].copy()
    d2 = df_peer[[date_col, peer_val_col]].copy()

    merged = pd.merge(d1, d2, on=date_col, how="inner").dropna()
    if method == "ratio":
        merged["benchmark_comparison"] = merged.apply(
            lambda row: row[my_val_col]/row[peer_val_col] if row[peer_val_col]!=0 else None, axis=1
        )
    elif method == "difference":
        merged["benchmark_comparison"] = merged[my_val_col] - merged[peer_val_col]
    else:
        raise ValueError(f"Unknown method: {method}")

    return merged

def detect_statistical_significance(
    groupA: list,
    groupB: list,
    test_type: str = "t-test",
    equal_var: bool = True
) -> dict:
    """
    Check if two groups differ significantly (t-test, chi-square, etc.).

    Parameters
    ----------
    groupA : list or array-like
        The first sample
    groupB : list or array-like
        The second sample
    test_type : str, default='t-test'
        't-test' => two-sample t-test,
        'chi-square' => if they are frequency counts for categories (less common here).
    equal_var : bool, default=True
        For t-test, assume equal variance. If not, set to False => Welch's t-test.

    Returns
    -------
    dict
        {
          'test_stat': float,
          'p_value': float,
          'df': (if available),
          'significant': bool
        }

    Notes
    -----
    - If test_type='t-test', we call ttest_ind. If either group is too small, we return None.
    - If test_type='chi-square', we call chisquare. 
      Typically you'd pass groupA as observed freq, groupB as expected freq or vice versa.
    """
    import numpy as np

    arrA = np.array(groupA)
    arrB = np.array(groupB)
    arrA = arrA[~np.isnan(arrA)]
    arrB = arrB[~np.isnan(arrB)]
    if len(arrA) < 2 or len(arrB) < 2:
        return {"test_stat": None, "p_value": None, "significant": False, "df": None}

    if test_type == "t-test":
        stat, pval = ttest_ind(arrA, arrB, equal_var=equal_var, nan_policy='omit')
        # degrees of freedom is approximate for Welch, exact for standard
        # For a standard two-sample t-test with n1+n2-2 dof, but let's skip or do approximate:
        dof = len(arrA)+len(arrB)-2 if equal_var else None
        significant = (pval < 0.05)
        return {
            "test_stat": float(stat),
            "p_value": float(pval),
            "significant": significant,
            "df": dof
        }
    elif test_type == "chi-square":
        # assume arrA is observed freq, arrB is expected freq or some scenario
        # often you'd do chisquare(observed, f_exp=expected)
        stat, pval = chisquare(arrA, f_exp=arrB)
        dof = len(arrA)-1
        significant = (pval < 0.05)
        return {
            "test_stat": float(stat),
            "p_value": float(pval),
            "significant": significant,
            "df": dof
        }
    else:
        return {"error": f"Unknown test_type {test_type}"}

def test_cointegration(
    df: pd.DataFrame, 
    colA: str, 
    colB: str
) -> dict:
    """
    Checks whether two series (colA, colB) in df are cointegrated via Engle-Granger test.

    Returns
    -------
    dict
      {
        'p_value': float,
        'test_stat': float,
        'critical_values': ...
      }
    """
    data = df[[colA,colB]].dropna()
    if len(data) < 10:
        return {"error": "Not enough data to test cointegration."}

    seriesA = data[colA]
    seriesB = data[colB]
    stat, pval, crit_values = coint(seriesA, seriesB)
    return {
        "p_value": float(pval),
        "test_stat": float(stat),
        "critical_values": crit_values
    }

def detect_lagged_influence(
    df: pd.DataFrame,
    colA: str,
    colB: str,
    max_lag: int = 10
) -> dict:
    """
    Simple cross-correlation approach to find best lag between colA and colB.

    Returns
    -------
    dict
      { 'best_lag': int, 'best_corr': float, 'all_lags': {lag: corr} }
    """
    data = df[[colA,colB]].dropna()
    A = data[colA].to_numpy()
    B = data[colB].to_numpy()
    # we can do a naive approach: shift B by each lag, compute correlation with A
    best_lag = 0
    best_corr = 0.0
    all_lags = {}

    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            # negative lag => shift B forward
            A_sub = A[:lag]  # up to lag from end
            B_sub = B[-lag:]
        else:
            # positive or zero lag => shift B backward
            A_sub = A[lag:]
            B_sub = B[:len(A_sub)]
        if len(A_sub) < 2:
            continue
        corr = np.corrcoef(A_sub, B_sub)[0,1]
        all_lags[lag] = float(corr)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return {
      "best_lag": best_lag,
      "best_corr": best_corr,
      "all_lags": all_lags
    }