import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import skew, kurtosis, median_abs_deviation

def calculate_descriptive_stats(
    df: pd.DataFrame, 
    value_col: str = "value",
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Compute a suite of robust descriptive statistics for the given column in df,
    including:
      - Basic distribution stats (count, min, max, median, std, etc.)
      - Coefficient of variation (CV)
      - Two outlier counts: Z-score-based and IQR-based.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing at least one numeric column specified by value_col.
    value_col : str, default 'value'
        Name of the numeric column to analyze.
    zscore_threshold : float, default 3.0
        Outlier threshold in terms of standard deviations from the mean for z-score outlier counting.
    iqr_multiplier : float, default 1.5
        Multiplier for the IQR-based outlier fence, typically 1.5 for a "mild" outlier fence.

    Returns
    -------
    Dict[str, Any]
        A dictionary of descriptive statistics that a 'best analyst' might want, including:
        - count (int): number of non-null, non-infinite values
        - null_count (int): number of NaN values
        - inf_count (int): number of +/- infinite values
        - min (float or None)
        - q25 (float or None)
        - median (float or None)
        - q75 (float or None)
        - max (float or None)
        - mean (float or None)
        - std (float or None)
        - iqr (float or None)
        - mad (float or None)
        - skew (float or None)
        - kurtosis (float or None)
        - cv (float or None): coefficient of variation (std / mean), None if mean=0
        - outlier_count_z (int): number of values exceeding ±(zscore_threshold * std)
        - outlier_count_iqr (int): number of values outside [Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR]

    Notes
    -----
    - If the column is entirely empty or non-numeric, this returns mostly None or zero for certain stats.
    - By default, this uses sample-based (not population-based) formulas for std, skew, and kurtosis.
    - Infinite values are counted in 'inf_count' but excluded from the numeric calculations.
    - By default, outlier thresholds (3 std-dev, 1.5 IQR) match common heuristics, but you can override them.
    """
    series = df[value_col]

    # Count NaNs and Infs specifically
    null_count = series.isna().sum()
    inf_count = np.isinf(series).sum()

    # Filter out all infinite values by converting them to NaN
    clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean_series)

    # If no valid data points remain, return an "empty" stats dict
    if n == 0:
        return {
            "count": 0,
            "null_count": int(null_count),
            "inf_count": int(inf_count),
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "mean": None,
            "std": None,
            "iqr": None,
            "mad": None,
            "skew": None,
            "kurtosis": None,
            "cv": None,
            "outlier_count_z": 0,
            "outlier_count_iqr": 0
        }

    # Convert to numpy array for SciPy operations
    arr = clean_series.to_numpy()

    # Basic stats
    minimum = float(arr.min())
    maximum = float(arr.max())
    mean_val = float(arr.mean())
    med = float(np.median(arr))
    std_val = float(arr.std(ddof=1))      # sample-based
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr_val = q75 - q25
    skew_val = float(skew(arr, bias=False))
    kurt_val = float(kurtosis(arr, bias=False))
    mad_val = float(median_abs_deviation(arr, scale='normal'))

    # Coefficient of Variation
    if mean_val == 0.0:
        cv = None
    else:
        cv = std_val / abs(mean_val)

    # Outlier Counting

    # 1) Z-score approach: outlier if abs(x - mean) / std > zscore_threshold
    #    If std=0 or missing, outlier_count_z defaults to 0.
    if std_val > 0:
        z_scores = np.abs((arr - mean_val) / std_val)
        outlier_count_z = int((z_scores > zscore_threshold).sum())
    else:
        outlier_count_z = 0

    # 2) IQR approach: outlier if outside [Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR]
    lower_fence = q25 - iqr_multiplier * iqr_val
    upper_fence = q75 + iqr_multiplier * iqr_val
    outlier_mask_iqr = (arr < lower_fence) | (arr > upper_fence)
    outlier_count_iqr = int(outlier_mask_iqr.sum())

    return {
        "count": int(n),
        "null_count": int(null_count),
        "inf_count": int(inf_count),
        "min": minimum,
        "q25": q25,
        "median": med,
        "q75": q75,
        "max": maximum,
        "mean": mean_val,
        "std": std_val,
        "iqr": float(iqr_val),
        "mad": mad_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "cv": cv,
        "outlier_count_z": outlier_count_z,
        "outlier_count_iqr": outlier_count_iqr
    }
