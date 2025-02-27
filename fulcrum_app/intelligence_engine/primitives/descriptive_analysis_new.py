# =============================================================================
# DescriptiveStats
#
# Incorporates advanced descriptive stats from your sample code:
# skew, kurtosis, IQR, outlier detection (zscore_threshold, iqr_multiplier).
#
# Dependencies:
#   - numpy as np
#   - pandas as pd
#   - scipy.stats (skew, kurtosis, median_abs_deviation)
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, median_abs_deviation

def calculate_descriptive_stats(
    array,
    zscore_threshold=3.0,
    iqr_multiplier=1.5,
    percentiles=None
):
    """
    Purpose: Compute advanced summary statistics (min, max, mean, median, std, quartiles, skew, kurtosis, etc.).

    Implementation Details:
    1. Accepts a numeric array or Series.
    2. Filters out NaNs or invalids.
    3. Computes many stats: min, max, mean, median, std, variance, skew, kurtosis, etc.
    4. Returns a dictionary with these stats, plus outlier counts by z-score & IQR.

    Parameters
    ----------
    array : array-like
    zscore_threshold : float
        threshold for z-score outlier detection
    iqr_multiplier : float
        multiplier for IQR-based outlier detection
    percentiles : list of float, optional
        extra percentiles to compute, e.g. [10, 25, 50, 75, 90]

    Returns
    -------
    dict
        Contains all computed metrics.
    """
    s = pd.Series(array).dropna()
    total_count = len(s)
    if total_count == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "variance": None,
            "skew": None,
            "kurtosis": None,
            "iqr": None,
            "mad": None,
            "outlier_count_z": 0,
            "outlier_count_iqr": 0,
            "mode": [],
        }

    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    arr = s.to_numpy()
    desc = s.describe(percentiles=[p/100 for p in percentiles if 0 < p < 100])

    min_val = float(desc["min"])
    max_val = float(desc["max"])
    mean_val = float(desc["mean"])
    median_val = float(s.median())
    std_val = float(s.std())  # ddof=1 by default
    variance_val = std_val**2

    # Skew, kurt
    skew_val = float(skew(arr, bias=False))
    kurt_val = float(kurtosis(arr, bias=False))

    # IQR
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    iqr_val = q75 - q25

    # MAD
    mad_val = float(median_abs_deviation(arr, scale='normal'))

    # z-score outlier detection
    outlier_count_z = 0
    if std_val > 1e-12:
        z_scores = np.abs((arr - mean_val) / std_val)
        outlier_count_z = int((z_scores > zscore_threshold).sum())

    # IQR-based outlier detection
    lower_fence = q25 - iqr_multiplier * iqr_val
    upper_fence = q75 + iqr_multiplier * iqr_val
    outlier_count_iqr = int(((arr < lower_fence) | (arr > upper_fence)).sum())

    # Mode
    mode_series = s.mode()
    mode_list = mode_series.tolist()

    stats = {
        "count": total_count,
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "std": std_val,
        "variance": variance_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "iqr": iqr_val,
        "mad": mad_val,
        "outlier_count_z": outlier_count_z,
        "outlier_count_iqr": outlier_count_iqr,
        "mode": mode_list,
    }

    # Insert percentiles
    for p in percentiles:
        key = f"p{int(p)}"
        if key in stats:
            continue
        val = np.percentile(arr, p)
        stats[key] = float(val)

    return stats
