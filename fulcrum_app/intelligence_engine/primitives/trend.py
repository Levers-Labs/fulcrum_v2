import pandas as pd
import numpy as np
from typing import Optional, Dict
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest

def analyze_metric_trend(
    df: pd.DataFrame, 
    value_col: str = "value", 
    date_col: Optional[str] = None,
    slope_threshold: float = 0.0
) -> dict:
    """
    Classify the entire series as upward/downward/stable by performing a linear regression 
    on either row index or date_col vs. value_col.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric column value_col. Optionally a date_col if we want to treat time as x-axis.
    value_col : str, default='value'
    date_col : str or None, default=None
        If provided, parse it as datetime, sort ascending, then convert to a numeric offset for regression.
        If None, we just use row indices (0..N-1).
    slope_threshold : float, default=0.0
        A minimum slope magnitude to label "up"/"down". If slope < slope_threshold in absolute value => "stable".

    Returns
    -------
    dict
        {
          'trend': 'up'|'down'|'stable',
          'slope': float,
          'r_value': float,
          'p_value': float,
          'std_err': float
        }

    Notes
    -----
    - If date_col is provided, we convert date to numeric offsets in days from the earliest date in df.
    - slope_threshold can handle small slopes as "stable."
    """
    dff = df.copy().dropna(subset=[value_col])
    if len(dff) < 2:
        return {"trend":"no_data","slope":0.0,"r_value":0.0,"p_value":None,"std_err":None}

    if date_col:
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff.sort_values(date_col, inplace=True)
        dff.reset_index(drop=True, inplace=True)
        start_date = dff[date_col].iloc[0]
        dff["_x"] = (dff[date_col] - start_date).dt.total_seconds()/86400.0  # in days
    else:
        dff.reset_index(drop=True, inplace=True)
        dff["_x"] = dff.index

    slope, intercept, r_value, p_value, std_err = linregress(dff["_x"], dff[value_col])
    if slope > slope_threshold:
        trend = "up"
    elif slope < -slope_threshold:
        trend = "down"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "slope": slope,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err
    }

def detect_trend_changes(
    df: pd.DataFrame, 
    value_col: str = "value",
    date_col: Optional[str] = None,
    window_size: int = 5,
    slope_change_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Locate points where the trend slope changes significantly by using a rolling window slope. 
    We add columns: 'slope', 'prev_slope', 'trend_change'(bool).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [value_col], optionally date_col for proper sorting.
    value_col : str, default='value'
    date_col : str or None, default=None
        If provided, we'll sort by it ascending before computing.
    window_size : int, default=5
        The number of points in each rolling window to compute slope.
    slope_change_threshold : float, default=0.0
        If we want to label a "change" only if the difference in slopes is > slope_change_threshold.

    Returns
    -------
    pd.DataFrame
        A copy of df with new columns: 
          - 'slope': slope in the last window, 
          - 'prev_slope': slope from the prior window, 
          - 'trend_change': boolean. 
            If slope flips sign or changes more than slope_change_threshold, mark True.

    Notes
    -----
    - This is a simplistic approach that uses linear regression in each sub-window. 
      For a robust approach, you might do a piecewise linear or advanced segmentation.
    """
    dff = df.copy()
    if date_col:
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff.sort_values(date_col, inplace=True)
    else:
        dff.sort_index(inplace=True)

    dff.reset_index(drop=True, inplace=True)

    slopes = []
    from scipy.stats import linregress
    for i in range(len(dff)):
        start = i - window_size + 1
        if start < 0:
            slopes.append(np.nan)
            continue
        sub = dff.iloc[start:i+1]
        # x can be numeric index or a date offset
        x_vals = np.arange(len(sub))
        y_vals = sub[value_col].to_numpy()
        slope, _, _, _, _ = linregress(x_vals, y_vals)
        slopes.append(slope)

    dff["slope"] = slopes
    dff["prev_slope"] = dff["slope"].shift(1)
    def is_change(row):
        if pd.isna(row["slope"]) or pd.isna(row["prev_slope"]):
            return False
        diff = row["slope"] - row["prev_slope"]
        # we detect sign flip or big difference
        sign_flip = (row["slope"]*row["prev_slope"]<0)
        big_diff = abs(diff) > slope_change_threshold
        return sign_flip or big_diff

    dff["trend_change"] = dff.apply(is_change, axis=1)
    return dff

def detect_new_trend_direction(
    df: pd.DataFrame,
    slope_col: str = "slope"
) -> str:
    """
    If the slope recently flipped from negative to positive => 'new_upward',
    or positive to negative => 'new_downward'. Otherwise 'no_change'.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a column slope_col. We'll look at the last two rows for slope.
    slope_col : str, default='slope'

    Returns
    -------
    str
        "new_upward", "new_downward", or "no_change".
    """
    if len(df) < 2 or slope_col not in df.columns:
        return "no_change"
    last_slope = df[slope_col].iloc[-1]
    prev_slope = df[slope_col].iloc[-2]
    if pd.isna(last_slope) or pd.isna(prev_slope):
        return "no_change"

    if prev_slope < 0 and last_slope > 0:
        return "new_upward"
    elif prev_slope > 0 and last_slope < 0:
        return "new_downward"
    else:
        return "no_change"

def detect_performance_plateau(
    df: pd.DataFrame, 
    value_col: str = "value", 
    tolerance: float = 0.01, 
    window: int = 7
) -> bool:
    """
    Identify if the last 'window' points are within a tight band => plateau.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric column value_col, sorted in chronological order.
    value_col : str, default='value'
    tolerance : float, default=0.01
        If (max-min)/mean in the last window is < tolerance => plateau.
    window : int, default=7
        Number of recent points to check.

    Returns
    -------
    bool
        True if a plateau is detected, else False.

    Notes
    -----
    - You might also do a rolling version to find all intervals with a plateau,
      but here we just do the "latest" window.
    """
    if len(df) < window:
        return False
    sub = df[value_col].tail(window).dropna()
    if len(sub) < window:
        return False
    avg = sub.mean()
    if abs(avg) < 1e-12:
        return False
    mm_range = sub.max() - sub.min()
    ratio = mm_range / avg
    return (ratio < tolerance)

def detect_record_high(df: pd.DataFrame, value_col: str = "value") -> bool:
    """
    Check if the latest value is at a record high (ties included).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric column value_col.
    value_col : str, default='value'

    Returns
    -------
    bool
        True if df.iloc[-1][value_col] is the highest in the entire series.
    """
    if len(df) == 0:
        return False
    latest = df[value_col].iloc[-1]
    highest = df[value_col].max()
    return np.isclose(latest, highest) or (latest == highest)


def detect_record_low(df: pd.DataFrame, value_col: str = "value") -> bool:
    """
    Check if the latest value is at a record low.

    Returns True if df.iloc[-1][value_col] is the lowest in the entire series.
    """
    if len(df) == 0:
        return False
    latest = df[value_col].iloc[-1]
    lowest = df[value_col].min()
    return np.isclose(latest, lowest) or (latest == lowest)

def detect_anomaly_with_variance(
    df: pd.DataFrame,
    value_col: str = "value",
    window: int = 7,
    z_thresh: float = 3.0
) -> pd.DataFrame:
    """
    Flag points if they're beyond z_thresh std dev from a rolling mean.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric column value_col, presumably sorted.
    value_col : str
    window : int, default=7
    z_thresh : float, default=3.0

    Returns
    -------
    pd.DataFrame
        A copy of df with columns [rolling_mean, rolling_std, is_anomaly] (bool).

    Notes
    -----
    - If there's insufficient data in the first few rows, rolling mean/std is NaN => is_anomaly=False.
    - This is a simplistic approach; we skip seasonality or drift adjustments.
    """
    dff = df.copy()
    dff["rolling_mean"] = dff[value_col].rolling(window).mean()
    dff["rolling_std"] = dff[value_col].rolling(window).std()

    def check_anom(row):
        if pd.isna(row["rolling_mean"]) or pd.isna(row["rolling_std"]) or row["rolling_std"]==0:
            return False
        return (abs(row[value_col] - row["rolling_mean"]) > z_thresh*row["rolling_std"])

    dff["is_anomaly"] = dff.apply(check_anom, axis=1)
    return dff

def detect_spc_anomalies(
    df: pd.DataFrame, 
    value_col: str = "value",
    window: int = 7
) -> pd.DataFrame:
    """
    Implement a basic SPC rule using ±3 sigma from rolling mean.

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str
    window : int
        rolling window size

    Returns
    -------
    pd.DataFrame
        with columns: [UCL, LCL, spc_anomaly] (bool)

    Notes
    -----
    - Real Western Electric rules also look for runs above/below center, etc.
    """
    dff = df.copy()
    dff["rolling_mean"] = dff[value_col].rolling(window).mean()
    dff["rolling_std"] = dff[value_col].rolling(window).std()
    dff["UCL"] = dff["rolling_mean"] + 3*dff["rolling_std"]
    dff["LCL"] = dff["rolling_mean"] - 3*dff["rolling_std"]

    def check_spc(row):
        if pd.isna(row["rolling_mean"]) or pd.isna(row["rolling_std"]):
            return False
        return (row[value_col] > row["UCL"]) or (row[value_col] < row["LCL"])

    dff["spc_anomaly"] = dff.apply(check_spc, axis=1)
    return dff

def detect_anomaly_ml(
    df: pd.DataFrame, 
    value_col: str="value",
    contamination: float=0.05
) -> pd.DataFrame:
    """
    Apply an ML-based outlier detection (IsolationForest) on the single dimension value_col.

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str
    contamination : float, default=0.05
        The proportion of outliers in data to find.

    Returns
    -------
    pd.DataFrame
        A copy of df with a new boolean column 'is_anomaly_ml'.

    Notes
    -----
    - For multi-dimensional outlier detection, we'd pass multiple columns to X.
    - Single-dimensional is less interesting for ML, but still feasible.
    """
    dff = df.copy().dropna(subset=[value_col])
    if len(dff) < 5:
        dff["is_anomaly_ml"] = False
        return dff

    X = dff[[value_col]]
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    preds = model.predict(X)  # -1 => outlier
    dff["is_anomaly_ml"] = (preds==-1)
    return dff

def detect_anomalies(
    df: pd.DataFrame, 
    value_col: str="value"
) -> pd.DataFrame:
    """
    Combine multiple anomaly detection methods in an ensemble approach:
    - variance-based
    - spc
    - ml
    Mark final_anomaly = True if any method flags an anomaly.

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str

    Returns
    -------
    pd.DataFrame
        Columns: [is_anomaly_variance, spc_anomaly, is_anomaly_ml, final_anomaly]
    """
    dff = df.copy()
    # 1) variance
    var_df = detect_anomaly_with_variance(dff, value_col=value_col)
    dff["is_anomaly_variance"] = var_df["is_anomaly"]

    # 2) spc
    spc_df = detect_spc_anomalies(dff, value_col=value_col)
    dff["spc_anomaly"] = spc_df["spc_anomaly"]

    # 3) ml
    ml_df = detect_anomaly_ml(dff, value_col=value_col)
    dff["is_anomaly_ml"] = ml_df["is_anomaly_ml"]

    # final => union (any method => True)
    dff["final_anomaly"] = dff["is_anomaly_variance"] | dff["spc_anomaly"] | dff["is_anomaly_ml"]

    return dff

def detect_volatility_spike(
    df: pd.DataFrame, 
    value_col: str="value", 
    window: int=7, 
    ratio_thresh: float=2.0
) -> pd.DataFrame:
    """
    Identify a sudden increase in rolling std dev by factor ratio_thresh. 
    We add columns [rolling_std, prev_std, vol_spike].

    Parameters
    ----------
    df : pd.DataFrame
        Must have numeric column value_col.
    value_col : str, default='value'
    window : int, default=7
    ratio_thresh : float, default=2.0
        If rolling_std(t)/rolling_std(t-1) > ratio_thresh => vol_spike=True

    Returns
    -------
    pd.DataFrame
        with columns [rolling_std, prev_std, vol_spike].
    """
    dff = df.copy()
    dff["rolling_std"] = dff[value_col].rolling(window).std()
    dff["prev_std"] = dff["rolling_std"].shift(1)

    def check_spike(row):
        if pd.isna(row["rolling_std"]) or pd.isna(row["prev_std"]):
            return False
        if row["prev_std"]==0:
            return False
        ratio = row["rolling_std"]/row["prev_std"]
        return ratio > ratio_thresh

    dff["vol_spike"] = dff.apply(check_spike, axis=1)
    return dff