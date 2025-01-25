import pandas as pd
import numpy as np

def analyze_metric_trend(df: pd.DataFrame, value_col: str="value", slope_threshold: float=0.0) -> dict:
    """
    Classify the entire series as upward/downward/stable, 
    returning { 'trend': 'up'|'down'|'stable', 'slope':..., 'r_value':... }
    """
    from scipy.stats import linregress
    df2 = df.copy().sort_values("date")
    df2["t"] = np.arange(len(df2))
    slope, intercept, r_value, p_value, std_err = linregress(df2["t"], df2[value_col])
    trend = "stable"
    if slope > slope_threshold:
        trend = "up"
    elif slope < -slope_threshold:
        trend = "down"
    return {
        "trend": trend,
        "slope": slope,
        "r_value": r_value
    }

def detect_trend_changes(df: pd.DataFrame, value_col: str="value", window_size: int=5) -> pd.DataFrame:
    """
    Locate points where the trend slope changes significantly by using a rolling window slope.
    Adds a 'trend_change' boolean column. This is a simplistic approach.
    """
    from scipy.stats import linregress
    df = df.copy().sort_values("date")
    # We'll compute slope in each rolling window
    slopes = []
    for i in range(len(df)):
        start = max(0, i-window_size+1)
        sub = df.iloc[start:i+1]
        if len(sub) < 2:
            slopes.append(np.nan)
            continue
        sub = sub.copy()
        sub["t"] = np.arange(len(sub))
        slope, *_ = linregress(sub["t"], sub[value_col])
        slopes.append(slope)
    df["slope"] = slopes
    df["prev_slope"] = df["slope"].shift(1)
    df["trend_change"] = (df["slope"].notna()) & (df["prev_slope"].notna()) & \
                         ((df["slope"] * df["prev_slope"]) < 0)  # sign flip
    return df

def detect_new_trend_direction(df: pd.DataFrame, slope_col: str="slope") -> str:
    """
    If the slope recently flipped from negative to positive or vice versa.
    Returns "new_upward", "new_downward", or "no_change".
    """
    if len(df) < 2 or slope_col not in df.columns:
        return "no_change"
    last_slope = df[slope_col].iloc[-1]
    prev_slope = df[slope_col].iloc[-2]
    if prev_slope < 0 and last_slope > 0:
        return "new_upward"
    elif prev_slope > 0 and last_slope < 0:
        return "new_downward"
    else:
        return "no_change"

def detect_performance_plateau(df: pd.DataFrame, value_col: str="value", tolerance: float=0.01, window: int=7) -> bool:
    """
    Identify if the last 'window' points are within a tight band.
    Returns True if (max-min)/avg < tolerance.
    """
    if len(df) < window:
        return False
    sub = df[value_col].tail(window)
    avg = sub.mean()
    if avg == 0:
        return False
    if (sub.max() - sub.min()) / avg < tolerance:
        return True
    return False

def detect_record_high(df: pd.DataFrame, value_col: str="value") -> bool:
    """
    Check if the latest value is at a record high.
    Returns True/False.
    """
    if len(df) == 0:
        return False
    latest = df[value_col].iloc[-1]
    highest = df[value_col].max()
    return np.isclose(latest, highest)

def detect_record_low(df: pd.DataFrame, value_col: str="value") -> bool:
    """
    Check if the latest value is at a record low.
    """
    if len(df) == 0:
        return False
    latest = df[value_col].iloc[-1]
    lowest = df[value_col].min()
    return np.isclose(latest, lowest)

def detect_anomaly_with_variance(df: pd.DataFrame, value_col: str="value", window: int=7, z_thresh: float=3.0) -> pd.DataFrame:
    """
    Implementation same as previously shown.
    """
    df = df.copy()
    df["rolling_mean"] = df[value_col].rolling(window).mean()
    df["rolling_std"] = df[value_col].rolling(window).std()

    def detect(row):
        if pd.isna(row["rolling_mean"]) or pd.isna(row["rolling_std"]):
            return False
        diff = abs(row[value_col] - row["rolling_mean"])
        return diff > z_thresh * row["rolling_std"]

    df["is_anomaly"] = df.apply(detect, axis=1)
    return df

def detect_spc_anomalies(df: pd.DataFrame, value_col: str="value", window: int=7) -> pd.DataFrame:
    """
    Stub for Western Electric SPC rule detection. 
    We'll just do a 3-sigma limit for demonstration.
    """
    df = df.copy()
    rolling_mean = df[value_col].rolling(window).mean()
    rolling_std = df[value_col].rolling(window).std()
    df["UCL"] = rolling_mean + 3.0 * rolling_std
    df["LCL"] = rolling_mean - 3.0 * rolling_std
    df["spc_anomaly"] = (df[value_col] > df["UCL"]) | (df[value_col] < df["LCL"])
    return df

def detect_anomaly_ml(df: pd.DataFrame, value_col: str="value"):
    """
    Placeholder for an ML-based outlier detection (IsolationForest, etc.).
    Returns the DF with an 'is_anomaly_ml' column for demonstration.
    """
    from sklearn.ensemble import IsolationForest
    df = df.copy()
    X = df[[value_col]].dropna()
    if len(X) < 5:
        df["is_anomaly_ml"] = False
        return df
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    preds = model.predict(X)  # -1 => outlier
    df["is_anomaly_ml"] = False
    df.loc[X.index, "is_anomaly_ml"] = (preds == -1)
    return df

def detect_anomalies(df: pd.DataFrame, value_col: str="value"):
    """
    Combine multiple anomaly detection methods in an ensemble approach.
    For brevity, we'll just union 'is_anomaly' + 'spc_anomaly' + 'is_anomaly_ml'.
    """
    df1 = detect_anomaly_with_variance(df, value_col=value_col)
    df2 = detect_spc_anomalies(df1, value_col=value_col)
    df3 = detect_anomaly_ml(df2, value_col=value_col)
    df3["final_anomaly"] = df3["is_anomaly"] | df3["spc_anomaly"] | df3["is_anomaly_ml"]
    return df3

def detect_volatility_spike(df: pd.DataFrame, value_col: str="value", window: int=7, ratio_thresh: float=2.0):
    """
    Identify a sudden increase in rolling std dev by factor (ratio_thresh).
    We'll return a new DF with 'vol_spike' boolean.
    """
    df = df.copy()
    df["rolling_std"] = df[value_col].rolling(window).std()
    df["prev_std"] = df["rolling_std"].shift(1)
    df["vol_spike"] = (df["rolling_std"] > ratio_thresh * df["prev_std"]) & (df["prev_std"] > 0)
    return df
