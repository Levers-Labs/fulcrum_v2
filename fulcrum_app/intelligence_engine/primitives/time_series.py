import pandas as pd
from typing import Optional

def calculate_pop_growth(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """
    Calculate period-over-period growth for each row relative to the previous row.
    Assumes df is sorted by date ascending and has columns ["date", value_col].
    Returns a new df with an additional column "pop_growth".
    """
    df = df.copy()
    df["pop_growth"] = df[value_col].pct_change() * 100.0
    return df

def classify_metric_status(
    df: pd.DataFrame,
    value_col: str = "value",
    target_col: str = "target",
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Classify each row as "on_track" or "off_track" if value is within +/- threshold of target.
      - If value >= (1 - threshold)*target, label "on_track"
      - else label "off_track"
    Returns a new df with an additional column "status".
    """
    df = df.copy()
    def classify_row(row):
        val = row[value_col]
        tgt = row[target_col]
        if tgt == 0:
            return "unknown"
        if val >= (tgt * (1 - threshold)):
            return "on_track"
        else:
            return "off_track"

    df["status"] = df.apply(classify_row, axis=1)
    return df

def detect_anomaly_with_variance(
    df: pd.DataFrame, 
    value_col: str = "value",
    window: int = 7,
    z_thresh: float = 3.0
) -> pd.DataFrame:
    """
    Flag points if they're beyond z_thresh standard deviations from a rolling mean window.
    Returns a new df with "is_anomaly" boolean column.
    """
    df = df.copy()
    df["rolling_mean"] = df[value_col].rolling(window).mean()
    df["rolling_std"] = df[value_col].rolling(window).std()

    def detect(row):
        mean_val = row["rolling_mean"]
        std_val = row["rolling_std"]
        val = row[value_col]
        if pd.isna(mean_val) or pd.isna(std_val):
            return False
        if abs(val - mean_val) > z_thresh * std_val:
            return True
        return False

    df["is_anomaly"] = df.apply(detect, axis=1)
    return df
