import pandas as pd

def calculate_metric_gva(actual: float, target: float) -> dict:
    """
    Compute Goal vs Actual difference at a single time point.
    Returns { 'abs_diff':..., 'pct_diff':... }
    """
    if target == 0:
        return {"abs_diff": None, "pct_diff": None}
    abs_diff = actual - target
    pct_diff = abs_diff / target * 100.0
    return {"abs_diff": abs_diff, "pct_diff": pct_diff}

def calculate_historical_gva(df_actual: pd.DataFrame, df_target: pd.DataFrame,
                             value_col: str = "value") -> pd.DataFrame:
    """
    Compute GvA across a time series. Expects two DataFrames with columns: [date, value].
    Returns a merged DF with abs_gva, pct_gva columns.
    """
    merged = pd.merge(df_actual, df_target, on="date", how="left", suffixes=("_actual","_target"))
    merged["abs_gva"] = merged[f"{value_col}_actual"] - merged[f"{value_col}_target"]
    merged["pct_gva"] = (merged["abs_gva"] / merged[f"{value_col}_target"]) * 100.0
    return merged

def classify_metric_status(row_val: float, row_target: float, threshold: float=0.05) -> str:
    """
    For a single row with an actual value and target, returns "on_track" or "off_track".
    If actual >= (1 - threshold)*target => "on_track", else "off_track".
    """
    if row_target == 0:
        return "no_target"
    if row_val >= (row_target * (1 - threshold)):
        return "on_track"
    else:
        return "off_track"

def detect_status_changes(df: pd.DataFrame, status_col: str = "status") -> pd.DataFrame:
    """
    Identify rows where status flips from one to another.
    Adds a boolean 'status_flip' column.
    """
    df = df.copy()
    df["prev_status"] = df[status_col].shift(1)
    df["status_flip"] = (df[status_col] != df["prev_status"]) & (df["prev_status"].notna())
    return df

def track_status_durations(df: pd.DataFrame, status_col: str = "status") -> pd.DataFrame:
    """
    Calculate consecutive runs of the same status.
    Returns a DF with columns [start_idx, end_idx, status, run_length].
    """
    results = []
    current_status = None
    start_idx = 0
    for i, row in df.iterrows():
        if row[status_col] != current_status:
            if current_status is not None:
                # finish previous run
                results.append({
                    "start_idx": start_idx,
                    "end_idx": i-1,
                    "status": current_status,
                    "run_length": i - start_idx
                })
            current_status = row[status_col]
            start_idx = i
    # close the final run if needed
    if current_status is not None:
        results.append({
            "start_idx": start_idx,
            "end_idx": len(df)-1,
            "status": current_status,
            "run_length": len(df) - start_idx
        })
    return pd.DataFrame(results)

def monitor_threshold_proximity(val: float, target: float, margin: float=0.05) -> bool:
    """
    Returns True if val is within +/- margin of target.
    """
    if target == 0:
        return False
    diff_pct = abs(val - target) / abs(target)
    return diff_pct <= margin

def calculate_required_growth(current_value: float, target_value: float, periods_left: int) -> float:
    """
    Determine needed compound growth to reach target_value from current_value over periods_left.
    Returns rate as decimal, e.g. 0.02 => 2% growth per period.
    """
    if current_value <= 0 or target_value <= 0 or periods_left <= 0:
        return 0
    rate = (target_value / current_value) ** (1.0 / periods_left) - 1
    return rate
