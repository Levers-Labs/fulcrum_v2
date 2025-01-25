import pandas as pd
import math
from typing import Optional

def calculate_metric_gva(
    actual: float, 
    target: float, 
    allow_negative_target: bool = False
) -> dict:
    """
    Compute the difference between an actual metric value and its target.

    Parameters
    ----------
    actual : float
        The actual observed value.
    target : float
        The target value. Typically positive, but can be zero or negative
        if allow_negative_target is True.
    allow_negative_target : bool, default=False
        If True, we allow negative or zero targets. In such scenarios, the
        percentage difference is computed carefully:
          pct_diff = (actual - target) / abs(target) * 100
        If False and target <= 0, we'll return None for pct_diff.

    Returns
    -------
    dict
        {
            'abs_diff': float or None,
            'pct_diff': float or None
        }
        abs_diff = actual - target
        pct_diff = (abs_diff / target)*100 if (target != 0 and >0),
                   else (abs_diff / abs(target))*100 if allow_negative_target=True,
                   else None if target <= 0 and allow_negative_target=False

    Notes
    -----
    - If target is zero or negative and not allowed, we return None for pct_diff
      to signal an invalid scenario.
    - For correct usage with negative targets, set allow_negative_target=True.
    """
    abs_diff = actual - target

    if target == 0 and not allow_negative_target:
        # Standard approach: no well-defined percentage difference
        return {
            "abs_diff": abs_diff,
            "pct_diff": None
        }
    elif target <= 0 and not allow_negative_target:
        # We do not handle negative or zero properly => yield None for pct_diff
        return {
            "abs_diff": abs_diff,
            "pct_diff": None
        }
    else:
        # Safe to compute ratio vs. abs(target)
        if target == 0:
            # but we do allow negative or zero => actual - 0 => ratio vs. 0 is infinite
            # We'll define a safe approach
            if actual == 0:
                pct = 0.0
            else:
                pct = float('inf') if actual > 0 else float('-inf')
        else:
            pct = (abs_diff / abs(target)) * 100.0

        return {
            "abs_diff": abs_diff,
            "pct_diff": pct
        }

def calculate_historical_gva(
    df_actual: pd.DataFrame,
    df_target: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    allow_negative_target: bool = False
) -> pd.DataFrame:
    """
    Compute GvA (Goal vs. Actual) across a time series. Expects two DataFrames:
      df_actual : columns = [date_col, value_col]
      df_target : columns = [date_col, value_col]
    Merges them on date_col, then calculates:
      abs_gva = actual_value - target_value
      pct_gva = (abs_gva / abs(target_value)) * 100  (if target_value != 0 or allow_negative_target=True)

    Parameters
    ----------
    df_actual : pd.DataFrame
        Must have columns [date_col, value_col].
    df_target : pd.DataFrame
        Must have columns [date_col, value_col].
    date_col : str, default='date'
        Name of the datetime column used for merging.
    value_col : str, default='value'
        Name of the numeric column in each DF that represents the actual or target values.
    allow_negative_target : bool, default=False
        If True, negative or zero targets are allowed and pct_gva is computed
        relative to abs(target_value). Otherwise, if target <= 0, pct_gva is None.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame with columns:
          [date_col, value_col+'_actual', value_col+'_target', 'abs_gva', 'pct_gva'].
        If target is <= 0 and allow_negative_target=False, 'pct_gva' is None for that row.

    Notes
    -----
    - The function merges on date_col with a left join from df_actual. 
      (You can adjust the join type if needed.)
    - If rows exist in df_target that don't match df_actual's date, they'll be dropped unless
      you change the merge or handle missing data differently.
    - A typical usage might be daily time series for actual vs. target.
    """
    # Ensure the date columns are recognized as datetime, if needed
    # df_actual[date_col] = pd.to_datetime(df_actual[date_col])
    # df_target[date_col] = pd.to_datetime(df_target[date_col])

    merged = pd.merge(
        df_actual[[date_col, value_col]],
        df_target[[date_col, value_col]],
        on=date_col,
        how="left",
        suffixes=("_actual", "_target")
    )

    def compute_gva(row):
        act = row[f"{value_col}_actual"]
        tgt = row[f"{value_col}_target"]
        if pd.isna(tgt) or pd.isna(act):
            return pd.Series([None, None])

        abs_diff = act - tgt
        if tgt == 0 and not allow_negative_target:
            return pd.Series([abs_diff, None])
        elif tgt <= 0 and not allow_negative_target:
            return pd.Series([abs_diff, None])
        else:
            if tgt == 0:
                # allow negative or zero => handle ratio
                if act == 0:
                    pct = 0.0
                else:
                    pct = float('inf') if act > 0 else float('-inf')
            else:
                pct = (abs_diff / abs(tgt)) * 100.0
            return pd.Series([abs_diff, pct])

    merged[["abs_gva", "pct_gva"]] = merged.apply(compute_gva, axis=1)

    return merged

def classify_metric_status(
    row_val: float, 
    row_target: float, 
    threshold: float = 0.05, 
    allow_negative_target: bool = False, 
    status_if_no_target: str = "no_target"
) -> str:
    """
    Classify a single row as "on_track" or "off_track" based on threshold rules.

    Parameters
    ----------
    row_val : float
        The actual observed value.
    row_target : float
        The target value.
    threshold : float, default=0.05
        A fraction representing allowable slack (5% default).
        If row_val >= (1 - threshold)*row_target (for positive targets), we label "on_track".
    allow_negative_target : bool, default=False
        If True, negative or zero targets are handled by comparing row_val >= row_target*(1 - threshold)
        with sign checks. If False and row_target <= 0, we return 'no_target' or other fallback.
    status_if_no_target : str, default='no_target'
        The status label to return if row_target <= 0 (when allow_negative_target=False) or is None.

    Returns
    -------
    str
        "on_track" or "off_track" or `status_if_no_target`.

    Notes
    -----
    - If row_target <= 0 and allow_negative_target=False, we assume no valid target => return status_if_no_target.
    - If row_target > 0, the usual formula for on_track is:
        row_val >= row_target*(1 - threshold).
    - If allow_negative_target=True and row_target < 0, we do:
        row_val <= row_target*(1 + threshold) 
      because for negative targets, being "on track" means not exceeding the negative threshold
      (i.e. more negative is "bad"?). This can be domain-specific, so tweak as needed.
    """
    if row_target is None or pd.isna(row_target):
        return status_if_no_target

    if row_target <= 0 and not allow_negative_target:
        return status_if_no_target

    # Positive target scenario:
    if row_target > 0:
        # on track if row_val >= row_target * (1 - threshold)
        cutoff = row_target * (1 - threshold)
        return "on_track" if row_val >= cutoff else "off_track"

    else:
        # Negative target scenario if allow_negative_target
        # Possibly the user wants "on_track" if row_val is not too negative:
        # e.g. row_val >= row_target*(1 - threshold)? or row_val <= row_target*(1 + threshold)?
        # We define an example approach that "more negative" is "worse," so being above the threshold is good.
        # This logic is domain specific — adapt as needed.
        cutoff = row_target * (1 - threshold)
        # If row_val >= cutoff => "on_track", else "off_track"
        # E.g. target=-100, threshold=0.05 => cutoff=-95 => if row_val > -95 => on_track
        # This might invert the sense of the threshold. Tweak as needed for your domain.
        return "on_track" if row_val >= cutoff else "off_track"

def detect_status_changes(
    df: pd.DataFrame, 
    status_col: str = "status", 
    sort_by_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Identify rows where status flips from one to another (e.g., off_track -> on_track).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least one column (status_col).
    status_col : str, default='status'
        Column name holding string statuses (e.g. 'on_track', 'off_track', 'no_target').
    sort_by_date : str, optional
        If provided, the function will first sort by this column before detecting flips.
        Useful if the DataFrame isn't already sorted chronologically.

    Returns
    -------
    pd.DataFrame
        A copy of df with two added columns:
          - 'prev_status': the status of the previous row
          - 'status_flip': boolean indicating if status changed from prev row

    Notes
    -----
    - If multiple consecutive rows have the same status, only the first row in the 
      new block is marked as flipped.
    - If there's no previous row (i.e. first row), 'status_flip' is False by definition.
    """
    out_df = df.copy()
    if sort_by_date:
        out_df.sort_values(sort_by_date, inplace=True)

    out_df["prev_status"] = out_df[status_col].shift(1)
    out_df["status_flip"] = (
        out_df[status_col] != out_df["prev_status"]
    ) & out_df["prev_status"].notna()

    return out_df


def track_status_durations(
    df: pd.DataFrame, 
    status_col: str = "status",
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate consecutive runs of the same status. If date_col is provided, 
    we measure actual time durations in days (or whatever the date_col unit is).
    Otherwise, we measure by row counts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'status_col' and optionally 'date_col'.
    status_col : str, default='status'
        Column name for the metric status.
    date_col : str or None, default=None
        If provided, we assume it's a datetime or numeric column that we can
        use to measure durations. 
        - The data must be sorted by date_col in ascending order.

    Returns
    -------
    pd.DataFrame
        A DataFrame of consecutive status runs with columns:
        ['status', 'start_index', 'end_index', 'run_length', 'start_date', 'end_date', 'duration_days'] 
        (duration_days if date_col is provided)

    Notes
    -----
    - If no date_col is given, run_length is a count of consecutive rows.
    - If date_col is given, you must ensure df is sorted ascending by that date_col.
      'duration_days' is computed as (end_date - start_date).days + 1 or similar logic.
    """
    out = []
    df = df.copy()
    if date_col:
        df.sort_values(date_col, inplace=True)

    current_status = None
    start_idx = 0
    start_date_val = None

    for i, row in df.iterrows():
        s_val = row[status_col]
        if s_val != current_status:
            # if we had a previous run, close it out
            if current_status is not None:
                end_idx = i - 1
                end_date_val = df.loc[end_idx, date_col] if date_col else None

                # compute run_length
                run_length = end_idx - start_idx + 1
                # compute duration_days
                if date_col:
                    start_dt = pd.to_datetime(start_date_val)
                    end_dt = pd.to_datetime(end_date_val)
                    duration_days = (end_dt - start_dt).days + 1
                else:
                    duration_days = None

                out.append({
                    "status": current_status,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "run_length": run_length,
                    "start_date": start_date_val,
                    "end_date": end_date_val,
                    "duration_days": duration_days
                })

            # start a new run
            current_status = s_val
            start_idx = i
            start_date_val = row[date_col] if date_col else None

    # close out the final run
    if current_status is not None:
        end_idx = df.index[-1]
        end_date_val = df.iloc[-1][date_col] if date_col else None
        run_length = end_idx - start_idx + 1
        if date_col:
            start_dt = pd.to_datetime(start_date_val)
            end_dt = pd.to_datetime(end_date_val)
            duration_days = (end_dt - start_dt).days + 1
        else:
            duration_days = None

        out.append({
            "status": current_status,
            "start_index": start_idx,
            "end_index": end_idx,
            "run_length": run_length,
            "start_date": start_date_val,
            "end_date": end_date_val,
            "duration_days": duration_days
        })

    return pd.DataFrame(out)

def monitor_threshold_proximity(
    val: float, 
    target: float, 
    margin: float = 0.05,
    allow_negative_target: bool = False
) -> bool:
    """
    Returns True if 'val' is within +/- margin fraction of 'target'.

    Parameters
    ----------
    val : float
        The actual observed value.
    target : float
        The target value.
    margin : float, default=0.05
        E.g. 0.05 => within 5% of target is "close".
    allow_negative_target : bool, default=False
        If True, we compute ratio vs. abs(target). If false and target <= 0,
        we simply return False or handle scenario differently.

    Returns
    -------
    bool
        True if within margin * target (or abs(target)), otherwise False.

    Notes
    -----
    - If target <= 0 and allow_negative_target=False, returns False by default.
    - If margin=0.05, we interpret "within +/-5% of target" as:
        abs(val - target) / abs(target) <= 0.05
    """
    if target == 0 and not allow_negative_target:
        return False
    if target <= 0 and not allow_negative_target:
        return False

    if target == 0 and allow_negative_target:
        # If both val and target are 0 => difference=0 => proximity => True
        return (val == 0)

    diff_ratio = abs(val - target) / abs(target)
    return diff_ratio <= margin

def calculate_required_growth(
    current_value: float, 
    target_value: float, 
    periods_left: int,
    allow_negative: bool = False
) -> Optional[float]:
    """
    Determine needed compound growth rate to reach target_value from current_value 
    over periods_left. 

    This returns a decimal for the per-period growth. For example, 0.02 => 2% per period.

    Parameters
    ----------
    current_value : float
        The current metric value.
    target_value : float
        The desired metric value at the end of 'periods_left'.
    periods_left : int
        Number of discrete periods (days, weeks, months) in which to grow from current_value to target_value.
    allow_negative : bool, default=False
        If True, we allow negative or zero values for current_value or target_value and attempt a 
        ratio-based approach if possible (like negative->less negative?). If the signs of current_value 
        and target_value differ, we return None (domain-specific complexity).

    Returns
    -------
    float or None
        The compound growth rate per period. e.g. 0.02 => 2% growth per period. 
        None if it's not feasible or if periods_left <= 0 or domain constraints fail.

    Notes
    -----
    - If current_value <= 0 or target_value <= 0 and allow_negative=False, returns None.
    - If current_value > 0 and target_value > 0, rate = (target_value / current_value)^(1/periods_left) - 1.
    - If both are negative and allow_negative=True, we might interpret 
        rate = (abs(target_value)/abs(current_value))^(1/periods_left) - 1,
        but the meaning of "growth" from negative to negative can be domain-specific.
    - If sign of current_value != sign of target_value, we return None to avoid crossing zero with 
      a single rate. That's typically a special scenario requiring more advanced logic.
    """
    if periods_left <= 0:
        return None

    # Basic domain checks
    if not allow_negative:
        if (current_value <= 0) or (target_value <= 0):
            return None
    else:
        # If they differ in sign or one is zero, handle carefully
        if current_value < 0 and target_value < 0:
            # We'll define "growth" in absolute terms for negative => negative
            current_value = abs(current_value)
            target_value = abs(target_value)
        elif current_value == 0 and target_value == 0:
            # 0 => 0 in periods_left => rate=0
            return 0.0
        elif current_value == 0 and target_value < 0:
            # 0 => negative? Hard to do with a single growth rate => return None
            return None
        elif current_value < 0 and target_value == 0:
            # negative => 0 => also domain-specific => None
            return None
        else:
            # If the sign is different, we return None
            if (current_value < 0 and target_value > 0) or (current_value > 0 and target_value < 0):
                return None

    # Now we assume current_value>0, target_value>0 after the above logic,
    # or we've turned them both positive if originally negative
    if current_value <= 0:
        return None  # just in case
    ratio = target_value / current_value
    # if ratio=0 => rate is -1 => domain meltdown => None
    if ratio <= 0:
        return None

    rate = ratio ** (1.0 / periods_left) - 1.0
    return rate