# =============================================================================
# Performance
#
#   - calculate_metric_gva => optional allow_negative_target (#11)
#   - calculate_historical_gva => vectorized negative-target handling (#12)
#   - classify_metric_status => threshold ratio and negative-target logic (#13)
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import numpy as np
import pandas as pd

def _safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator/denominator

def calculate_metric_gva(actual_value, target_value, allow_negative_target=False):
    """
    Purpose: Compute GvA difference, with optional negative-target logic.

    (Suggested Update #11)
    """
    abs_diff = actual_value - target_value

    # If target <=0 and not allowed => pct_diff=None
    if (target_value <= 0) and not allow_negative_target:
        return {"abs_diff": abs_diff, "pct_diff": None}

    if target_value == 0:
        # define infinite or 0
        if actual_value == 0:
            pct_diff = 0.0
        elif actual_value>0:
            pct_diff = float('inf')
        else:
            pct_diff = float('-inf')
    else:
        pct_diff = (abs_diff/abs(target_value))*100.0

    return {"abs_diff": abs_diff, "pct_diff": pct_diff}


def calculate_historical_gva(df_actual, df_target, date_col='date', value_col='value',
                             allow_negative_target=False):
    """
    Purpose: Compute historical GvA with negative target handling.
    (Suggested Update #12)
    """
    merged = pd.merge(
        df_actual[[date_col, value_col]],
        df_target[[date_col, value_col]],
        on=date_col, how='inner', suffixes=('_actual','_target')
    )
    def _compute_gva(row):
        a = row[f"{value_col}_actual"]
        t = row[f"{value_col}_target"]
        # same logic as above
        out = calculate_metric_gva(a, t, allow_negative_target=allow_negative_target)
        return pd.Series([out['abs_diff'], out['pct_diff']], index=['abs_diff','pct_diff'])

    merged[['abs_diff','pct_diff']] = merged.apply(_compute_gva, axis=1)
    return merged.rename(columns={
        f"{value_col}_actual": "actual",
        f"{value_col}_target": "target"
    })


def classify_metric_status(actual_value, target_value,
                           threshold_ratio=0.05, allow_negative_target=False,
                           status_if_no_target="no_target"):
    """
    Purpose: Classify a metric as 'on_track' or 'off_track' given a threshold ratio,
             with optional negative-target handling. (Suggested Update #13)

    If target <=0 and not allow_negative_target => return status_if_no_target
    If target>0 => on_track if actual_value >= target*(1-threshold_ratio)
    If target<0 => on_track if actual_value <= target*(1+threshold_ratio)  # negative sense
    """
    if target_value is None or np.isnan(target_value):
        return status_if_no_target

    if (target_value <= 0) and (not allow_negative_target):
        return status_if_no_target

    if target_value > 0:
        cutoff = target_value*(1.0 - threshold_ratio)
        return "on_track" if actual_value >= cutoff else "off_track"
    else:
        # negative target => 'on track' means not less than the negative threshold
        # e.g. if target is -100, threshold_ratio=0.05 => cutoff=-100*(1+0.05)=-105
        cutoff = target_value*(1.0 + threshold_ratio)
        return "on_track" if actual_value<=cutoff else "off_track"


def detect_status_changes(df, status_col='status'):
    df = df.copy()
    df['prev_status'] = df[status_col].shift(1)
    df['status_flip'] = (df[status_col]!=df['prev_status']) & df['prev_status'].notna()
    return df


def track_status_durations(df, date_col='date', status_col='status'):
    """
    Return [status, start_date, end_date, run_length].
    """
    df = df.sort_values(by=date_col).reset_index(drop=True)
    runs = []
    current_status = None
    current_start = None

    for i, row in df.iterrows():
        st = row[status_col]
        dt = row[date_col]
        if st != current_status:
            if current_status is not None:
                runs.append({
                    "status": current_status,
                    "start_date": current_start,
                    "end_date": prev_date,
                    "run_length": (prev_date - current_start).days+1
                })
            current_status=st
            current_start=dt
        prev_date=dt
    # close last run
    if current_status is not None:
        runs.append({
            "status": current_status,
            "start_date": current_start,
            "end_date": prev_date,
            "run_length": (prev_date - current_start).days+1
        })
    return pd.DataFrame(runs)


def monitor_threshold_proximity(df, margin_percent=5.0, actual_col='actual', target_col='target',
                                allow_negative_target=False):
    """
    For each row => check if actual is within +/- margin% of target.
    Returns df with 'near_flip' bool col.
    """
    df = df.copy()
    margin = margin_percent/100.0

    def _near(row):
        t = row[target_col]
        a = row[actual_col]
        if (t <=0) and not allow_negative_target:
            return False
        if t==0:
            return (a==0)
        return abs(a-t)/abs(t) <= margin

    df['near_flip'] = df.apply(_near, axis=1)
    return df


def calculate_required_growth(current_value, target_value, periods_left, allow_negative=False):
    """
    Returns compound growth => (target/current)^(1/periods_left)-1, or None if invalid.
    Updated with optional allow_negative. (Similar logic from your code).
    """
    if periods_left<=0:
        return None
    if not allow_negative:
        if current_value<=0 or target_value<=0:
            return None
    else:
        if current_value<0 and target_value<0:
            current_value=abs(current_value)
            target_value=abs(target_value)
        elif current_value==0 or target_value==0:
            return None

    ratio = target_value/current_value
    if ratio<=0:
        return None
    return ratio**(1.0/periods_left)-1.0
