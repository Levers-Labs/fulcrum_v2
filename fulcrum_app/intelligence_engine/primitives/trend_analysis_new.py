# =============================================================================
# TrendAnalysis
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - scipy.stats for linregress
# =============================================================================

import math
import numpy as np
import pandas as pd
from scipy.stats import linregress

def process_control_analysis(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    half_average_point: int = 9,
    consecutive_signal_threshold: int = 5,
    min_data_points: int = 10,
    moving_range_multiplier: float = 2.66,
    consecutive_run_length: int = 7,
    long_run_total_length: int = 12,
    long_run_min_length: int = 10,
    short_run_total_length: int = 4,
    short_run_min_length: int = 3,
) -> pd.DataFrame:
    """
    Your advanced SPC method from sample code.
    (Suggested Update #14)
    """
    dff = df.copy()
    if date_col not in dff.columns or value_col not in dff.columns:
        raise ValueError("DataFrame must contain date_col and value_col")

    dff[date_col] = pd.to_datetime(dff[date_col])
    dff.sort_values(date_col, inplace=True)
    dff.reset_index(drop=True, inplace=True)

    if len(dff)<min_data_points:
        dff["central_line"]=np.nan
        dff["ucl"]=np.nan
        dff["lcl"]=np.nan
        dff["slope"]=np.nan
        dff["slope_change"]=np.nan
        dff["trend_signal_detected"]=False
        return dff

    n_points=len(dff)
    central_line_array=[None]*n_points
    slope_array=[None]*n_points
    ucl_array=[None]*n_points
    lcl_array=[None]*n_points
    signal_array=[False]*n_points

    start_idx=0
    while start_idx<n_points:
        end_idx = min(start_idx+half_average_point*2, n_points)
        seg_length=end_idx-start_idx
        if seg_length<2:
            break
        segment_center,segment_slope = _compute_segment_center_line(dff, start_idx,end_idx,half_average_point,value_col)
        for i in range(seg_length):
            idx=start_idx+i
            central_line_array[idx]=segment_center[i]
            slope_array[idx]=segment_slope
        segment_values=dff[value_col].iloc[start_idx:end_idx].reset_index(drop=True)
        avgrange=_average_moving_range(segment_values)
        for i in range(seg_length):
            idx=start_idx+i
            cl_val=central_line_array[idx]
            if cl_val is not None and not math.isnan(cl_val):
                ucl_array[idx]= cl_val + avgrange*moving_range_multiplier
                lcl_array[idx]= cl_val - avgrange*moving_range_multiplier
            else:
                ucl_array[idx]=np.nan
                lcl_array[idx]=np.nan

        seg_signals = _detect_spc_signals(
            df_segment=dff.iloc[start_idx:end_idx],
            offset=start_idx,
            central_line_array=central_line_array,
            ucl_array=ucl_array,
            lcl_array=lcl_array,
            value_col=value_col,
            consecutive_run_length=consecutive_run_length,
            long_run_total_length=long_run_total_length,
            long_run_min_length=long_run_min_length,
            short_run_total_length=short_run_total_length,
            short_run_min_length=short_run_min_length
        )
        for sidx in seg_signals:
            signal_array[sidx]=True
        recalc_idx = _check_consecutive_signals(seg_signals, consecutive_signal_threshold)
        if recalc_idx is not None and recalc_idx<n_points:
            if recalc_idx>=n_points-1:
                break
            start_idx=recalc_idx
        else:
            start_idx=end_idx

    slope_change_array=[None]*n_points
    for i in range(1,n_points):
        s_now=slope_array[i]
        s_prev=slope_array[i-1]
        if s_now is not None and s_prev is not None:
            if abs(s_prev)<1e-9:
                slope_change_array[i]=None
            else:
                slope_change_array[i]=(s_now-s_prev)/abs(s_prev)*100.0
        else:
            slope_change_array[i]=None

    dff["central_line"]=central_line_array
    dff["ucl"]=ucl_array
    dff["lcl"]=lcl_array
    dff["slope"]=slope_array
    dff["slope_change"]=slope_change_array
    dff["trend_signal_detected"]=signal_array
    return dff

def _compute_segment_center_line(df, start_idx, end_idx, half_average_point, value_col):
    seg=df[value_col].iloc[start_idx:end_idx].reset_index(drop=True)
    n=len(seg)
    if n<2:
        return ([None]*n, 0.0)
    half_pt=min(half_average_point, n//2)
    first_avg=seg.iloc[:half_pt].mean()
    second_avg=seg.iloc[-half_pt:].mean()
    slope=(second_avg-first_avg)/float(half_pt) if half_pt>0 else 0.0
    center_line=[None]*n
    mid_idx=half_pt//2 if half_pt>0 else 0
    if mid_idx>=n:
        center_line=[seg.mean()]*n
        slope=0.0
        return (center_line,slope)
    center_line[mid_idx]=first_avg
    for i in range(mid_idx+1,n):
        center_line[i]=center_line[i-1]+slope
    for i in range(mid_idx-1, -1, -1):
        center_line[i]=center_line[i+1]-slope
    return (center_line, slope)

def _average_moving_range(values: pd.Series)->float:
    diffs=values.diff().abs().dropna()
    if len(diffs)==0:
        return 0.0
    return diffs.mean()

def _detect_spc_signals(
    df_segment,
    offset,
    central_line_array,
    ucl_array,
    lcl_array,
    value_col,
    consecutive_run_length,
    long_run_total_length,
    long_run_min_length,
    short_run_total_length,
    short_run_min_length
):
    n=len(df_segment)
    idx_start=offset
    local_df=df_segment.reset_index(drop=True).copy()
    local_df["central_line"]=[central_line_array[idx_start+i] for i in range(n)]
    local_df["ucl"]=[ucl_array[idx_start+i] for i in range(n)]
    local_df["lcl"]=[lcl_array[idx_start+i] for i in range(n)]

    rule1=(local_df[value_col]>local_df["ucl"])|(local_df[value_col]<local_df["lcl"])

    above_center=local_df[value_col]>local_df["central_line"]
    below_center=local_df[value_col]<local_df["central_line"]
    rule2_above= above_center.rolling(window=consecutive_run_length,min_periods=consecutive_run_length).sum()==consecutive_run_length
    rule2_below= below_center.rolling(window=consecutive_run_length,min_periods=consecutive_run_length).sum()==consecutive_run_length
    rule2=rule2_above|rule2_below

    rolling_up=above_center.rolling(window=long_run_total_length,min_periods=long_run_total_length).sum()
    rolling_down=below_center.rolling(window=long_run_total_length,min_periods=long_run_total_length).sum()
    rule3=(rolling_up>=long_run_min_length)|(rolling_down>=long_run_min_length)

    local_df["one_sigma_up"]= local_df["central_line"] + (local_df["ucl"]-local_df["central_line"])/3
    local_df["one_sigma_down"]= local_df["central_line"] - (local_df["central_line"]-local_df["lcl"])/3
    near_ucl= local_df[value_col]>local_df["one_sigma_up"]
    near_lcl= local_df[value_col]<local_df["one_sigma_down"]
    rule4_up= near_ucl.rolling(window=short_run_total_length,min_periods=short_run_total_length).sum()>=short_run_min_length
    rule4_down= near_lcl.rolling(window=short_run_total_length,min_periods=short_run_total_length).sum()>=short_run_min_length
    rule4=rule4_up|rule4_down

    combined= rule1|rule2|rule3|rule4
    local_signal_idx=combined[combined.fillna(False)].index
    global_signal_idx=[idx_start+int(i) for i in local_signal_idx]
    return global_signal_idx

def _check_consecutive_signals(signal_idxes, threshold):
    if not signal_idxes:
        return None
    s=sorted(signal_idxes)
    consecutive_count=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]+1:
            consecutive_count+=1
            if consecutive_count>=threshold:
                return s[i-threshold+1]
        else:
            consecutive_count=1
    return None

def detect_record_high(df: pd.DataFrame, value_col="value") -> bool:
    if df.empty:
        return False
    latest=df[value_col].iloc[-1]
    highest=df[value_col].max()
    return (latest==highest or np.isclose(latest, highest))

def detect_record_low(df: pd.DataFrame, value_col="value") -> bool:
    if df.empty:
        return False
    latest=df[value_col].iloc[-1]
    lowest=df[value_col].min()
    return (latest==lowest or np.isclose(latest, lowest))

def detect_performance_plateau(df: pd.DataFrame, value_col="value", tolerance=0.01, window=7) -> bool:
    if len(df)<window:
        return False
    sub=df[value_col].tail(window).dropna()
    if len(sub)<window:
        return False
    avg=sub.mean()
    if abs(avg)<1e-12:
        return False
    return ((sub.max()-sub.min())/abs(avg))<tolerance