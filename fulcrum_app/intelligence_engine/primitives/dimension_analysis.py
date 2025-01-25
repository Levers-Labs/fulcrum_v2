import pandas as pd
import numpy as np

def calculate_slice_metrics(df: pd.DataFrame, slice_col: str, value_col: str, agg="sum") -> pd.DataFrame:
    grouped = df.groupby(slice_col)[value_col]
    if agg == "sum":
        out = grouped.sum().reset_index()
    elif agg == "mean":
        out = grouped.mean().reset_index()
    else:
        raise ValueError(f"Unknown agg: {agg}")
    out.rename(columns={value_col: "aggregated_value"}, inplace=True)
    return out

def compute_slice_shares(agg_df: pd.DataFrame, slice_col: str, val_col: str="aggregated_value"):
    df = agg_df.copy()
    total = df[val_col].sum()
    if total == 0:
        df["share_pct"] = 0.0
    else:
        df["share_pct"] = df[val_col] / total * 100.0
    return df

def rank_metric_slices(agg_df: pd.DataFrame, val_col: str="aggregated_value", top_n: int=5, ascending: bool=False):
    """
    Return top N (or bottom N if ascending=True) slices by aggregated_value.
    """
    df = agg_df.copy()
    df.sort_values(val_col, ascending=ascending, inplace=True)
    return df.head(top_n)

def analyze_composition_changes(df_t0: pd.DataFrame, df_t1: pd.DataFrame, slice_col: str="segment", val_col: str="aggregated_value"):
    """
    Show how each slice's share changed from T0->T1.
    """
    # Expect each DF to have [slice_col, val_col, share_pct], for example
    merged = pd.merge(df_t0, df_t1, on=slice_col, suffixes=("_t0","_t1"), how="outer").fillna(0)
    merged["share_diff"] = merged[f"share_pct_t1"] - merged[f"share_pct_t0"]
    return merged

def detect_anomalies_in_slices(df: pd.DataFrame, slice_col: str, value_col: str):
    """
    For each slice, compute mean & std across time, then see if current value is out of range.
    This is a stub: real logic depends on having time-series by slice.
    """
    # Placeholder approach: treat entire DF as the 'history' of each slice
    # Then compare the latest vs historical mean. Very simplistic.
    pass

def compare_dimension_slices_over_time(df: pd.DataFrame, slice_col: str, date_col: str="date", value_col: str="value"):
    """
    Compare T0 vs T1 for each slice. A stub for demonstration.
    """
    pass

def calculate_concentration_index(df: pd.DataFrame, val_col: str="aggregated_value") -> float:
    """
    Compute Herfindahl-Hirschman Index (HHI) as sum of (share^2).
    share in [0..1].
    """
    total = df[val_col].sum()
    if total == 0:
        return 0
    shares = df[val_col] / total
    hhi = (shares**2).sum()
    return float(hhi)
