import pandas as pd
import numpy as np
from typing import Optional, Union

def calculate_slice_metrics(
    df: pd.DataFrame, 
    slice_col: str, 
    value_col: str, 
    agg: str = "sum",
    top_n: Optional[int] = None,
    other_label: str = "Other",
    dropna_slices: bool = True
) -> pd.DataFrame:
    """
    Group by slice_col and aggregate value_col using a specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least [slice_col, value_col].
    slice_col : str
        Column representing the dimension or category.
    value_col : str
        Numeric column to be aggregated (sum, mean, etc.).
    agg : str, default='sum'
        Aggregation method. One of ['sum', 'mean', 'count', 'median', 'min', 'max'] or any valid DataFrameGroupBy agg.
    top_n : int or None, default=None
        If provided, we keep only the top N slices by aggregated_value and group the rest into a single "Other" slice.
        This is useful when slice_col has high cardinality.
    other_label : str, default='Other'
        Label to use for the combined group if top_n is not None.
    dropna_slices : bool, default=True
        If True, rows where slice_col is NaN or None are dropped before grouping.
        If False, those rows become their own category (NaN).

    Returns
    -------
    pd.DataFrame
        Columns: [slice_col, aggregated_value], sorted descending by aggregated_value (if top_n is used).
        The aggregated_value column name is always 'aggregated_value' for consistency.

    Example
    -------
    Suppose you have user spend data with columns [user_country, revenue].
      df_agg = calculate_slice_metrics(df, 'user_country', 'revenue', agg='sum', top_n=5)
      => returns a DataFrame with the top 5 countries + an 'Other' row.
    """
    df = df.copy()
    if dropna_slices:
        df = df[~df[slice_col].isna()]

    # Perform the groupby + aggregation
    grouped = df.groupby(slice_col)[value_col]
    try:
        out = grouped.agg(agg).reset_index()
    except Exception as e:
        raise ValueError(f"Unknown or invalid agg '{agg}': {e}")

    out.rename(columns={value_col: "aggregated_value"}, inplace=True)

    # If top_n is specified, group the rest into "Other"
    if top_n is not None:
        # Sort descending by aggregated_value
        out.sort_values("aggregated_value", ascending=False, inplace=True)
        # Slices beyond top_n => "Other"
        if len(out) > top_n:
            top_df = out.iloc[:top_n].copy()
            other_df = out.iloc[top_n:]
            # Sum the aggregated values for 'Other'
            other_val = other_df["aggregated_value"].sum()
            # Create a single "Other" row
            new_row = pd.DataFrame({
                slice_col: [other_label],
                "aggregated_value": [other_val]
            })
            out = pd.concat([top_df, new_row], ignore_index=True)

    # For consistency, let's sort by aggregated_value descending at the end
    out.sort_values("aggregated_value", ascending=False, inplace=True, ignore_index=True)
    return out

def compute_slice_shares(
    agg_df: pd.DataFrame, 
    slice_col: str, 
    val_col: str = "aggregated_value",
    share_col_name: str = "share_pct"
) -> pd.DataFrame:
    """
    Compute each slice's percentage share of the total in agg_df.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Must contain at least [slice_col, val_col].
    slice_col : str
        Dimension column name (e.g., region).
    val_col : str, default='aggregated_value'
        The numeric column representing aggregated sums or means.
    share_col_name : str, default='share_pct'
        Name of the new column that will store the percentage share (0..100).

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same rows as agg_df plus one additional column (share_col_name).

    Notes
    -----
    - If the sum of val_col is 0, we set share to 0.0 to avoid division by zero.
    """
    df = agg_df.copy()
    total = df[val_col].sum()
    if total == 0:
        df[share_col_name] = 0.0
    else:
        df[share_col_name] = (df[val_col] / total) * 100.0

    return df

def rank_metric_slices(
    agg_df: pd.DataFrame, 
    val_col: str = "aggregated_value", 
    top_n: int = 5, 
    ascending: bool = False
) -> pd.DataFrame:
    """
    Return the top N (or bottom N if ascending=True) slices by the specified val_col.

    Parameters
    ----------
    agg_df : pd.DataFrame
        DataFrame with at least [val_col].
    val_col : str, default='aggregated_value'
        The numeric column to rank.
    top_n : int, default=5
        Number of slices to return.
    ascending : bool, default=False
        If False, we return the top slices. If True, we return the bottom slices.

    Returns
    -------
    pd.DataFrame
        The filtered and sorted subset of agg_df.

    Notes
    -----
    - If len(agg_df) <= top_n, we return agg_df as is, optionally sorted.
    """
    df = agg_df.copy()
    df.sort_values(val_col, ascending=ascending, inplace=True)
    return df.head(top_n)

def analyze_composition_changes(
    df_t0: pd.DataFrame, 
    df_t1: pd.DataFrame, 
    slice_col: str = "segment", 
    val_col: str = "aggregated_value"
) -> pd.DataFrame:
    """
    Show how each slice's absolute and percentage shares changed from T0 to T1.

    Parameters
    ----------
    df_t0 : pd.DataFrame
        The aggregated slice-level data at time T0 (e.g., [slice_col, val_col, share_pct]).
    df_t1 : pd.DataFrame
        The aggregated slice-level data at time T1.
    slice_col : str, default='segment'
        The dimension column.
    val_col : str, default='aggregated_value'
        The numeric aggregated column. If share_pct also exists, we'll use it.
    Returns
    -------
    pd.DataFrame
        Columns:
          [slice_col,
           val_col+'_t0', val_col+'_t1',
           'abs_diff',  (the difference in absolute values)
           'share_pct_t0', 'share_pct_t1', 
           'share_diff']
        For slices that don't exist in T0 or T1, we fill with 0.
        The DataFrame is sorted by largest absolute difference or share difference.

    Example
    -------
    If we have T0 data for slices (A=100, B=50, C=30) and T1 data for slices (A=120, B=60, D=10),
    we see slice C disappears, slice D appears, slice A & B changed, etc.
    """
    t0 = df_t0.copy()
    t1 = df_t1.copy()

    # If share_pct not in columns, we can compute it quickly
    if "share_pct" not in t0.columns:
        t0 = compute_slice_shares(t0, slice_col, val_col=val_col)
    if "share_pct" not in t1.columns:
        t1 = compute_slice_shares(t1, slice_col, val_col=val_col)

    merged = pd.merge(
        t0[[slice_col, val_col, "share_pct"]], 
        t1[[slice_col, val_col, "share_pct"]],
        on=slice_col,
        how="outer",
        suffixes=("_t0", "_t1")
    ).fillna(0)

    merged["abs_diff"] = merged[f"{val_col}_t1"] - merged[f"{val_col}_t0"]
    merged["share_diff"] = merged["share_pct_t1"] - merged["share_pct_t0"]

    # Sort by largest absolute difference (descending)
    merged.sort_values("abs_diff", ascending=False, inplace=True, ignore_index=True)

    return merged

def detect_anomalies_in_slices(
    df: pd.DataFrame, 
    slice_col: str, 
    value_col: str,
    date_col: Optional[str] = None,
    z_thresh: float = 3.0,
    min_points_per_slice: int = 5
) -> pd.DataFrame:
    """
    For each slice, we compute its mean & std (across time if date_col is relevant).
    Then flag the latest row(s) that deviate > z_thresh stdev from the slice's mean.

    Parameters
    ----------
    df : pd.DataFrame
        Should contain [slice_col, value_col], optionally date_col.
        If date_col is present, we assume multiple records over time for each slice.
    slice_col : str
        The dimension column (e.g. region, product line).
    value_col : str
        The numeric metric to evaluate.
    date_col : str or None, default=None
        If provided, we can consider the "latest" date. If not, we just consider the entire data.
    z_thresh : float, default=3.0
        If the slice's value is more than z_thresh * std away from the slice mean => anomaly.
    min_points_per_slice : int, default=5
        Only compute anomalies if a slice has at least 5 data points historically. Otherwise, skip.

    Returns
    -------
    pd.DataFrame
        The original df with an added boolean column 'slice_anomaly', plus 'slice_mean', 'slice_std'.
        If date_col is provided, we only label the row(s) with the max date in that slice as anomaly
        checks. Otherwise, we label the entire row if it meets the threshold.

    Example
    -------
    - If you have daily revenue per region, you can see if today's region revenue is an outlier
      vs. that region's historical distribution.
    """
    df = df.copy()
    if date_col:
        # sort so the "latest" is at the bottom
        df.sort_values(date_col, inplace=True)

    # We'll group by slice_col. For each group, compute mean & std overall.
    stats = df.groupby(slice_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    stats = stats.rename(columns={"mean": "slice_mean", "std": "slice_std", "count": "slice_count"})

    # merge stats back
    merged = pd.merge(df, stats, on=slice_col, how="left")

    # define a function to check if a row is anomaly
    def is_anomaly(row):
        if row["slice_count"] < min_points_per_slice:
            return False
        # if the slice_std=0 => no variability => can't detect outliers => skip or check if row[value_col]!=mean
        if pd.isna(row["slice_std"]) or row["slice_std"] == 0:
            return False

        diff = abs(row[value_col] - row["slice_mean"])
        return diff > z_thresh * row["slice_std"]

    # We'll focus only on the "latest" if date_col is given => i.e. the max date in that slice
    # Approach: group by slice_col, find the max date, and only check that row.
    if date_col:
        # find the max date for each slice
        slice_latest = merged.groupby(slice_col)[date_col].transform("max")
        # anomaly check only on rows where date_col == slice_latest
        merged["slice_anomaly"] = False
        mask = (merged[date_col] == slice_latest)
        merged.loc[mask, "slice_anomaly"] = merged[mask].apply(is_anomaly, axis=1)
    else:
        # no date => check all rows
        merged["slice_anomaly"] = merged.apply(is_anomaly, axis=1)

    return merged

def compare_dimension_slices_over_time(
    df: pd.DataFrame,
    slice_col: str,
    date_col: str = "date",
    value_col: str = "value",
    t0: Optional[str] = None,
    t1: Optional[str] = None,
    agg: str = "sum"
) -> pd.DataFrame:
    """
    Compare each slice's aggregated metric at two distinct time points (t0 vs t1).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [slice_col, date_col, value_col].
    slice_col : str
        Dimension column name (e.g., region).
    date_col : str, default='date'
        Date/time column name.
    value_col : str, default='value'
        Numeric metric to aggregate.
    t0 : str or None, default=None
        A string or timestamp representing the T0 date. 
        If None, we'll pick the min(date_col).
    t1 : str or None, default=None
        If None, pick the max(date_col).
    agg : str, default='sum'
        Aggregation function when grouping.

    Returns
    -------
    pd.DataFrame
        [slice_col, val_t0, val_t1, abs_diff, pct_diff].
        Missing slices in either T0 or T1 => 0.

    Example
    -------
    df_agg = compare_dimension_slices_over_time(df, 'region', 'date', 'sales',
                                                t0='2025-01-01', t1='2025-02-01')
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # default t0/t1 if not provided
    if t0 is None:
        t0 = df[date_col].min()
    else:
        t0 = pd.to_datetime(t0)
    if t1 is None:
        t1 = df[date_col].max()
    else:
        t1 = pd.to_datetime(t1)

    # Filter for T0 range and T1 range. 
    # If T0/T1 are single dates, we might do df[date_col]==t0 or a small window. 
    # If we want entire month, define your approach.
    df_t0 = df[df[date_col] == t0]
    df_t1 = df[df[date_col] == t1]
    # Or if you want "before t0" or "between t0 and t1," adjust as needed.

    # group each one
    g_t0 = df_t0.groupby(slice_col)[value_col].agg(agg).reset_index().rename(columns={value_col:"val_t0"})
    g_t1 = df_t1.groupby(slice_col)[value_col].agg(agg).reset_index().rename(columns={value_col:"val_t1"})

    merged = pd.merge(g_t0, g_t1, on=slice_col, how="outer").fillna(0)
    merged["abs_diff"] = merged["val_t1"] - merged["val_t0"]
    # avoid division by zero
    merged["pct_diff"] = merged.apply(
        lambda r: (r["abs_diff"] / abs(r["val_t0"]) * 100) if r["val_t0"] != 0 else None,
        axis=1
    )

    merged.sort_values("abs_diff", ascending=False, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged

def calculate_concentration_index(
    df: pd.DataFrame, 
    val_col: str = "aggregated_value", 
    method: str = "HHI"
) -> float:
    """
    Compute a concentration index for the distribution of val_col across slices.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a numeric column val_col, which is the aggregated value by slice.
    val_col : str, default='aggregated_value'
        The numeric column for which to calculate concentration.
    method : str, default='HHI'
        - 'HHI' => Herfindahl-Hirschman Index, sum(share^2).
        - 'gini' => Gini coefficient (0 = perfect equality, 1 = perfect inequality). 
          Implementation below is a standard formula.

    Returns
    -------
    float
        The concentration index in [0..1+] for HHI, [0..1] for Gini.

    Notes
    -----
    - For HHI, we interpret share = val / sum(val), then HHI = sum(share^2).
      If there's only one slice, HHI=1. If slices are evenly distributed across N slices, HHI=1/N.
    - For Gini, we implement a common formula that sorts by val and sums up cumulative distributions.
      Gini can be in [0..1]. Zero => perfect equality, 1 => total inequality.
    """
    df = df.copy()
    total = df[val_col].sum()
    if total <= 0:
        # no distribution => index=0 or something
        return 0.0

    if method.lower() == "hhi":
        shares = df[val_col] / total
        hhi = (shares ** 2).sum()
        return float(hhi)

    elif method.lower() == "gini":
        # standard approach: sort by val, compute partial sums
        sorted_vals = np.sort(df[val_col].values)
        cum = np.cumsum(sorted_vals)
        # G = (1 / (n*mean)) * sum( (n-i+1)*x_i )
        n = len(sorted_vals)
        rel_cum = cum / total
        # A common formula for Gini is:
        # G = 1 - 2 * sum_i( (n - i) * x_i ) / (n * sum(vals))
        # We'll do a more direct approach:
        # https://en.wikipedia.org/wiki/Gini_coefficient#Definition
        # index-based approach:
        i = np.arange(1, n+1)
        gini = (n + 1 - 2*(np.sum((n+1 - i)*sorted_vals))) / (n * total)
        # or we can do a simpler approach:
        # area under Lorenz curve etc.
        # We'll do a well-known scikit-like approach:
        # G = 1 - 2 * (area under the Lorenz curve).
        # Because the user might want a quick approach, let's do a direct formula:
        #   Lorenz = cum / total
        #   area_L = sum((Lorenz[i] + Lorenz[i-1]) / 2) * (1/n) etc.
        # For brevity, let's keep the formula consistent with the above code.
        
        # Alternatively, there's a simpler approach often used:
        # G = 1 - sum( (x_i / total) * (2*i - n - 1) ), i=1..n, sorted x_i ascending
        # We'll do that:
        # Source: https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python
        sorted_vals = sorted_vals / total
        i = np.arange(1, n+1)
        gini_val = 1 - 2 * np.sum(sorted_vals * (n+1 - i)) / (n)
        return float(gini_val)

    else:
        raise ValueError(f"Unknown method: {method}")