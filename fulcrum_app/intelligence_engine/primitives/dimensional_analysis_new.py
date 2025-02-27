# =============================================================================
# DimensionAnalysis
#
# UPDATED: "calculate_slice_metrics" now has a `top_n` param and lumps
#          remaining slices into 'Other'.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import pandas as pd
import numpy as np

def calculate_slice_metrics(
    df: pd.DataFrame, 
    slice_col: str, 
    value_col: str, 
    agg_func='sum',
    top_n: int = None,
    other_label: str = "Other",
    dropna_slices: bool = True
) -> pd.DataFrame:
    """
    Purpose: Aggregate metric values by dimension slice, with optional top_n consolidation.

    UPDATED: Now lumps slices beyond top_n into a single 'Other' slice if top_n is provided.

    Parameters
    ----------
    df : pd.DataFrame
    slice_col : str
    value_col : str
    agg_func : str, default='sum'
    top_n : int, optional
    other_label : str, default="Other"
    dropna_slices : bool, default True

    Returns
    -------
    pd.DataFrame
        [slice_col, 'aggregated_value']
    """
    dff = df.copy()
    if dropna_slices:
        dff = dff[~dff[slice_col].isna()]

    grouped = dff.groupby(slice_col)[value_col]
    if agg_func not in ['sum', 'mean', 'min', 'max', 'count']:
        raise ValueError(f"agg_func '{agg_func}' not supported")

    if agg_func == 'sum':
        result = grouped.sum().reset_index(name='aggregated_value')
    elif agg_func == 'mean':
        result = grouped.mean().reset_index(name='aggregated_value')
    elif agg_func == 'min':
        result = grouped.min().reset_index(name='aggregated_value')
    elif agg_func == 'max':
        result = grouped.max().reset_index(name='aggregated_value')
    elif agg_func == 'count':
        result = grouped.count().reset_index(name='aggregated_value')

    result.sort_values('aggregated_value', ascending=False, inplace=True, ignore_index=True)

    # (Suggested Update #8): Lumping into 'Other'
    if top_n is not None and len(result) > top_n:
        top_part = result.iloc[:top_n]
        other_part = result.iloc[top_n:]
        other_val = other_part['aggregated_value'].sum()
        other_row = pd.DataFrame({slice_col: [other_label], 'aggregated_value': [other_val]})
        result = pd.concat([top_part, other_row], ignore_index=True)

    return result


def compute_slice_shares(df, slice_col, value_col='aggregated_value'):
    total = df[value_col].sum()
    df = df.copy()
    df['share'] = np.where(total != 0, df[value_col]/total*100.0, 0.0)
    return df


def rank_metric_slices(df, slice_col, value_col='aggregated_value', top_n=5, ascending=False):
    dff = df.sort_values(value_col, ascending=ascending).copy()
    return dff.head(top_n)


def compare_dimension_slices_over_time(df_t0, df_t1, slice_col, value_col):
    agg_t0 = df_t0.groupby(slice_col)[value_col].sum().reset_index(name='valT0')
    agg_t1 = df_t1.groupby(slice_col)[value_col].sum().reset_index(name='valT1')
    merged = pd.merge(agg_t0, agg_t1, on=slice_col, how='outer').fillna(0)
    merged['abs_diff'] = merged['valT1'] - merged['valT0']

    def _safe_pct_diff(row):
        if row['valT0'] == 0:
            return np.nan
        return (row['abs_diff'] / row['valT0']) * 100.0

    merged['pct_diff'] = merged.apply(_safe_pct_diff, axis=1)
    merged.sort_values('abs_diff', ascending=False, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def analyze_composition_changes(df_t0, df_t1, slice_col, value_col):
    """
    Purpose: Show how slice shares changed from T0→T1, highlighting biggest shifts.

    Implementation Details:
    1. For T0, compute each slice's share; same for T1.
    2. share_diff = shareT1 - shareT0.
    3. Sort by share_diff descending.

    Parameters
    ----------
    df_t0 : pd.DataFrame
    df_t1 : pd.DataFrame
    slice_col : str
    value_col : str

    Returns
    -------
    pd.DataFrame
        [slice_col, shareT0, shareT1, share_diff].
    """
    agg_t0 = df_t0.groupby(slice_col)[value_col].sum().reset_index(name='valT0')
    agg_t1 = df_t1.groupby(slice_col)[value_col].sum().reset_index(name='valT1')

    total_t0 = agg_t0['valT0'].sum()
    total_t1 = agg_t1['valT1'].sum()

    merged = pd.merge(agg_t0, agg_t1, on=slice_col, how='outer').fillna(0)
    merged['shareT0'] = np.where(total_t0 != 0, (merged['valT0']/total_t0)*100, 0)
    merged['shareT1'] = np.where(total_t1 != 0, (merged['valT1']/total_t1)*100, 0)
    merged['share_diff'] = merged['shareT1'] - merged['shareT0']
    merged.sort_values(by='share_diff', ascending=False, inplace=True)
    return merged[[slice_col, 'shareT0', 'shareT1', 'share_diff']]


def calculate_concentration_index(df, slice_col, value_col, index_type='HHI'):
    """
    Purpose: Compute a standard measure of concentration (e.g., HHI) to assess concentration risk.

    Implementation Details:
    1. For each slice, compute share in [0..1].
    2. HHI = sum(share^2). Possibly also Gini if desired.
    3. Return numeric concentration index.
    4. Compare to threshold to label "high concentration" (not done here).

    Parameters
    ----------
    df : pd.DataFrame
    slice_col : str
    value_col : str
    index_type : str, default='HHI'
        Could be 'HHI' or 'Gini' for more advanced usage.

    Returns
    -------
    float
        Concentration index value (e.g., 0.0 < HHI <= 1.0 if shares are in fraction form).
    """
    total = df[value_col].sum()
    if total == 0:
        return 0.0

    df = df.copy()
    df['share'] = df[value_col] / total  # fraction in [0..1]

    if index_type == 'HHI':
        # Hirschman-Herfindahl Index = sum( (share_i)^2 )
        hhi = (df['share']**2).sum()
        return hhi
    else:
        # You could implement Gini or other measure
        # For demonstration, we only do HHI here
        return (df['share']**2).sum()
