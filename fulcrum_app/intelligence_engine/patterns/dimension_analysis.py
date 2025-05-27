# dimensional_analysis.py
"""
Pattern: DimensionAnalysis
Version: 1.0

Purpose:
  Performs slice-level analysis for a given metric and dimension at a specified
  time grain (day, week, or month). It compares the current period to the
  immediately prior period, calculates share of total metric volume, performance
  deltas, ranks slices, identifies top/bottom slices, and more.

Inputs:
  ledger_df (pd.DataFrame): Contains metric data. Required columns:
    - 'metric_id' (str): Metric identifier.
    - 'time_grain' (str): Granularity ('day', 'week', 'month').
    - 'date' (datetime-like): Date of the metric value.
    - 'dimension' (str): Dimension name being analyzed (e.g., "region").
    - 'slice_value' (str): Specific slice within the dimension (e.g., "North America").
    - 'metric_value' (float): Numeric value of the metric for the slice.
  metric_id (str): The metric to analyze.
  dimension_name (str): The specific dimension column in ledger_df to analyze.
  grain (str): Time grain for analysis ('day', 'week', 'month').
  analysis_date (datetime or str): Focal date for defining the "current" period.

Outputs (JSON-serializable dict):
  A dictionary containing various analytical breakdowns as specified in the
  initial problem description, including "slices", "topSlicesByPerformance",
  "largestSlice", "newStrongestSlice", etc.

Relies on dimension analysis primitives from:
  intelligence_engine.primitives.dimensional_analysis
"""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, Union, List, Tuple

# Assuming primitives are in the specified path and correctly imported
from intelligence_engine.primitives.dimensional_analysis import (
    compute_slice_shares,
    rank_metric_slices,
    calculate_slice_metrics
    # analyze_composition_changes, # Might be too high-level for direct use here
    # compare_dimension_slices_over_time, # Not suitable for period comparison as designed
)


def _get_period_range_for_grain(analysis_date: Union[str, pd.Timestamp], grain: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Converts an analysis_date and grain into a (start_date, end_date) tuple.
    'day': analysis_date itself.
    'week': Monday to Sunday containing analysis_date.
    'month': Calendar month containing analysis_date.
    """
    dt = pd.to_datetime(analysis_date)
    if grain == 'day':
        start = dt.normalize()
        end = start
    elif grain == 'week':
        start = (dt - pd.Timedelta(days=dt.dayofweek)).normalize() # Monday is 0
        end = (start + pd.Timedelta(days=6)).normalize()
    elif grain == 'month':
        start = dt.replace(day=1).normalize()
        end = (start + pd.offsets.MonthEnd(0)).normalize()
    else:
        raise ValueError(f"Unsupported grain '{grain}'. Use 'day', 'week', or 'month'.")
    return start, end


def _get_prior_period_range(current_start_date: pd.Timestamp, current_end_date: pd.Timestamp, grain: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculates the start and end dates of the period immediately prior to the current period.
    """
    if grain == 'day':
        prior_start = current_start_date - pd.Timedelta(days=1)
        prior_end = current_end_date - pd.Timedelta(days=1)
    elif grain == 'week':
        prior_start = current_start_date - pd.Timedelta(weeks=1)
        prior_end = current_end_date - pd.Timedelta(weeks=1)
    elif grain == 'month':
        prior_start = (current_start_date - pd.offsets.MonthBegin(1)).normalize()
        prior_end = (prior_start + pd.offsets.MonthEnd(0)).normalize()
    else:
        raise ValueError(f"Unsupported grain '{grain}' for prior-period calculation.")
    return prior_start, prior_end


def _difference_from_average(df: pd.DataFrame, slice_identity_col: str, current_val_col: str = "val_t1") -> pd.DataFrame:
    """
    Adds 'avgOtherSlicesValue', 'absoluteDiffFromAvg', 'absoluteDiffPercentFromAvg' columns.
    Compares each slice's current_val_col to the mean of all other slices.
    """
    if df.empty or current_val_col not in df.columns:
        df["avgOtherSlicesValue"] = np.nan
        df["absoluteDiffFromAvg"] = np.nan
        df["absoluteDiffPercentFromAvg"] = np.nan
        return df

    df_out = df.copy()
    total_sum_all_slices = df_out[current_val_col].sum()
    num_total_slices = len(df_out)

    avg_other_slices_values = []
    abs_diff_from_avg_values = []
    pct_diff_from_avg_values = []

    for index, row in df_out.iterrows():
        current_slice_value = row[current_val_col]
        if num_total_slices > 1:
            sum_of_other_slices = total_sum_all_slices - current_slice_value
            avg_of_other_slices = sum_of_other_slices / (num_total_slices - 1)
            
            abs_diff = current_slice_value - avg_of_other_slices
            pct_diff = (abs_diff / abs(avg_of_other_slices) * 100.0) if avg_of_other_slices != 0 else np.nan
            
            avg_other_slices_values.append(avg_of_other_slices)
            abs_diff_from_avg_values.append(abs_diff)
            pct_diff_from_avg_values.append(pct_diff)
        else: # Only one slice
            avg_other_slices_values.append(np.nan)
            abs_diff_from_avg_values.append(np.nan)
            pct_diff_from_avg_values.append(np.nan)
            
    df_out["avgOtherSlicesValue"] = avg_other_slices_values
    df_out["absoluteDiffFromAvg"] = abs_diff_from_avg_values
    df_out["absoluteDiffPercentFromAvg"] = pct_diff_from_avg_values
    return df_out


def _build_slices_list(merged_df: pd.DataFrame, slice_identity_col: str) -> list:
    """
    Converts rows in the processed merged_df into the final 'slices' list format.
    """
    slices_list = []
    for _, row in merged_df.iterrows():
        slices_list.append({
            "sliceValue": row[slice_identity_col],
            "currentValue": row.get("val_t1", np.nan),
            "priorValue": row.get("val_t0", np.nan),
            "absoluteChange": row.get("abs_diff", np.nan),
            "relativeChangePercent": row.get("pct_diff", np.nan),
            "currentShareOfVolumePercent": row.get("share_pct_t1", np.nan),
            "priorShareOfVolumePercent": row.get("share_pct_t0", np.nan),
            "shareOfVolumeChangePercent": row.get("share_diff", np.nan),
            # "absoluteMarginalImpact": row.get("abs_diff", np.nan), # This is same as absoluteChange
            "avgOtherSlicesValue": row.get("avgOtherSlicesValue", np.nan),
            "absoluteDiffFromAvg": row.get("absoluteDiffFromAvg", np.nan),
            "absoluteDiffPercentFromAvg": row.get("absoluteDiffPercentFromAvg", np.nan),
            "consecutiveAboveAvgStreak": row.get("streak_above_avg", None), # Placeholder, add logic if needed
            "rankByPerformance": int(row["rank_performance"]) if pd.notna(row.get("rank_performance")) else None,
            "rankByShare": int(row["rank_share"]) if pd.notna(row.get("rank_share")) else None,
        })
    return slices_list


def _compute_top_bottom_slices(merged_df: pd.DataFrame, slice_identity_col:str, performance_val_col: str, top_n: int = 4) -> Tuple[list, list]:
    """
    Identifies top/bottom slices by `performance_val_col` ('val_t1').
    """
    if merged_df.empty:
        return [], []

    # Use the rank_metric_slices primitive
    top_slices_df = rank_metric_slices(
        agg_df=merged_df,
        val_col=performance_val_col, # This is 'val_t1'
        top_n=top_n,
        ascending=False
    )
    # The primitive returns the top_n, so we re-use its output directly.
    # It expects 'aggregated_value' as val_col default, so we must pass `performance_val_col`
    
    bottom_slices_df = rank_metric_slices(
        agg_df=merged_df,
        val_col=performance_val_col,
        top_n=top_n,
        ascending=True
    )

    def format_slice_summary(row_df, rank_val_col, perf_rank_col):
        return {
            "sliceValue": row_df[slice_identity_col],
            "metricValue": row_df[rank_val_col], # This is 'val_t1'
            "avgOtherSlicesValue": row_df.get("avgOtherSlicesValue", np.nan),
            "absoluteDiffFromAvg": row_df.get("absoluteDiffFromAvg", np.nan),
            "absoluteDiffPercentFromAvg": row_df.get("absoluteDiffPercentFromAvg", np.nan),
            "performanceRank": int(row_df[perf_rank_col]) if pd.notna(row_df.get(perf_rank_col)) else None
        }

    top_list = [format_slice_summary(row, performance_val_col, "rank_performance") for _, row in top_slices_df.iterrows()]
    bottom_list = [format_slice_summary(row, performance_val_col, "rank_performance") for _, row in bottom_slices_df.iterrows()]
    
    return top_list, bottom_list


def _largest_smallest_by_share(merged_df: pd.DataFrame, slice_identity_col: str) -> Tuple[dict, dict]:
    """
    Returns the largest and smallest slice by current share, and info about prior largest/smallest.
    """
    if merged_df.empty or "share_pct_t1" not in merged_df.columns or "share_pct_t0" not in merged_df.columns:
        return {}, {}

    # Current period sorting
    df_sorted_current_share_desc = merged_df.sort_values("share_pct_t1", ascending=False, na_position='last').reset_index(drop=True)
    df_sorted_current_share_asc = merged_df.sort_values("share_pct_t1", ascending=True, na_position='last').reset_index(drop=True)

    # Prior period sorting
    df_sorted_prior_share_desc = merged_df.sort_values("share_pct_t0", ascending=False, na_position='last').reset_index(drop=True)
    df_sorted_prior_share_asc = merged_df.sort_values("share_pct_t0", ascending=True, na_position='last').reset_index(drop=True)

    largest_dict = {}
    if not df_sorted_current_share_desc.empty:
        current_largest_row = df_sorted_current_share_desc.iloc[0]
        largest_dict = {
            "sliceValue": current_largest_row[slice_identity_col],
            "currentShareOfVolumePercent": current_largest_row["share_pct_t1"],
        }
        if not df_sorted_prior_share_desc.empty:
            prior_largest_row = df_sorted_prior_share_desc.iloc[0]
            largest_dict["previousLargestSliceValue"] = prior_largest_row[slice_identity_col]
            largest_dict["previousLargestSharePercent"] = prior_largest_row["share_pct_t0"]

    smallest_dict = {}
    if not df_sorted_current_share_asc.empty:
        current_smallest_row = df_sorted_current_share_asc.iloc[0]
        smallest_dict = {
            "sliceValue": current_smallest_row[slice_identity_col],
            "currentShareOfVolumePercent": current_smallest_row["share_pct_t1"],
        }
        if not df_sorted_prior_share_asc.empty:
            prior_smallest_row = df_sorted_prior_share_asc.iloc[0]
            smallest_dict["previousSmallestSliceValue"] = prior_smallest_row[slice_identity_col]
            smallest_dict["previousSmallestSharePercent"] = prior_smallest_row["share_pct_t0"]
            
    return largest_dict, smallest_dict


def _strongest_weakest(merged_df: pd.DataFrame, slice_identity_col: str) -> Tuple[dict, dict]:
    """
    Identifies if the strongest/weakest performing slice (by val_t1 vs val_t0) has changed.
    """
    if merged_df.empty:
        return {}, {}

    # Sort by current performance (val_t1)
    df_sorted_val_t1_desc = merged_df.sort_values("val_t1", ascending=False, na_position='last').reset_index(drop=True)
    df_sorted_val_t1_asc = merged_df.sort_values("val_t1", ascending=True, na_position='last').reset_index(drop=True)

    # Sort by prior performance (val_t0)
    df_sorted_val_t0_desc = merged_df.sort_values("val_t0", ascending=False, na_position='last').reset_index(drop=True)
    df_sorted_val_t0_asc = merged_df.sort_values("val_t0", ascending=True, na_position='last').reset_index(drop=True)

    new_strongest_dict = {}
    if not df_sorted_val_t1_desc.empty and not df_sorted_val_t0_desc.empty:
        current_strongest_slice_row = df_sorted_val_t1_desc.iloc[0]
        prior_strongest_slice_row = df_sorted_val_t0_desc.iloc[0]

        if current_strongest_slice_row[slice_identity_col] != prior_strongest_slice_row[slice_identity_col]:
            # Find the prior period data for the current strongest slice
            prior_data_for_current_strongest = merged_df[merged_df[slice_identity_col] == current_strongest_slice_row[slice_identity_col]]
            
            new_strongest_dict = {
                "sliceValue": current_strongest_slice_row[slice_identity_col],
                "currentValue": current_strongest_slice_row["val_t1"],
                "previousStrongestSliceValue": prior_strongest_slice_row[slice_identity_col],
                "previousStrongestSliceValuePriorPeriodValue": prior_strongest_slice_row["val_t0"], # Value of the *previously* strongest slice in T0
                "currentSlicePriorPeriodValue": prior_data_for_current_strongest["val_t0"].iloc[0] if not prior_data_for_current_strongest.empty else np.nan, # Value of the *current* strongest slice in T0
            }

    new_weakest_dict = {}
    if not df_sorted_val_t1_asc.empty and not df_sorted_val_t0_asc.empty:
        current_weakest_slice_row = df_sorted_val_t1_asc.iloc[0]
        prior_weakest_slice_row = df_sorted_val_t0_asc.iloc[0]

        if current_weakest_slice_row[slice_identity_col] != prior_weakest_slice_row[slice_identity_col]:
            prior_data_for_current_weakest = merged_df[merged_df[slice_identity_col] == current_weakest_slice_row[slice_identity_col]]

            new_weakest_dict = {
                "sliceValue": current_weakest_slice_row[slice_identity_col],
                "currentValue": current_weakest_slice_row["val_t1"],
                "previousWeakestSliceValue": prior_weakest_slice_row[slice_identity_col],
                "previousWeakestSliceValuePriorPeriodValue": prior_weakest_slice_row["val_t0"],
                "currentSlicePriorPeriodValue": prior_data_for_current_weakest["val_t0"].iloc[0] if not prior_data_for_current_weakest.empty else np.nan,
            }
            
    return new_strongest_dict, new_weakest_dict


def _highlight_comparison(merged_df: pd.DataFrame, slice_identity_col: str) -> list:
    """
    Compares the top 2 performing slices (by current value 'val_t1').
    Calculates current performance gap and change in gap from prior period.
    """
    if merged_df.empty or len(merged_df) < 2:
        return []

    df_sorted_by_val_t1 = merged_df.sort_values("val_t1", ascending=False, na_position='last').reset_index(drop=True)
    
    slice_a_row = df_sorted_by_val_t1.iloc[0]
    slice_b_row = df_sorted_by_val_t1.iloc[1]

    # Current performance gap: (A_t1 - B_t1) / |B_t1| * 100
    current_gap_pct = np.nan
    if slice_b_row["val_t1"] != 0:
        current_gap_pct = (slice_a_row["val_t1"] - slice_b_row["val_t1"]) / abs(slice_b_row["val_t1"]) * 100.0

    # Prior performance gap: (A_t0 - B_t0) / |B_t0| * 100
    prior_gap_pct = np.nan
    if slice_b_row["val_t0"] != 0: # Using B's prior value as denominator
        prior_gap_pct = (slice_a_row["val_t0"] - slice_b_row["val_t0"]) / abs(slice_b_row["val_t0"]) * 100.0
        
    gap_change_pct_points = np.nan
    if pd.notna(current_gap_pct) and pd.notna(prior_gap_pct):
        gap_change_pct_points = current_gap_pct - prior_gap_pct

    return [{
        "sliceA_value": slice_a_row[slice_identity_col],
        "sliceA_currentValue": slice_a_row["val_t1"],
        "sliceA_priorValue": slice_a_row["val_t0"],
        "sliceB_value": slice_b_row[slice_identity_col],
        "sliceB_currentValue": slice_b_row["val_t1"],
        "sliceB_priorValue": slice_b_row["val_t0"],
        "currentPerformanceGapPercent": current_gap_pct, # A vs B
        "priorPerformanceGapPercent": prior_gap_pct,
        "gapChangePercentPoints": gap_change_pct_points # Change in the gap percentage
    }]


def _compute_historical_slice_rankings(
    ledger_df_metric_dim_grain: pd.DataFrame, # Pre-filtered for metric, dimension, grain
    initial_analysis_date: pd.Timestamp,
    grain: str,
    slice_col_name_in_ledger: str, # e.g., 'slice_value'
    metric_col_name_in_ledger: str, # e.g., 'metric_value'
    num_periods: int = 8,
    top_n_slices_per_period: int = 5
) -> Dict[str, Any]:
    """
    Computes top N slice rankings for several historical periods.
    """
    period_rankings_list = []
    # Start with the period *before* the one defined by initial_analysis_date
    # because historical rankings are for periods leading up to the "current" one.
    initial_current_start, initial_current_end = _get_period_range_for_grain(initial_analysis_date, grain)
    current_period_iteration_start_date, _ = _get_prior_period_range(initial_current_start, initial_current_end, grain)


    for i in range(num_periods):
        # Determine the start and end of the historical period for this iteration
        hist_period_start, hist_period_end = _get_period_range_for_grain(current_period_iteration_start_date, grain)
        
        period_data_df = ledger_df_metric_dim_grain[
            (ledger_df_metric_dim_grain["date"] >= hist_period_start) &
            (ledger_df_metric_dim_grain["date"] <= hist_period_end)
        ]
        
        top_slices_for_this_period = []
        if not period_data_df.empty:
            # Aggregate slices for this historical period
            aggregated_slices_df = calculate_slice_metrics(
                df=period_data_df,
                slice_col=slice_col_name_in_ledger,
                value_col=metric_col_name_in_ledger,
                agg_func='sum' # Assuming sum aggregation for ranking
            )
            
            # Rank and get top N
            ranked_top_df = rank_metric_slices(
                agg_df=aggregated_slices_df,
                val_col='aggregated_value', # Default output from calculate_slice_metrics
                top_n=top_n_slices_per_period,
                ascending=False
            )
            
            for _, row in ranked_top_df.iterrows():
                top_slices_for_this_period.append({
                    "sliceValue": row[slice_col_name_in_ledger], # Use original slice col name
                    "metricValue": row['aggregated_value']
                })
            
        period_rankings_list.append({
            "periodNumber": i + 1, # 1-indexed, 1 is most recent historical period
            "periodStartDate": hist_period_start.strftime("%Y-%m-%d"),
            "periodEndDate": hist_period_end.strftime("%Y-%m-%d"),
            "topSlicesByPerformance": top_slices_for_this_period
        })
        
        # Move to the next earlier historical period
        next_hist_start, _ = _get_prior_period_range(hist_period_start, hist_period_end, grain)
        current_period_iteration_start_date = next_hist_start

    period_rankings_list.reverse() # Show oldest historical period first
    return {
        "periodsAnalyzed": len(period_rankings_list),
        "periodRankings": period_rankings_list
    }


def run_dimension_analysis(
    ledger_df: pd.DataFrame,
    metric_id: str,
    dimension_name: str, # This is the column name in ledger_df that holds dimension types like "Region"
                         # The *values* within this column are what we usually call dimensions,
                         # but here it's the name of the dimension category.
                         # The actual slices are in 'slice_value'.
    grain: str,
    analysis_date: Union[str, pd.Timestamp] # This is the focal date for the "current" period
) -> Dict[str, Any]:
    """
    Main function to perform slice-level analysis for a given metric & dimension.
    """
    # Standardize column names used internally for slices
    # The ledger_df is expected to have 'dimension' (e.g., "Region") and 'slice_value' (e.g., "NA")
    # For this pattern, dimension_name is the *value* we filter for in the 'dimension' column.
    # The actual slices are then taken from 'slice_value'.
    slice_identity_col_ledger = 'slice_value'
    metric_value_col_ledger = 'metric_value'
    time_grain_col_ledger = 'time_grain'
    date_col_ledger = 'date'
    metric_id_col_ledger = 'metric_id'
    dimension_col_ledger = 'dimension' # Column in ledger_df that identifies the dimension type

    # 1. Basic validation and initial filtering
    if ledger_df.empty:
        # Return empty shell if ledger_df is empty
        return {
            "schemaVersion": "1.0.0", "patternName": "DimensionAnalysis", "metricId": metric_id,
            "grain": grain, "analysisDate": str(analysis_date),
            "evaluationTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimensionName": dimension_name, "slices": [], "topSlicesByPerformance": [],
            "bottomSlicesByPerformance": [], "largestSlice": {}, "smallestSlice": {},
            "newStrongestSlice": {}, "newWeakestSlice": {}, "comparisonHighlights": [],
            "historicalSliceRankings": {"periodsAnalyzed": 0, "periodRankings": []}
        }

    df_filtered_metric_dim_grain = ledger_df[
        (ledger_df[metric_id_col_ledger] == metric_id) &
        (ledger_df[dimension_col_ledger] == dimension_name) & # Filter by the dimension *type*
        (ledger_df[time_grain_col_ledger] == grain)
    ].copy()

    if df_filtered_metric_dim_grain.empty:
        # Return empty shell if no data for this metric/dimension/grain
        # (same as above, slightly different reason for emptiness)
        return {
            "schemaVersion": "1.0.0", "patternName": "DimensionAnalysis", "metricId": metric_id,
            "grain": grain, "analysisDate": str(analysis_date),
            "evaluationTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimensionName": dimension_name, "slices": [], "topSlicesByPerformance": [],
            "bottomSlicesByPerformance": [], "largestSlice": {}, "smallestSlice": {},
            "newStrongestSlice": {}, "newWeakestSlice": {}, "comparisonHighlights": [],
            "historicalSliceRankings": {"periodsAnalyzed": 0, "periodRankings": []}
        }
    df_filtered_metric_dim_grain[date_col_ledger] = pd.to_datetime(df_filtered_metric_dim_grain[date_col_ledger])

    # 2. Determine current and prior period ranges
    current_start, current_end = _get_period_range_for_grain(analysis_date, grain)
    prior_start, prior_end = _get_prior_period_range(current_start, current_end, grain)

    # 3. Aggregate data for T1 (current period)
    df_t1_period_raw = df_filtered_metric_dim_grain[
        (df_filtered_metric_dim_grain[date_col_ledger] >= current_start) &
        (df_filtered_metric_dim_grain[date_col_ledger] <= current_end)
    ]
    agg_t1_df = df_t1_period_raw.groupby(slice_identity_col_ledger)[metric_value_col_ledger].sum().reset_index()
    agg_t1_df = agg_t1_df.rename(columns={metric_value_col_ledger: 'val_t1'})

    # 4. Aggregate data for T0 (prior period)
    df_t0_period_raw = df_filtered_metric_dim_grain[
        (df_filtered_metric_dim_grain[date_col_ledger] >= prior_start) &
        (df_filtered_metric_dim_grain[date_col_ledger] <= prior_end)
    ]
    agg_t0_df = df_t0_period_raw.groupby(slice_identity_col_ledger)[metric_value_col_ledger].sum().reset_index()
    agg_t0_df = agg_t0_df.rename(columns={metric_value_col_ledger: 'val_t0'})

    # 5. Merge aggregated T0 and T1 data
    merged_df = pd.merge(agg_t1_df, agg_t0_df, on=slice_identity_col_ledger, how='outer').fillna(0.0)
    merged_df['abs_diff'] = merged_df['val_t1'] - merged_df['val_t0']
    merged_df['pct_diff'] = ((merged_df['abs_diff'] / merged_df['val_t0'].abs().replace(0, np.nan)) * 100.0)
    
    if merged_df.empty: # Handle case where no common slices or no data in periods
        return {
            "schemaVersion": "1.0.0", "patternName": "DimensionAnalysis", "metricId": metric_id,
            "grain": grain, "analysisDate": str(analysis_date),
            "evaluationTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimensionName": dimension_name, "slices": [], "topSlicesByPerformance": [],
            "bottomSlicesByPerformance": [], "largestSlice": {}, "smallestSlice": {},
            "newStrongestSlice": {}, "newWeakestSlice": {}, "comparisonHighlights": [],
            "historicalSliceRankings": {"periodsAnalyzed": 0, "periodRankings": []}
        }

    # 6. Compute shares for T1 and T0
    # Use compute_slice_shares primitive, ensuring correct column names
    t1_shares_df = compute_slice_shares(agg_t1_df.rename(columns={'val_t1': 'aggregated_value'}),
                                        slice_col=slice_identity_col_ledger,
                                        val_col='aggregated_value', share_col_name='share_pct_t1')
    t0_shares_df = compute_slice_shares(agg_t0_df.rename(columns={'val_t0': 'aggregated_value'}),
                                        slice_col=slice_identity_col_ledger,
                                        val_col='aggregated_value', share_col_name='share_pct_t0')

    merged_df = pd.merge(merged_df, t1_shares_df[[slice_identity_col_ledger, 'share_pct_t1']], on=slice_identity_col_ledger, how='left')
    merged_df = pd.merge(merged_df, t0_shares_df[[slice_identity_col_ledger, 'share_pct_t0']], on=slice_identity_col_ledger, how='left')
    merged_df['share_diff'] = merged_df['share_pct_t1'].fillna(0.0) - merged_df['share_pct_t0'].fillna(0.0)

    # 7. Augment with difference from average (based on val_t1)
    merged_df = _difference_from_average(merged_df, slice_identity_col=slice_identity_col_ledger, current_val_col='val_t1')

    # 8. Augment with performance and share ranks for T1
    merged_df['rank_performance'] = merged_df['val_t1'].rank(method='dense', ascending=False).fillna(-1).astype(int)
    merged_df['rank_share'] = merged_df['share_pct_t1'].rank(method='dense', ascending=False).fillna(-1).astype(int)

    # 9. Generate output components using helper functions
    slices_list_output = _build_slices_list(merged_df, slice_identity_col_ledger)
    top_slices_output, bottom_slices_output = _compute_top_bottom_slices(merged_df, slice_identity_col_ledger, 'val_t1')
    largest_slice_output, smallest_slice_output = _largest_smallest_by_share(merged_df, slice_identity_col_ledger)
    new_strongest_output, new_weakest_output = _strongest_weakest(merged_df, slice_identity_col_ledger)
    comparison_highlights_output = _highlight_comparison(merged_df, slice_identity_col_ledger)
    
    # For historical rankings, pass the DataFrame filtered for the specific metric, dimension, and grain
    historical_rankings_output = _compute_historical_slice_rankings(
        ledger_df_metric_dim_grain=df_filtered_metric_dim_grain,
        initial_analysis_date=pd.to_datetime(analysis_date),
        grain=grain,
        slice_col_name_in_ledger=slice_identity_col_ledger,
        metric_col_name_in_ledger=metric_value_col_ledger
    )

    # 10. Assemble final dictionary
    output_result = {
        "schemaVersion": "1.0.0",
        "patternName": "DimensionAnalysis",
        "metricId": metric_id,
        "grain": grain,
        "analysisDate": pd.to_datetime(analysis_date).strftime("%Y-%m-%d"),
        "evaluationTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dimensionName": dimension_name, # The *type* of dimension analyzed

        "slices": slices_list_output,
        "topSlicesByPerformance": top_slices_output,
        "bottomSlicesByPerformance": bottom_slices_output,
        "largestSlice": largest_slice_output,
        "smallestSlice": smallest_slice_output,
        "newStrongestSlice": new_strongest_output,
        "newWeakestSlice": new_weakest_output,
        "comparisonHighlights": comparison_highlights_output,
        "historicalSliceRankings": historical_rankings_output
    }
    return output_result

if __name__ == '__main__':
    # Create Sample Ledger Data
    num_days_history = 90 # Approx 3 months
    base_date = datetime.datetime(2024, 3, 15) # Our analysis_date will be this

    data_rows = []
    for i in range(num_days_history):
        d = base_date - datetime.timedelta(days=i)
        # Metric: "revenue", Dimension: "region"
        for region_slice in ["North", "South", "East", "West", "Central"]:
            base_val = {"North": 100, "South": 150, "East": 80, "West": 120, "Central": 90}[region_slice]
            # Simulate some trend and noise
            val = base_val + (num_days_history - i) * 0.1 * {"North": 1, "South": 1.2, "East": 0.8, "West": 1.1, "Central": 0.9}[region_slice] + np.random.normal(0,5)
            data_rows.append(["revenue", "day", d, "region", region_slice, round(max(0, val),2) ])
        # Metric: "users", Dimension: "platform"
        for platform_slice in ["iOS", "Android", "Web"]:
            base_val_user = {"iOS": 50, "Android": 70, "Web": 100}[platform_slice]
            val_user = base_val_user + (num_days_history - i) * 0.05 * {"iOS": 1.1, "Android": 1, "Web": 0.9}[platform_slice] + np.random.normal(0,3)
            data_rows.append(["users", "day", d, "platform", platform_slice, round(max(0, val_user),2) ])

    ledger_df_sample = pd.DataFrame(data_rows, columns=[
        "metric_id", "time_grain", "date", "dimension", "slice_value", "metric_value"
    ])
    
    # Test for 'revenue' by 'region' for the latest full day (March 15, 2024)
    print("--- Testing DimensionAnalysis for 'revenue' by 'region' (day grain) ---")
    analysis_date_test = base_date 
    
    result_revenue_day = run_dimension_analysis(
        ledger_df=ledger_df_sample,
        metric_id="revenue",
        dimension_name="region", # This is the value in the 'dimension' column
        grain="day",
        analysis_date=analysis_date_test
    )
    import json
    print(json.dumps(result_revenue_day, indent=2, default=str)) # default=str for datetime/numpy conversion

    # Test for 'users' by 'platform' for the week of March 15, 2024
    # Week of March 15, 2024 (Friday): Monday March 11 to Sunday March 17
    print("\n--- Testing DimensionAnalysis for 'users' by 'platform' (week grain) ---")
    analysis_date_test_week = datetime.datetime(2024, 3, 15) # A date within the week
    result_users_week = run_dimension_analysis(
        ledger_df=ledger_df_sample,
        metric_id="users",
        dimension_name="platform",
        grain="week",
        analysis_date=analysis_date_test_week
    )
    print(json.dumps(result_users_week, indent=2, default=str))

    # Test for the month of February 2024
    print("\n--- Testing DimensionAnalysis for 'revenue' by 'region' (month grain, Feb 2024) ---")
    analysis_date_test_month = datetime.datetime(2024, 2, 10) # A date within Feb
    result_revenue_month = run_dimension_analysis(
        ledger_df=ledger_df_sample,
        metric_id="revenue",
        dimension_name="region",
        grain="month",
        analysis_date=analysis_date_test_month
    )
    print(json.dumps(result_revenue_month, indent=2, default=str))

    # Test with a metric that has no data
    print("\n--- Testing with no data for metric ---")
    result_empty = run_dimension_analysis(
        ledger_df=ledger_df_sample,
        metric_id="non_existent_metric",
        dimension_name="region",
        grain="day",
        analysis_date=analysis_date_test
    )
    print(json.dumps(result_empty, indent=2, default=str))