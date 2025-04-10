"""
DimensionAnalysis Pattern

This Pattern performs slice-level analysis for a given metric and dimension at a specified
time grain (day, week, or month). It compares the current period to the immediately prior
period, calculates share of total metric volume, performance deltas, ranks slices, identifies
top and bottom slices, and more.

Inputs:
    - ledger_df (pd.DataFrame): Must contain columns:
        metric_id, time_grain, date, dimension, slice_value, metric_value
    - metric_id (str): The metric to analyze (e.g., "revenue").
    - dimension_name (str): Which dimension to analyze (e.g., "region").
    - grain (str): One of "day", "week", or "month".
    - analysis_date (datetime or str): Focal date. For 'day', uses that day;
      for 'week', uses that calendar week (Monday-Sunday); for 'month', uses that calendar month.

Outputs a dictionary with keys:
  "schemaVersion"
  "patternName"
  "metricId"
  "grain"
  "analysisDate"
  "evaluationTime"
  "dimensionName"
  "slices"
  "topSlicesByPerformance"
  "bottomSlicesByPerformance"
  "largestSlice"
  "smallestSlice"
  "newStrongestSlice"
  "newWeakestSlice"
  "comparisonHighlights"
  "historicalSliceRankings"

Relies on dimension analysis primitives from:
    intelligence_engine.primitives.dimensional_analysis

Author: Fulcrum Intelligence
"""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, Union

# Example imports from your DimensionAnalysis primitives:
from intelligence_engine.primitives.dimensional_analysis import (
    compare_dimension_slices_over_time,
    compute_slice_shares,
    rank_metric_slices,
    calculate_slice_metrics  # if needed
    # ... other functions as needed
)


def _get_period_range_for_grain(analysis_date: Union[str, pd.Timestamp], grain: str) -> (pd.Timestamp, pd.Timestamp):
    """
    Convert an analysis_date plus a grain into (start_date, end_date).
    For grain='day', returns (analysis_date, analysis_date).
    For grain='week', uses Monday-Sunday. For grain='month', uses full calendar month.
    """
    dt = pd.to_datetime(analysis_date)
    if grain == 'day':
        start = dt.normalize()
        end = start
    elif grain == 'week':
        # Monday-based
        day_of_week = dt.isoweekday()  # Monday=1
        start = (dt - pd.Timedelta(days=(day_of_week - 1))).normalize()
        end = start + pd.Timedelta(days=6)
    elif grain == 'month':
        start = dt.replace(day=1).normalize()
        next_month_start = start + pd.offsets.MonthBegin(1)
        end = (next_month_start - pd.Timedelta(days=1)).normalize()
    else:
        raise ValueError(f"Unsupported grain '{grain}'.")
    return start, end


def _get_prior_period_range(start_date: pd.Timestamp, end_date: pd.Timestamp, grain: str) -> (pd.Timestamp, pd.Timestamp):
    """
    Given current period start/end, return the immediately prior period of equal length.
    For 'day', subtract 1 day; for 'week', subtract 7 days; for 'month', subtract 1 month.
    """
    if grain == 'day':
        prior_start = start_date - pd.Timedelta(days=1)
        prior_end = end_date - pd.Timedelta(days=1)
    elif grain == 'week':
        prior_start = start_date - pd.Timedelta(days=7)
        prior_end = end_date - pd.Timedelta(days=7)
    elif grain == 'month':
        prior_start = start_date - pd.offsets.MonthBegin(1)
        prior_end = prior_start + pd.offsets.MonthEnd(0)
    else:
        raise ValueError(f"Unsupported grain '{grain}' for prior-period calculation.")
    return prior_start, prior_end


def _difference_from_average(df: pd.DataFrame, current_col: str = "val_t1") -> pd.DataFrame:
    """
    Add columns: 'avgOtherSlicesValue', 'absoluteDiffFromAvg', 'absoluteDiffPercentFromAvg'.
    This computes the difference of each slice's current_col from the mean
    of all other slices in that same DataFrame.
    """
    if df.empty:
        df["avgOtherSlicesValue"] = np.nan
        df["absoluteDiffFromAvg"] = np.nan
        df["absoluteDiffPercentFromAvg"] = np.nan
        return df

    total = df[current_col].sum()
    n_slices = len(df)
    df["avgOtherSlicesValue"] = np.nan
    df["absoluteDiffFromAvg"] = np.nan
    df["absoluteDiffPercentFromAvg"] = np.nan

    for i in range(n_slices):
        row_val = df.iloc[i][current_col]
        if n_slices > 1:
            sum_others = total - row_val
            avg_others = sum_others / (n_slices - 1)
            abs_diff = row_val - avg_others
            pct_diff = (abs_diff / avg_others) * 100 if avg_others != 0 else np.nan
            df.at[df.index[i], "avgOtherSlicesValue"] = avg_others
            df.at[df.index[i], "absoluteDiffFromAvg"] = abs_diff
            df.at[df.index[i], "absoluteDiffPercentFromAvg"] = pct_diff
        else:
            df.at[df.index[i], "avgOtherSlicesValue"] = np.nan
            df.at[df.index[i], "absoluteDiffFromAvg"] = np.nan
            df.at[df.index[i], "absoluteDiffPercentFromAvg"] = np.nan
    return df


def _build_slices_list(merged_df: pd.DataFrame) -> list:
    """
    Convert rows in merged_df into the final 'slices' list format.
    """
    slices_list = []
    for _, row in merged_df.iterrows():
        slices_list.append({
            "sliceValue": row["slice_col"],
            "currentValue": row["val_t1"],
            "priorValue": row["val_t0"],
            "absoluteChange": row["abs_diff"],
            "relativeChangePercent": row["pct_diff"],
            "currentShareOfVolumePercent": row.get("share_pct_t1", None),
            "priorShareOfVolumePercent": row.get("share_pct_t0", None),
            "shareOfVolumeChangePercent": row.get("share_diff", None),
            "absoluteMarginalImpact": row.get("abs_diff", None),  # same as absoluteChange
            "relativeMarginalImpactPercent": None,  # can fill if needed
            "avgOtherSlicesValue": row.get("avgOtherSlicesValue", None),
            "absoluteDiffFromAvg": row.get("absoluteDiffFromAvg", None),
            "absoluteDiffPercentFromAvg": row.get("absoluteDiffPercentFromAvg", None),
            "consecutiveAboveAvgStreak": None,  # placeholder or your logic
            "rankByPerformance": row.get("rank_performance", None),
            "rankByShare": row.get("rank_share", None),
        })
    return slices_list


def _compute_top_bottom_slices(merged_df: pd.DataFrame, top_n: int = 3) -> (list, list):
    """
    Identify the top and bottom slices by current performance (val_t1).
    Returns (topSlicesByPerformance, bottomSlicesByPerformance).
    """
    df_sorted_desc = merged_df.sort_values("val_t1", ascending=False).reset_index(drop=True)
    df_sorted_asc = merged_df.sort_values("val_t1", ascending=True).reset_index(drop=True)

    top_rows = df_sorted_desc.head(top_n)
    bottom_rows = df_sorted_asc.head(top_n)

    def to_dict(row):
        return {
            "sliceValue": row["slice_col"],
            "metricValue": row["val_t1"],
            "avgOtherSlicesValue": row.get("avgOtherSlicesValue", None),
            "absoluteDiffFromAvg": row.get("absoluteDiffFromAvg", None),
            "absoluteDiffPercentFromAvg": row.get("absoluteDiffPercentFromAvg", None),
            "performanceRank": row.get("rank_performance", None)
        }

    top_list = [to_dict(r) for _, r in top_rows.iterrows()]
    # For bottom slices, invert the rank if you want negative. 
    bottom_list = []
    for _, r in bottom_rows.iterrows():
        tmp = to_dict(r)
        if tmp["performanceRank"] is not None:
            tmp["performanceRank"] = -1 * tmp["performanceRank"]
        bottom_list.append(tmp)

    return top_list, bottom_list


def _largest_smallest_by_share(merged_df: pd.DataFrame) -> (dict, dict):
    """
    Return the largest and smallest slice by current share. Also figure out
    who was largest/smallest by prior share. 
    """
    if "share_pct_t1" not in merged_df.columns or merged_df.empty:
        return {}, {}

    df_desc_current = merged_df.sort_values("share_pct_t1", ascending=False).reset_index(drop=True)
    df_desc_prior = merged_df.sort_values("share_pct_t0", ascending=False).reset_index(drop=True)
    df_asc_current = merged_df.sort_values("share_pct_t1", ascending=True).reset_index(drop=True)
    df_asc_prior = merged_df.sort_values("share_pct_t0", ascending=True).reset_index(drop=True)

    largest_dict = {}
    smallest_dict = {}

    if len(df_desc_current) > 0 and len(df_desc_prior) > 0:
        largest_dict = {
            "sliceValue": df_desc_current.iloc[0]["slice_col"],
            "currentShareOfVolumePercent": df_desc_current.iloc[0]["share_pct_t1"],
            "previousLargestSliceValue": df_desc_prior.iloc[0]["slice_col"],
            "previousLargestSharePercent": df_desc_prior.iloc[0]["share_pct_t0"]
        }

    if len(df_asc_current) > 0 and len(df_asc_prior) > 0:
        smallest_dict = {
            "sliceValue": df_asc_current.iloc[0]["slice_col"],
            "currentShareOfVolumePercent": df_asc_current.iloc[0]["share_pct_t1"],
            "previousSmallestSliceValue": df_asc_prior.iloc[0]["slice_col"],
            "previousSmallestSharePercent": df_asc_prior.iloc[0]["share_pct_t0"]
        }

    return largest_dict, smallest_dict


def _strongest_weakest(merged_df: pd.DataFrame) -> (dict, dict):
    """
    Identify the new strongest (highest val_t1) vs. prior strongest (highest val_t0).
    Same for the weakest. If they differ, fill in the dict.
    """
    if merged_df.empty:
        return {}, {}

    df_current_desc = merged_df.sort_values("val_t1", ascending=False).reset_index(drop=True)
    df_prior_desc = merged_df.sort_values("val_t0", ascending=False).reset_index(drop=True)
    df_current_asc = merged_df.sort_values("val_t1", ascending=True).reset_index(drop=True)
    df_prior_asc = merged_df.sort_values("val_t0", ascending=True).reset_index(drop=True)

    strongest_dict = {}
    weakest_dict = {}

    if len(df_current_desc) > 0 and len(df_prior_desc) > 0:
        curr_slice = df_current_desc.iloc[0]["slice_col"]
        prior_slice = df_prior_desc.iloc[0]["slice_col"]
        if curr_slice != prior_slice:
            strongest_dict = {
                "sliceValue": curr_slice,
                "previousStrongestSliceValue": prior_slice,
                "currentValue": df_current_desc.iloc[0]["val_t1"],
                "priorValue": df_current_desc.iloc[0]["val_t0"]
            }
            abs_delta = strongest_dict["currentValue"] - strongest_dict["priorValue"]
            strongest_dict["absoluteDelta"] = abs_delta
            strongest_dict["relativeDeltaPercent"] = (
                (abs_delta / strongest_dict["priorValue"]) * 100
                if strongest_dict["priorValue"] != 0 else np.nan
            )

    if len(df_current_asc) > 0 and len(df_prior_asc) > 0:
        curr_slice = df_current_asc.iloc[0]["slice_col"]
        prior_slice = df_prior_asc.iloc[0]["slice_col"]
        if curr_slice != prior_slice:
            weakest_dict = {
                "sliceValue": curr_slice,
                "previousWeakestSliceValue": prior_slice,
                "currentValue": df_current_asc.iloc[0]["val_t1"],
                "priorValue": df_current_asc.iloc[0]["val_t0"]
            }
            abs_delta = weakest_dict["currentValue"] - weakest_dict["priorValue"]
            weakest_dict["absoluteDelta"] = abs_delta
            weakest_dict["relativeDeltaPercent"] = (
                (abs_delta / weakest_dict["priorValue"]) * 100
                if weakest_dict["priorValue"] != 0 else np.nan
            )

    return strongest_dict, weakest_dict


def _highlight_comparison(merged_df: pd.DataFrame) -> list:
    """
    Simple example: Compare the top 2 slices by current_value. Compute gap vs. prior gap.
    """
    if len(merged_df) < 2:
        return []
    df_desc = merged_df.sort_values("val_t1", ascending=False).reset_index(drop=True)
    sliceA = df_desc.iloc[0]
    sliceB = df_desc.iloc[1]
    # current gap
    gap_now = None
    if sliceB["val_t1"] != 0:
        gap_now = (sliceA["val_t1"] - sliceB["val_t1"]) / sliceB["val_t1"] * 100
    # prior gap
    gap_prior = None
    if sliceB["val_t0"] != 0:
        gap_prior = (sliceA["val_t0"] - sliceB["val_t0"]) / sliceB["val_t0"] * 100
    gap_change = None
    if gap_now is not None and gap_prior is not None:
        gap_change = gap_now - gap_prior

    return [{
        "sliceA": sliceA["slice_col"],
        "currentValueA": sliceA["val_t1"],
        "priorValueA": sliceA["val_t0"],
        "sliceB": sliceB["slice_col"],
        "currentValueB": sliceB["val_t1"],
        "priorValueB": sliceB["val_t0"],
        "performanceGapPercent": gap_now,
        "gapChangePercent": gap_change
    }]


def _compute_historical_slice_rankings(ledger_df: pd.DataFrame,
                                       metric_id: str,
                                       dimension_name: str,
                                       num_periods: int = 8) -> Dict[str, Any]:
    """
    Example approach: For the last `num_periods` weeks, pick top 5 slices in each period.
    """
    dff = ledger_df.copy()
    dff["date"] = pd.to_datetime(dff["date"])
    dff = dff[
        (dff["metric_id"] == metric_id) &
        (dff["dimension"] == dimension_name)
    ]
    if dff.empty:
        return {"periodsAnalyzed": 0, "periodRankings": []}

    end_of_latest = dff["date"].max()
    period_rankings = []
    current_end = end_of_latest

    for _ in range(num_periods):
        period_start = (current_end - pd.Timedelta(days=6)).normalize()
        mask = (dff["date"] >= period_start) & (dff["date"] <= current_end)
        tmp = dff[mask].groupby("slice_value")["metric_value"].sum().reset_index()
        tmp.sort_values("metric_value", ascending=False, inplace=True)
        top5 = tmp.head(5)
        top5_list = []
        for _, row in top5.iterrows():
            top5_list.append({
                "sliceValue": row["slice_value"],
                "metricValue": row["metric_value"]
            })
        period_rankings.append({
            "startDate": str(period_start.date()),
            "endDate": str(current_end.date()),
            "top5SlicesByPerformance": top5_list
        })
        current_end = (period_start - pd.Timedelta(days=1))

    period_rankings.reverse()
    return {
        "periodsAnalyzed": num_periods,
        "periodRankings": period_rankings
    }


def run_dimension_analysis(
    ledger_df: pd.DataFrame,
    metric_id: str,
    dimension_name: str,
    grain: str,
    analysis_date: Union[str, pd.Timestamp]
) -> Dict[str, Any]:
    """
    Perform slice-level analysis for a given metric & dimension at the specified grain.
    Uses the dimension-analysis primitives to compare current vs. prior, compute shares,
    rank slices, etc.
    """
    # Basic validation
    if ledger_df.empty or metric_id not in ledger_df["metric_id"].unique():
        return {
            "schemaVersion": "1.0.0",
            "patternName": "DimensionAnalysis",
            "metricId": metric_id,
            "grain": grain,
            "analysisDate": str(analysis_date),
            "evaluationTime": str(datetime.datetime.now()),
            "dimensionName": dimension_name,
            "slices": [],
            "topSlicesByPerformance": [],
            "bottomSlicesByPerformance": [],
            "largestSlice": {},
            "smallestSlice": {},
            "newStrongestSlice": {},
            "newWeakestSlice": {},
            "comparisonHighlights": [],
            "historicalSliceRankings": {
                "periodsAnalyzed": 0,
                "periodRankings": []
            }
        }

    # 1) Determine current and prior period ranges
    (start_date, end_date) = _get_period_range_for_grain(analysis_date, grain)
    (prior_start, prior_end) = _get_prior_period_range(start_date, end_date, grain)

    # 2) Use the dimension analysis primitive to compare slices between the two periods
    #    We'll rename columns for clarity after the function returns.
    compare_df = compare_dimension_slices_over_time(
        df=ledger_df[
            (ledger_df["metric_id"] == metric_id) &
            (ledger_df["time_grain"] == grain) &
            (ledger_df["dimension"] == dimension_name)
        ],
        slice_col="slice_value",
        date_col="date",
        value_col="metric_value",
        t0=str(prior_start.date()),
        t1=str(start_date.date()),  # We assume the 'start_date' is the anchor date for T1
        agg="sum"
    )

    # The returned DataFrame has columns:
    # [slice_value, val_t0, val_t1, abs_diff, pct_diff]

    # 3) Compute share of volume for T0 and T1
    #    We'll need to do it separately for T0 and T1, then merge.
    t0_df = compare_df[["slice_value", "val_t0"]].copy()
    t0_df = t0_df.rename(columns={"val_t0": "aggregated_value"})
    t0_df = compute_slice_shares(t0_df, "slice_value", val_col="aggregated_value", share_col_name="share_pct_t0")

    t1_df = compare_df[["slice_value", "val_t1"]].copy()
    t1_df = t1_df.rename(columns={"val_t1": "aggregated_value"})
    t1_df = compute_slice_shares(t1_df, "slice_value", val_col="aggregated_value", share_col_name="share_pct_t1")

    merged = compare_df.merge(t0_df[["slice_value", "share_pct_t0"]], on="slice_value", how="left")
    merged = merged.merge(t1_df[["slice_value", "share_pct_t1"]], on="slice_value", how="left")

    # We'll define share_diff = share_pct_t1 - share_pct_t0
    merged["share_diff"] = merged["share_pct_t1"] - merged["share_pct_t0"]

    # 4) Compute difference from average (for val_t1)
    merged = merged.rename(columns={"slice_value": "slice_col"})
    merged = _difference_from_average(merged, current_col="val_t1")

    # 5) Rank slices by performance (val_t1) and share (share_pct_t1)
    #    We'll store ranks in new columns rank_performance, rank_share
    #    Using your rank_metric_slices for val_t1
    performance_ranks = rank_metric_slices(
        agg_df=merged[["slice_col", "val_t1"]].rename(columns={"val_t1": "aggregated_value"}),
        val_col="aggregated_value",
        top_n=len(merged),
        ascending=False
    )
    # performance_ranks returns a subset of rows or the same shape? 
    # Typically it returns the top/bottom. We'll just keep the order. We'll join on slice_col.
    performance_ranks = performance_ranks.reset_index(drop=True).reset_index().rename(columns={"index": "perf_rank"})
    performance_ranks["perf_rank"] = performance_ranks["perf_rank"] + 1

    # We'll do a similar approach for share
    share_ranks = rank_metric_slices(
        agg_df=merged[["slice_col", "share_pct_t1"]].rename(columns={"share_pct_t1": "aggregated_value"}),
        val_col="aggregated_value",
        top_n=len(merged),
        ascending=False
    )
    share_ranks = share_ranks.reset_index(drop=True).reset_index().rename(columns={"index": "share_rank"})
    share_ranks["share_rank"] = share_ranks["share_rank"] + 1

    merged = merged.merge(
        performance_ranks[["slice_col", "perf_rank"]],
        on="slice_col", how="left"
    ).merge(
        share_ranks[["slice_col", "share_rank"]],
        on="slice_col", how="left"
    )

    merged = merged.rename(columns={"perf_rank": "rank_performance", "share_rank": "rank_share"})

    # 6) Build the final "slices" list
    slices_list = _build_slices_list(merged)

    # 7) Identify top and bottom slices by performance
    top_slices, bottom_slices = _compute_top_bottom_slices(merged)

    # 8) Largest & smallest slices by share
    largest_slice, smallest_slice = _largest_smallest_by_share(merged)

    # 9) Check for new strongest / weakest
    new_strongest, new_weakest = _strongest_weakest(merged)

    # 10) Comparison highlights
    comparison_highlights = _highlight_comparison(merged)

    # 11) Historical slice rankings
    historical_rankings = _compute_historical_slice_rankings(ledger_df, metric_id, dimension_name, num_periods=8)

    # Assemble final output
    result = {
        "schemaVersion": "1.0.0",
        "patternName": "DimensionAnalysis",
        "metricId": metric_id,
        "grain": grain,
        "analysisDate": str(analysis_date),
        "evaluationTime": str(datetime.datetime.now()),
        "dimensionName": dimension_name,

        "slices": slices_list,

        "topSlicesByPerformance": top_slices,
        "bottomSlicesByPerformance": bottom_slices,

        "largestSlice": largest_slice,
        "smallestSlice": smallest_slice,

        "newStrongestSlice": new_strongest,
        "newWeakestSlice": new_weakest,

        "comparisonHighlights": comparison_highlights,

        "historicalSliceRankings": historical_rankings
    }
    return result