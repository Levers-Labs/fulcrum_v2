"""
DimensionAnalysis Pattern
-------------------------

This Pattern performs slice-level analysis for a given metric and dimension at a specified
time grain (day, week, or month). It compares the current period to the immediately prior
period, calculates share of total metric volume, performance deltas, ranks slices, identifies
top and bottom slices, and more.

INPUT FORMAT
============

We expect a pandas DataFrame called `ledger_df` with (at least) the following columns:

    - metric_id: str           (e.g., "revenue", "active_users")
    - time_grain: str          ("day", "week", or "month")
    - date: datetime or string (the date associated with each record; for weekly or monthly
                                grains, this might be the start or end of the period)
    - dimension: str           (the dimension name, e.g., "region" or "product_category")
    - slice_value: str         (a particular slice of that dimension, e.g., "North America")
    - metric_value: float      (the metric value for that slice in that period)

We also accept:
    - metric_id (str): the metric we want to analyze (e.g., "revenue")
    - dimension_name (str): which dimension we're analyzing (e.g., "region")
    - grain (str): "day", "week", or "month"
    - analysis_date (str or datetime): the focal date for the analysis. For a "day" grain,
      we look at that particular day. For "week" grain, we look at that calendar week.
      For "month" grain, we look at that calendar month.

OUTPUT
======

A dictionary conforming to the structure below (pseudo-JSON). It includes:

{
  "schemaVersion": "1.0.0",
  "patternName": "DimensionAnalysis",
  "metricId": "revenue",
  "grain": "day" | "week" | "month",

  "analysisDate": "2025-02-05",
  "evaluationTime": "2025-02-05 03:15:00",

  "dimensionName": "region",

  // -- Detailed slice-level data --
  "slices": [
    {
      // The dimension slice (e.g. "North America")
      "sliceValue": "North America",

      // Current & prior average metric values for this slice
      "currentValue": 123.4,
      "priorValue": 115.0,
      "absoluteChange": 8.4,             // (123.4 - 115.0)
      "relativeChangePercent": 7.3,      // (8.4 / 115.0) * 100

      // Current & prior share of the total metric volume
      "currentShareOfVolumePercent": 25.0,
      "priorShareOfVolumePercent": 22.0,
      "shareOfVolumeChangePercent": 3.0, // (3 / 22 = ~13.6% but we store it in "points" or direct difference—up to you

      // Marginal impact on the overall metric. 
      // E.g., how many absolute points of the total metric’s delta are attributable to changes in this slice?
      "absoluteMarginalImpact": 5.0,
      "relativeMarginalImpactPercent": 4.2,

      // For comparing to the average across all other slices
      "avgOtherSlicesValue": 107.3,
      "absoluteDiffFromAvg": 16.1,               // 123.4 - 107.3
      "absoluteDiffPercentFromAvg": 15.0,         // (16.1 / 107.3) * 100

      // Streak / ranking info
      "consecutiveAboveAvgStreak": 3,
      "rankByPerformance": 1,
      "rankByShare": 2
    }
  ],

  // -- Top slices by performance --
  "topSlicesByPerformance": [
    {
      "sliceValue": "North America",
      "metricValue": 123.4,

      // The average across all OTHER slices
      "avgOtherSlicesValue": 98.3,
      // Absolute difference from that average
      "absoluteDiffFromAvg": 25.1,
      // Percentage difference from that average
      "absoluteDiffPercentFromAvg": 25.52,
      "performanceRank": 1
    }
  ],

  // -- Bottom slices by performance --
  "bottomSlicesByPerformance": [
    {
      "sliceValue": "APAC",
      "metricValue": 70.0,
      "avgOtherSlicesValue": 99.5,
      "absoluteDiffFromAvg": -29.5,
      "absoluteDiffPercentFromAvg": -29.65,
      "performanceRank": -1
    }
  ],

  // -- Largest & smallest slices by volume share (if relevant) --
  "largestSlice": {
    "sliceValue": "North America",
    "currentShareOfVolumePercent": 25.0,
    "previousLargestSliceValue": "Europe",
    "previousLargestSharePercent": 21.0
  },
  "smallestSlice": {
    "sliceValue": "APAC",
    "currentShareOfVolumePercent": 15.0,
    "previousSmallestSliceValue": "Latin America",
    "previousSmallestSharePercent": 16.0,
  },

  // -- Newly strongest / weakest slice (by performance) --
  "newStrongestSlice": {
    "sliceValue": "North America",
    "previousStrongestSliceValue": "Europe",

    // Current vs. prior average metric value for this slice
    "currentValue": 123.4,
    "priorValue": 115.0,

    "absoluteDelta": 8.4,
    "relativeDeltaPercent": 7.3
  },
  "newWeakestSlice": {
    "sliceValue": "APAC",
    "previousWeakestSliceValue": "Latin America",

    "currentValue": 70.0,
    "priorValue": 65.0,

    "absoluteDelta": 5.0,
    "relativeDeltaPercent": 7.69
  },

  // -- Slice-to-slice comparison highlights --
  "comparisonHighlights": [
    {
      "sliceA": "North America",
      "currentValueA": 123.4,
      "priorValueA": 115.0,

      "sliceB": "Europe",
      "currentValueB": 95.0,
      "priorValueB": 100.0,

      "performanceGapPercent": 29.89,   // e.g. (123.4 - 95) / 95 * 100
      "gapChangePercent": 4.5          // how the gap grew or shrank vs. prior period
    }
  ],

  // -- Historical slice rank data: top 5 slices each period for the last 8 weeks, for example --
  "historicalSliceRankings": {
    "periodsAnalyzed": 8,
    "periodRankings": [
      {
        "startDate": "2024-12-10",
        "endDate": "2024-12-16",
        "top5SlicesByPerformance": [
          { "sliceValue": "North America", "metricValue": 110.0 },
          { "sliceValue": "Latin America", "metricValue": 85.0 }
          // ...
        ]
      },
      {
        "startDate": "2024-12-17",
        "endDate": "2024-12-23",
        "top5SlicesByPerformance": [
          { "sliceValue": "North America", "metricValue": 115.0 },
          { "sliceValue": "Europe", "metricValue": 100.0 }
          // ...
        ]
      }
      // ...
    ]
  }
}

We rely on existing Dimension-related primitives in `DimensionAnalysis.py` (e.g.,
`calculate_slice_metrics`, `compute_slice_shares`, etc.). We do NOT rewrite or duplicate
those primitives here; we simply reference them. If you need changes to those primitives,
please call it out.

Author: Fulcrum Intelligence Engineering
Date: 2025-02-05
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any

# We reference the dimension-related primitives here. We assume they exist in:
# from intelligence_engine.primitives.DimensionAnalysis import (
#     calculate_slice_metrics,
#     compute_slice_shares,
#     rank_metric_slices,
#     compare_dimension_slices_over_time,
#     analyze_composition_changes,
#     calculate_concentration_index,
#     ...
# )
#
# For brevity, we won't rewrite them in this file.

def _get_period_range_for_grain(analysis_date, grain='day'):
    """
    Given an analysis_date and a grain ('day','week','month'), return
    a (start_date, end_date) range for that period.

    This is a helper. E.g.:
      - if grain='day', we return (analysis_date, analysis_date).
      - if grain='week', we find the Monday-Sunday range containing analysis_date.
      - if grain='month', we find the 1st..end-of-month range containing analysis_date.
    """
    dt = pd.to_datetime(analysis_date)
    if grain == 'day':
        start = dt.normalize()
        end = start
    elif grain == 'week':
        # We'll assume Monday-based weeks.
        # ISO weekday: Monday=1, Sunday=7
        day_of_week = dt.isoweekday()  # Monday=1
        # start = dt - pd.Timedelta(days=(day_of_week-1))
        start = (dt - pd.Timedelta(days=(day_of_week - 1))).normalize()
        end = start + pd.Timedelta(days=6)
    elif grain == 'month':
        start = dt.replace(day=1).normalize()
        # next month:
        next_month = (start + pd.offsets.MonthBegin(1))
        end = (next_month - pd.Timedelta(days=1)).normalize()
    else:
        raise ValueError(f"Unsupported grain '{grain}'")
    return (start, end)

def _get_prior_period_range(start_date, end_date, grain='day'):
    """
    Given the (start_date, end_date) for the current period,
    return the (start_date, end_date) for the immediately prior period
    with the same length. For example:
      - if the current period is a single day, prior period is the previous day.
      - if the current period is a week, prior period is the previous 7 days, etc.
      - if the current period is a month, prior is the previous calendar month.
    """
    if grain == 'day':
        # Prior day
        prior_start = start_date - pd.Timedelta(days=1)
        prior_end = end_date - pd.Timedelta(days=1)
    elif grain == 'week':
        # shift by 7 days
        prior_start = start_date - pd.Timedelta(days=7)
        prior_end = end_date - pd.Timedelta(days=7)
    elif grain == 'month':
        # If start_date is the 1st, end_date is the last day. Let's subtract 1 month.
        # We'll do a naive approach: subtract a MonthBegin from start_date, etc.
        prior_month_start = start_date - pd.offsets.MonthBegin(1)
        prior_month_end = prior_month_start + pd.offsets.MonthEnd(0)
        prior_start = prior_month_start
        prior_end = prior_month_end
    else:
        raise ValueError(f"Unsupported grain '{grain}' in prior period calculation.")
    return (prior_start, prior_end)

def _compute_current_vs_prior(ledger_df, metric_id, dimension_name, grain, analysis_date):
    """
    Filter ledger_df to the current period and the prior period for the given metric/dimension,
    then compute per-slice sums for both periods.

    Returns:
        current_df:  DataFrame of shape [n_slices, 2], columns=[slice_value, current_value]
        prior_df:    DataFrame of shape [n_slices, 2], columns=[slice_value, prior_value]
        merged_df:   DataFrame with columns [slice_value, current_value, prior_value],
                     with 0 filled where slices missing in one period.
    """
    # 1) Get current period range
    (start_date, end_date) = _get_period_range_for_grain(analysis_date, grain)
    # 2) Get prior period range
    (pstart, pend) = _get_prior_period_range(start_date, end_date, grain)

    # Filter ledger for the metric, dimension_name
    dff = ledger_df[
        (ledger_df['metric_id'] == metric_id) &
        (ledger_df['time_grain'] == grain) &
        (ledger_df['dimension'] == dimension_name)
    ].copy()

    # ensure date col is datetime
    dff['date'] = pd.to_datetime(dff['date'])

    # Current period sub-DF
    curr_mask = (dff['date'] >= start_date) & (dff['date'] <= end_date)
    curr_df = dff[curr_mask]
    # Prior period sub-DF
    prior_mask = (dff['date'] >= pstart) & (dff['date'] <= pend)
    pr_df = dff[prior_mask]

    # group by slice_value => sum
    current_agg = curr_df.groupby('slice_value')['metric_value'].sum().reset_index(name='current_value')
    prior_agg   = pr_df.groupby('slice_value')['metric_value'].sum().reset_index(name='prior_value')

    # Merge to get all slices
    merged = pd.merge(
        current_agg, prior_agg,
        on='slice_value', how='outer'
    ).fillna(0)

    return current_agg, prior_agg, merged

def _compute_slice_statistics(merged_df):
    """
    Given a DataFrame with columns [slice_value, current_value, prior_value],
    compute absoluteChange, relativeChangePercent, shareOfVolume, difference from average, etc.

    Returns an expanded DataFrame with the new fields:
      [slice_value, current_value, prior_value,
       absoluteChange, relativeChangePercent,
       currentShareOfVolumePercent, priorShareOfVolumePercent, shareOfVolumeChangePercent,
       absoluteMarginalImpact, relativeMarginalImpactPercent,
       avgOtherSlicesValue, absoluteDiffFromAvg, absoluteDiffPercentFromAvg]

    NOTE: We'll do sums for both periods, so "current_value" is the total sum
    for that slice in the current period. Similarly for "prior_value".
    """
    dff = merged_df.copy()

    # total current / total prior
    total_current = dff['current_value'].sum()
    total_prior   = dff['prior_value'].sum()
    total_delta   = total_current - total_prior

    # compute absolute & relative changes
    dff['absoluteChange'] = dff['current_value'] - dff['prior_value']
    def safe_rel(row):
        if row['prior_value'] == 0:
            return np.nan
        return (row['absoluteChange'] / row['prior_value']) * 100
    dff['relativeChangePercent'] = dff.apply(safe_rel, axis=1)

    # share of volume
    def safe_share(curr_val, total_val):
        if total_val == 0:
            return 0.0
        return (curr_val / total_val) * 100.0

    dff['currentShareOfVolumePercent'] = dff['current_value'].apply(lambda x: safe_share(x, total_current))
    dff['priorShareOfVolumePercent']   = dff['prior_value'].apply(lambda x: safe_share(x, total_prior))
    dff['shareOfVolumeChangePercent']  = dff['currentShareOfVolumePercent'] - dff['priorShareOfVolumePercent']

    # marginal impact (absolute & relative)
    def safe_marginal(slice_delta):
        # how many points of total metric delta are from this slice
        return slice_delta  # typically it's the slice_delta itself
    dff['absoluteMarginalImpact'] = dff['absoluteChange'].apply(safe_marginal)
    def safe_rel_marg(row):
        if total_delta == 0:
            return 0.0
        return (row['absoluteChange'] / total_delta) * 100
    dff['relativeMarginalImpactPercent'] = dff.apply(safe_rel_marg, axis=1)

    # difference from average of other slices (current)
    # for each slice s: sum_of_others = total_current - s.current_value
    # num_others = len(dff) - 1
    # if num_others <= 0 => skip
    n_slices = len(dff)
    def diff_from_avg(row):
        if n_slices <= 1:
            return (np.nan, np.nan)
        sum_others = total_current - row['current_value']
        avg_others = sum_others / (n_slices - 1)
        abs_diff   = row['current_value'] - avg_others
        if avg_others == 0:
            pct_diff = np.nan
        else:
            pct_diff = (abs_diff / avg_others) * 100
        return (avg_others, abs_diff, pct_diff)

    dff['avgOtherSlicesValue'] = np.nan
    dff['absoluteDiffFromAvg'] = np.nan
    dff['absoluteDiffPercentFromAvg'] = np.nan

    for i in range(n_slices):
        row = dff.iloc[i]
        (avgo, absd, pctd) = diff_from_avg(row)
        dff.at[i, 'avgOtherSlicesValue']           = avgo
        dff.at[i, 'absoluteDiffFromAvg']           = absd
        dff.at[i, 'absoluteDiffPercentFromAvg']    = pctd

    return dff

def _compute_ranks_and_streaks(expanded_df):
    """
    For each slice, compute:
      - rankByPerformance (1=best, 2=second, ...)
      - rankByShare
      - consecutiveAboveAvgStreak => For simplicity, we’ll do a placeholder approach
        or we could do a short historical lookback. 
        We'll just set it to 0 or 1 if the slice is above the average in this period.

    Returns the same DataFrame with new columns:
      - rankByPerformance
      - rankByShare
      - consecutiveAboveAvgStreak
    """
    dff = expanded_df.copy()
    # rank by performance => sort descending by current_value
    dff['rankByPerformance'] = dff['current_value'].rank(method='dense', ascending=False).astype(int)
    # rank by share => sort descending by currentShareOfVolumePercent
    dff['rankByShare']       = dff['currentShareOfVolumePercent'].rank(method='dense', ascending=False).astype(int)

    # consecutiveAboveAvgStreak => placeholder: if current_value > avgOtherSlicesValue => set 1, else 0
    # in real usage, you'd gather historical data and see how many consecutive periods it has been > average
    dff['consecutiveAboveAvgStreak'] = dff.apply(
        lambda row: 1 if row['current_value'] > row['avgOtherSlicesValue'] else 0,
        axis=1
    )

    return dff

def _compute_top_and_bottom_slices(dff, top_n=3):
    """
    Given the expanded DataFrame, pick the top slices by performance (current_value)
    and the bottom slices. Return two arrays of dict with relevant data.
    """
    # sort by current_value descending
    sorted_df = dff.sort_values(by='current_value', ascending=False).reset_index(drop=True)

    # top slices
    top_slices = []
    top_n = min(top_n, len(sorted_df))
    for i in range(top_n):
        row = sorted_df.iloc[i]
        top_slices.append({
            "sliceValue": row['slice_value'],
            "metricValue": row['current_value'],
            "avgOtherSlicesValue": row['avgOtherSlicesValue'],
            "absoluteDiffFromAvg": row['absoluteDiffFromAvg'],
            "absoluteDiffPercentFromAvg": row['absoluteDiffPercentFromAvg'],
            "performanceRank": row['rankByPerformance']
        })

    # bottom slices => sort ascending
    bottom_sorted_df = dff.sort_values(by='current_value', ascending=True).reset_index(drop=True)
    bottom_slices = []
    top_n = min(top_n, len(bottom_sorted_df))
    for i in range(top_n):
        row = bottom_sorted_df.iloc[i]
        bottom_slices.append({
            "sliceValue": row['slice_value'],
            "metricValue": row['current_value'],
            "avgOtherSlicesValue": row['avgOtherSlicesValue'],
            "absoluteDiffFromAvg": row['absoluteDiffFromAvg'],
            "absoluteDiffPercentFromAvg": row['absoluteDiffPercentFromAvg'],
            "performanceRank": -1 * row['rankByPerformance']
        })

    return top_slices, bottom_slices

def _compute_largest_and_smallest_slice(dff):
    """
    Identify largest slice by currentShareOfVolumePercent, and smallest slice,
    plus record who was largest/smallest in the prior period.
    Returns (largestSliceDict, smallestSliceDict).
    """
    # largest current
    largest_current = dff.sort_values(by='currentShareOfVolumePercent', ascending=False).head(1)
    # smallest current
    smallest_current = dff.sort_values(by='currentShareOfVolumePercent', ascending=True).head(1)

    # largest prior
    largest_prior = dff.sort_values(by='priorShareOfVolumePercent', ascending=False).head(1)
    # smallest prior
    smallest_prior = dff.sort_values(by='priorShareOfVolumePercent', ascending=True).head(1)

    if len(largest_current)==1 and len(largest_prior)==1:
        largest_dict = {
            "sliceValue": largest_current.iloc[0]['slice_value'],
            "currentShareOfVolumePercent": largest_current.iloc[0]['currentShareOfVolumePercent'],
            "previousLargestSliceValue": largest_prior.iloc[0]['slice_value'],
            "previousLargestSharePercent": largest_prior.iloc[0]['priorShareOfVolumePercent']
        }
    else:
        largest_dict = {}

    if len(smallest_current)==1 and len(smallest_prior)==1:
        smallest_dict = {
            "sliceValue": smallest_current.iloc[0]['slice_value'],
            "currentShareOfVolumePercent": smallest_current.iloc[0]['currentShareOfVolumePercent'],
            "previousSmallestSliceValue": smallest_prior.iloc[0]['slice_value'],
            "previousSmallestSharePercent": smallest_prior.iloc[0]['priorShareOfVolumePercent']
        }
    else:
        smallest_dict = {}

    return largest_dict, smallest_dict

def _compute_new_strongest_weakest(dff):
    """
    Identify the top slice by current_value, see who was top in prior_value,
    and if they're different, we have a 'newStrongestSlice'.
    Similarly for the weakest slice.

    Return (newStrongestSliceDict, newWeakestSliceDict).
    """
    # current top => highest current_value
    current_top = dff.sort_values(by='current_value', ascending=False).head(1)
    prior_top  = dff.sort_values(by='prior_value', ascending=False).head(1)

    # current bottom => lowest current_value
    current_bottom = dff.sort_values(by='current_value', ascending=True).head(1)
    prior_bottom  = dff.sort_values(by='prior_value', ascending=True).head(1)

    strongest = {}
    weakest   = {}

    if len(current_top)==1 and len(prior_top)==1:
        curr_slice = current_top.iloc[0]['slice_value']
        prior_slice= prior_top.iloc[0]['slice_value']
        if curr_slice != prior_slice:
            strongest = {
                "sliceValue": curr_slice,
                "previousStrongestSliceValue": prior_slice,
                "currentValue": current_top.iloc[0]['current_value'],
                "priorValue":  current_top.iloc[0]['prior_value'],
            }
            abs_delta = strongest["currentValue"] - strongest["priorValue"]
            strongest["absoluteDelta"] = abs_delta
            if strongest["priorValue"] == 0:
                strongest["relativeDeltaPercent"] = np.nan
            else:
                strongest["relativeDeltaPercent"] = (abs_delta / strongest["priorValue"])*100

    if len(current_bottom)==1 and len(prior_bottom)==1:
        curr_slice = current_bottom.iloc[0]['slice_value']
        prior_slice= prior_bottom.iloc[0]['slice_value']
        if curr_slice != prior_slice:
            weakest = {
                "sliceValue": curr_slice,
                "previousWeakestSliceValue": prior_slice,
                "currentValue": current_bottom.iloc[0]['current_value'],
                "priorValue":  current_bottom.iloc[0]['prior_value'],
            }
            abs_delta = weakest["currentValue"] - weakest["priorValue"]
            weakest["absoluteDelta"] = abs_delta
            if weakest["priorValue"] == 0:
                weakest["relativeDeltaPercent"] = np.nan
            else:
                weakest["relativeDeltaPercent"] = (abs_delta / weakest["priorValue"])*100

    return strongest, weakest

def _compute_comparison_highlights(dff):
    """
    For demonstration, let's compare the top 2 slices by current_value.
    We'll compute the performance gap and how it changed from the prior period.

    Return a list with up to 1 highlight (if we have at least 2 slices).
    """
    highlight_list = []
    sorted_current = dff.sort_values(by='current_value', ascending=False).reset_index(drop=True)
    if len(sorted_current) >= 2:
        sliceA = sorted_current.iloc[0]
        sliceB = sorted_current.iloc[1]
        # performanceGapPercent => (A-B)/B * 100
        gap_now = None
        if sliceB['current_value'] != 0:
            gap_now = ((sliceA['current_value'] - sliceB['current_value']) / sliceB['current_value'])*100
        # gap in prior period
        gap_prior = None
        if sliceB['prior_value'] != 0:
            gap_prior = ((sliceA['prior_value'] - sliceB['prior_value']) / sliceB['prior_value'])*100

        gap_change = None
        if gap_now is not None and gap_prior is not None:
            gap_change = gap_now - gap_prior

        highlight_list.append({
            "sliceA": sliceA['slice_value'],
            "currentValueA": sliceA['current_value'],
            "priorValueA": sliceA['prior_value'],

            "sliceB": sliceB['slice_value'],
            "currentValueB": sliceB['current_value'],
            "priorValueB": sliceB['prior_value'],

            "performanceGapPercent": gap_now,
            "gapChangePercent": gap_change
        })
    return highlight_list

def _compute_historical_slice_rankings(
    ledger_df,
    metric_id: str,
    dimension_name: str,
    num_periods=8
):
    """
    Example: for the last `num_periods` weeks, find the top 5 slices each period by sum of metric_value.
    We'll produce:
    {
      "periodsAnalyzed": 8,
      "periodRankings": [
        {
          "startDate": "...",
          "endDate": "...",
          "top5SlicesByPerformance": [
            { "sliceValue": ..., "metricValue": ...}, ...
          ]
        }, ...
      ]
    }

    Implementation: We'll look at the last num_periods weekly intervals from today.
    This is a simplified approach. For day or month grain, you'd adapt similarly.
    """
    # We'll do a simple approach: find the last num_periods weeks counting backwards from "today".
    # For real usage, you'd parameterize the grain.
    # Let "today" = max date in ledger for that metric/dimension, or just datetime.now().
    # We'll do something simplistic here.

    if ledger_df.empty:
        return {
            "periodsAnalyzed": 0,
            "periodRankings": []
        }

    # filter
    dff = ledger_df[
        (ledger_df['metric_id'] == metric_id) &
        (ledger_df['dimension'] == dimension_name)
    ].copy()
    if dff.empty:
        return {
            "periodsAnalyzed": 0,
            "periodRankings": []
        }

    # naive approach: let end_of_latest = max(dff['date'])
    end_of_latest = pd.to_datetime(dff['date'].max())
    # We'll build weekly intervals
    periodRankings = []
    current_end = end_of_latest

    for i in range(num_periods):
        period_start = (current_end - pd.Timedelta(days=6)).normalize()
        # slice that period
        mask = (dff['date'] >= period_start) & (dff['date'] <= current_end)
        d_slice = dff[mask].groupby('slice_value')['metric_value'].sum().reset_index()
        d_slice.sort_values('metric_value', ascending=False, inplace=True)
        # pick top 5
        top5 = d_slice.head(5)
        top5_list = [
            {"sliceValue": row['slice_value'], "metricValue": row['metric_value']}
            for _, row in top5.iterrows()
        ]
        periodRankings.append({
            "startDate": str(period_start.date()),
            "endDate": str(current_end.date()),
            "top5SlicesByPerformance": top5_list
        })
        # move to previous interval
        current_end = (period_start - pd.Timedelta(days=1))

    # reverse so the oldest is first
    periodRankings.reverse()

    return {
        "periodsAnalyzed": num_periods,
        "periodRankings": periodRankings
    }


def run_dimension_analysis(
    ledger_df: pd.DataFrame,
    metric_id: str,
    dimension_name: str,
    grain: str,
    analysis_date: str or pd.Timestamp
) -> Dict[str, Any]:
    """
    Orchestrates a full "DimensionAnalysis" Pattern for the given metric, dimension,
    and time grain. Compares the current period to the prior period, calculates slice-level
    stats, ranks slices by performance & share, identifies top/bottom slices, largest/smallest,
    newly strongest/weakest, produces comparison highlights, and builds a historical slice ranking.

    Returns a dict matching the JSON structure specified in the pattern requirements.
    """
    # Quick check: does ledger_df contain this dimension?
    if not ((ledger_df['dimension'] == dimension_name) & (ledger_df['metric_id'] == metric_id)).any():
        # This metric might not have the requested dimension. Skip or return an empty structure.
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

    # 1) get the current vs prior data
    current_df, prior_df, merged_df = _compute_current_vs_prior(
        ledger_df, metric_id, dimension_name, grain, analysis_date
    )
    if merged_df.empty:
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

    # 2) compute slice-level stats
    expanded = _compute_slice_statistics(merged_df)
    expanded = _compute_ranks_and_streaks(expanded)

    # build slices array for the final JSON
    slices_list = []
    for _, row in expanded.iterrows():
        # each row => build a dictionary
        slices_list.append({
            "sliceValue": row['slice_value'],
            "currentValue": row['current_value'],
            "priorValue": row['prior_value'],
            "absoluteChange": row['absoluteChange'],
            "relativeChangePercent": row['relativeChangePercent'],
            "currentShareOfVolumePercent": row['currentShareOfVolumePercent'],
            "priorShareOfVolumePercent": row['priorShareOfVolumePercent'],
            "shareOfVolumeChangePercent": row['shareOfVolumeChangePercent'],
            "absoluteMarginalImpact": row['absoluteMarginalImpact'],
            "relativeMarginalImpactPercent": row['relativeMarginalImpactPercent'],
            "avgOtherSlicesValue": row['avgOtherSlicesValue'],
            "absoluteDiffFromAvg": row['absoluteDiffFromAvg'],
            "absoluteDiffPercentFromAvg": row['absoluteDiffPercentFromAvg'],
            "consecutiveAboveAvgStreak": row['consecutiveAboveAvgStreak'],
            "rankByPerformance": row['rankByPerformance'],
            "rankByShare": row['rankByShare']
        })

    # 3) top & bottom slices
    top_slices, bottom_slices = _compute_top_and_bottom_slices(expanded)

    # 4) largest & smallest slices by share
    largest_slice, smallest_slice = _compute_largest_and_smallest_slice(expanded)

    # 5) newly strongest & weakest
    new_strongest, new_weakest = _compute_new_strongest_weakest(expanded)

    # 6) comparison highlights
    comparison_highlights = _compute_comparison_highlights(expanded)

    # 7) historical slice rankings
    #    We'll assume weekly approach for demonstration, but you might param by `grain`
    historical_rankings = _compute_historical_slice_rankings(ledger_df, metric_id, dimension_name, num_periods=8)

    # Finally, assemble the output
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