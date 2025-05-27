# impact_analysis.py
"""
Pattern: ImpactAnalysis
Version: 1.0
Author: Levers Labs Engineering

Purpose:
  Analyzes how an observed change in a parent metric has propagated to its
  direct child metrics. It quantifies the change in the parent and then, for
  each child, its own change and its share of the parent's total change.

Input Format:
  ledger_df (pd.DataFrame): Historical metric data. Required columns:
    - 'metric_id' (str)
    - 'time_grain' (str)
    - 'date' (datetime-like)
    - 'value' (float)
  parent_metric_id (str): The metric whose downstream impact is analyzed.
  child_metric_ids (List[str]): A list of metric IDs that are direct children
                                of the parent_metric_id. This would typically
                                come from the MetricGraph component.
  grain (str): The time_grain for analysis ('day', 'week', 'month').
  analysis_date (str or pd.Timestamp): The reference date for the "current" period.
  evaluation_time (str or pd.Timestamp): Timestamp of when the analysis is run.
  comparison_periods (int): How many grain periods to look back for comparison. Default 1.

Output Format (JSON-serializable dict):
{
  "schemaVersion": "1.0.0",
  "patternName": "ImpactAnalysis",
  "metricId": "<parent_metric_id>",
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",
  "grain": "day" | "week" | "month",
  "observedChangePercent": 10.0,
  "observedChangeAbsolute": 50.0,
  "childImpacts": [
    {
      "childMetricId": "string",
      "childChangePercent": 5.0,        // Can be None
      "childChangeAbsolute": 20.0,
      "shareOfParentImpactPercent": 50.0 // Can be None
    }
    // ... other children
  ]
}
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Union, List, Tuple, Optional

# --- Helper Functions (similar to those in dimensional_analysis.py) ---
def _get_period_boundaries_for_impact(
    analysis_dt: pd.Timestamp,
    grain: str,
    num_periods_offset: int = 0 # 0 for current, 1 for prior, etc.
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculates start and end dates for a period relative to analysis_dt.
    If num_periods_offset is 0, it's the period containing analysis_dt.
    If num_periods_offset is 1, it's the period immediately prior, and so on.
    """
    ref_date = analysis_dt
    if grain == 'day':
        ref_date = analysis_dt - pd.Timedelta(days=num_periods_offset)
        start_dt = ref_date.normalize()
        end_dt = start_dt
    elif grain == 'week':
        # Get start of the week containing analysis_dt
        current_week_start = (analysis_dt - pd.Timedelta(days=analysis_dt.dayofweek)).normalize()
        # Offset by number of weeks
        start_dt = (current_week_start - pd.Timedelta(weeks=num_periods_offset)).normalize()
        end_dt = (start_dt + pd.Timedelta(days=6)).normalize()
    elif grain == 'month':
        current_month_start = analysis_dt.replace(day=1).normalize()
        # Offset by number of months
        start_dt = (current_month_start - pd.DateOffset(months=num_periods_offset)).normalize()
        end_dt = (start_dt + pd.offsets.MonthEnd(0)).normalize()
    else:
        raise ValueError(f"Unsupported grain '{grain}' for period boundary calculation.")
    return start_dt, end_dt

def _get_metric_value_for_period(
    ledger_df: pd.DataFrame,
    metric_id_to_fetch: str,
    time_grain: str,
    period_start_dt: pd.Timestamp,
    period_end_dt: pd.Timestamp,
    aggregation_method: str = 'sum' # How to aggregate if multiple values in period
) -> Optional[float]:
    """
    Fetches and aggregates the value of a metric for a given period.
    """
    period_data = ledger_df[
        (ledger_df['metric_id'] == metric_id_to_fetch) &
        (ledger_df['time_grain'] == time_grain) &
        (pd.to_datetime(ledger_df['date']) >= period_start_dt) &
        (pd.to_datetime(ledger_df['date']) <= period_end_dt)
    ]

    if period_data.empty:
        return None

    if aggregation_method == 'sum':
        return period_data['value'].sum()
    elif aggregation_method == 'mean':
        return period_data['value'].mean()
    elif aggregation_method == 'last': # Get the last reported value in the period
        return period_data.sort_values(by='date')['value'].iloc[-1]
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")


def run_impact_analysis_pattern(
    ledger_df: pd.DataFrame,
    parent_metric_id: str,
    child_metric_ids: List[str], # From MetricGraph
    grain: str,
    analysis_date: Union[str, pd.Timestamp],
    evaluation_time: Union[str, pd.Timestamp],
    comparison_periods_offset: int = 1 # e.g., 1 means compare current vs 1 period ago
) -> Dict[str, Any]:
    analysis_dt = pd.to_datetime(analysis_date)
    eval_time_str = pd.to_datetime(evaluation_time).strftime("%Y-%m-%d %H:%M:%S")
    analysis_date_str = analysis_dt.strftime("%Y-%m-%d")

    output = {
        "schemaVersion": "1.0.0",
        "patternName": "ImpactAnalysis",
        "metricId": parent_metric_id, # The parent metric
        "analysisDate": analysis_date_str,
        "evaluationTime": eval_time_str,
        "grain": grain,
        "observedChangePercent": None,
        "observedChangeAbsolute": None,
        "childImpacts": []
    }

    if ledger_df.empty:
        output["error"] = "Ledger data is empty."
        return output

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(ledger_df['date']):
        ledger_df['date'] = pd.to_datetime(ledger_df['date'])

    # --- 1. Calculate Parent Metric Change ---
    # Current period (T1)
    t1_start, t1_end = _get_period_boundaries_for_impact(analysis_dt, grain, num_periods_offset=0)
    # Prior period (T0)
    t0_start, t0_end = _get_period_boundaries_for_impact(analysis_dt, grain, num_periods_offset=comparison_periods_offset)

    parent_value_t1 = _get_metric_value_for_period(ledger_df, parent_metric_id, grain, t1_start, t1_end)
    parent_value_t0 = _get_metric_value_for_period(ledger_df, parent_metric_id, grain, t0_start, t0_end)

    if parent_value_t1 is None or parent_value_t0 is None:
        output["error"] = f"Could not retrieve values for parent metric '{parent_metric_id}' for one or both periods."
        # Still proceed to check children if possible, but parent changes will be None
    else:
        observed_change_abs = parent_value_t1 - parent_value_t0
        output["observedChangeAbsolute"] = round(observed_change_abs, 2)
        if parent_value_t0 != 0 and pd.notna(parent_value_t0):
            observed_change_pct = (observed_change_abs / abs(parent_value_t0)) * 100.0
            output["observedChangePercent"] = round(observed_change_pct, 2)
        else:
            output["observedChangePercent"] = None # Cannot calculate percent if t0 is zero or NaN

    # --- 2. Calculate Child Metric Changes and Impact ---
    for child_id in child_metric_ids:
        child_value_t1 = _get_metric_value_for_period(ledger_df, child_id, grain, t1_start, t1_end)
        child_value_t0 = _get_metric_value_for_period(ledger_df, child_id, grain, t0_start, t0_end)

        child_impact_obj = {"childMetricId": child_id}

        if child_value_t1 is None or child_value_t0 is None:
            child_impact_obj["childChangeAbsolute"] = None
            child_impact_obj["childChangePercent"] = None
            child_impact_obj["shareOfParentImpactPercent"] = None
            child_impact_obj["error"] = "Missing data for one or both periods."
        else:
            child_change_abs = child_value_t1 - child_value_t0
            child_impact_obj["childChangeAbsolute"] = round(child_change_abs, 2)

            if child_value_t0 != 0 and pd.notna(child_value_t0):
                child_change_pct = (child_change_abs / abs(child_value_t0)) * 100.0
                child_impact_obj["childChangePercent"] = round(child_change_pct, 2)
            else:
                child_impact_obj["childChangePercent"] = None

            # Calculate share of parent's absolute impact
            parent_abs_change = output.get("observedChangeAbsolute")
            if parent_abs_change is not None and parent_abs_change != 0:
                share_pct = (child_change_abs / parent_abs_change) * 100.0
                child_impact_obj["shareOfParentImpactPercent"] = round(share_pct, 2)
            else:
                child_impact_obj["shareOfParentImpactPercent"] = None
        
        output["childImpacts"].append(child_impact_obj)

    return output

if __name__ == '__main__':
    # Sample Ledger Data
    base_analysis_date = datetime(2024, 3, 15)
    dates_data = []
    for i in range(60): # Approx 2 months of daily data
        d = base_analysis_date - timedelta(days=i)
        dates_data.append(d)
    dates_data.sort()

    ledger_rows = []
    np.random.seed(0)

    # Parent Metric: "total_revenue"
    parent_rev_t0 = 10000
    parent_rev_t1 = 12000 # 20% increase, +2000 absolute
    
    # Child Metric 1: "product_A_revenue"
    child_A_rev_t0 = 6000
    child_A_rev_t1 = 7500 # +1500 absolute

    # Child Metric 2: "product_B_revenue"
    child_B_rev_t0 = 4000
    child_B_rev_t1 = 4500 # +500 absolute

    # Populate ledger for daily grain
    for d_idx, current_date in enumerate(dates_data):
        # Simulate data distribution over the period
        # For simplicity, let's assume values are for T0 period if before analysis_date, and T1 if on/after
        # A more realistic simulation would have daily values that sum up to period totals.
        # For this test, we'll just ensure the key dates have the target values.
        
        # For simplicity, we'll make the last day of T0 and T1 periods have the total sum
        # T1 period (current): centered around base_analysis_date
        # T0 period (prior): centered around base_analysis_date - 1 period
        
        # Let's define periods more clearly for testing
        # analysis_date = 2024-03-15 (grain='day')
        # T1 date = 2024-03-15
        # T0 date = 2024-03-14 (comparison_periods_offset=1)
        
        # Parent
        if current_date == base_analysis_date: # T1
            ledger_rows.append(['total_revenue', 'day', current_date, parent_rev_t1])
        elif current_date == base_analysis_date - timedelta(days=1): # T0
            ledger_rows.append(['total_revenue', 'day', current_date, parent_rev_t0])
        else: # Other days
            ledger_rows.append(['total_revenue', 'day', current_date, parent_rev_t0 * (1 + np.random.uniform(-0.05,0.05))])

        # Child A
        if current_date == base_analysis_date:
            ledger_rows.append(['product_A_revenue', 'day', current_date, child_A_rev_t1])
        elif current_date == base_analysis_date - timedelta(days=1):
            ledger_rows.append(['product_A_revenue', 'day', current_date, child_A_rev_t0])
        else:
            ledger_rows.append(['product_A_revenue', 'day', current_date, child_A_rev_t0 * (1 + np.random.uniform(-0.05,0.05))])

        # Child B
        if current_date == base_analysis_date:
            ledger_rows.append(['product_B_revenue', 'day', current_date, child_B_rev_t1])
        elif current_date == base_analysis_date - timedelta(days=1):
            ledger_rows.append(['product_B_revenue', 'day', current_date, child_B_rev_t0])
        else:
            ledger_rows.append(['product_B_revenue', 'day', current_date, child_B_rev_t0 * (1 + np.random.uniform(-0.05,0.05))])
            
        # Child C (no change)
        ledger_rows.append(['product_C_revenue', 'day', current_date, 500])


    sample_ledger = pd.DataFrame(ledger_rows, columns=['metric_id', 'time_grain', 'date', 'value'])
    sample_ledger['date'] = pd.to_datetime(sample_ledger['date'])

    children_of_total_revenue = ['product_A_revenue', 'product_B_revenue', 'product_C_revenue']
    
    print("--- Testing ImpactAnalysis Pattern (grain: day) ---")
    result_day = run_impact_analysis_pattern(
        ledger_df=sample_ledger,
        parent_metric_id='total_revenue',
        child_metric_ids=children_of_total_revenue,
        grain='day',
        analysis_date=base_analysis_date,
        evaluation_time=datetime.now(),
        comparison_periods_offset=1 # Compare to 1 day ago
    )
    import json
    print(json.dumps(result_day, indent=2, default=str))

    # Example for weekly grain (requires data to be summable per week)
    # To make this test meaningful, we'd need to generate weekly sums in ledger_df
    # or ensure _get_metric_value_for_period aggregates daily data if grain is 'week'.
    # The current _get_metric_value_for_period filters by time_grain, so weekly data needs to exist.
    
    # Let's create some weekly data for the test
    weekly_ledger_rows = []
    analysis_week_start_test = (base_analysis_date - timedelta(days=base_analysis_date.dayofweek)).normalize()
    
    # T1 week
    parent_rev_t1_week = parent_rev_t1 * 5 # Simulating 5 days of average T1 value
    child_A_rev_t1_week = child_A_rev_t1 * 5
    child_B_rev_t1_week = child_B_rev_t1 * 5
    for i in range(7):
        d = analysis_week_start_test + timedelta(days=i)
        if d == analysis_week_start_test: # Store sum on the first day of the week
            weekly_ledger_rows.append(['total_revenue_w', 'week', d, parent_rev_t1_week])
            weekly_ledger_rows.append(['product_A_revenue_w', 'week', d, child_A_rev_t1_week])
            weekly_ledger_rows.append(['product_B_revenue_w', 'week', d, child_B_rev_t1_week])

    # T0 week
    prior_week_start_test = analysis_week_start_test - timedelta(weeks=1)
    parent_rev_t0_week = parent_rev_t0 * 5
    child_A_rev_t0_week = child_A_rev_t0 * 5
    child_B_rev_t0_week = child_B_rev_t0 * 5
    for i in range(7):
        d = prior_week_start_test + timedelta(days=i)
        if d == prior_week_start_test:
            weekly_ledger_rows.append(['total_revenue_w', 'week', d, parent_rev_t0_week])
            weekly_ledger_rows.append(['product_A_revenue_w', 'week', d, child_A_rev_t0_week])
            weekly_ledger_rows.append(['product_B_revenue_w', 'week', d, child_B_rev_t0_week])

    sample_ledger_weekly = pd.DataFrame(weekly_ledger_rows, columns=['metric_id', 'time_grain', 'date', 'value'])
    sample_ledger_weekly['date'] = pd.to_datetime(sample_ledger_weekly['date'])

    children_weekly = ['product_A_revenue_w', 'product_B_revenue_w']
    print("\n--- Testing ImpactAnalysis Pattern (grain: week) ---")
    result_week = run_impact_analysis_pattern(
        ledger_df=sample_ledger_weekly, # Use the weekly pre-aggregated data
        parent_metric_id='total_revenue_w',
        child_metric_ids=children_weekly,
        grain='week',
        analysis_date=base_analysis_date, # analysis_date is within the "current week"
        evaluation_time=datetime.now(),
        comparison_periods_offset=1 # Compare to 1 week ago
    )
    print(json.dumps(result_week, indent=2, default=str))

    # Test with missing parent data
    print("\n--- Testing with missing parent data ---")
    temp_ledger = sample_ledger[sample_ledger['metric_id'] != 'total_revenue']
    result_missing_parent = run_impact_analysis_pattern(
        ledger_df=temp_ledger,
        parent_metric_id='total_revenue',
        child_metric_ids=children_of_total_revenue,
        grain='day',
        analysis_date=base_analysis_date,
        evaluation_time=datetime.now()
    )
    print(json.dumps(result_missing_parent, indent=2, default=str))