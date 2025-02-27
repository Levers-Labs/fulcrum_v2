"""
AnomalyAnalysis Pattern

Purpose:
---------
This Pattern detects anomalies ("spikes" or "drops") in a given metric, at a specified time grain
(day, week, or month), for all available dimension slices. It generates a JSON output indicating
when anomalies occur, what type of anomaly they are, which method (z-score or SPC) detected them,
and the numerical context (e.g., z-score, deviation from control limits, etc.).

Input Format:
-------------
We assume you have a Pandas DataFrame ('ledger_df') that contains all metrics at various grains
and dimensions, with columns:
    - 'metric_id' (str): The unique ID or name of the metric.
    - 'time_grain' (str): One of ['day','week','month'] or some standard naming convention.
    - 'dimension' (str): The dimension name (e.g., "region", "product_category"). If a metric
        has multiple dimensions, you may store them differently; for simplicity, we assume
        a single dimension column. If a metric is dimensionless, this column might be "Total"
        or something similar.
    - 'slice' (str): The slice or value within the dimension (e.g., "North America", "Widgets", etc.).
      If no dimension applies, it could be "Total".
    - 'date' (datetime-like): The date or period start date for the metric value.
    - 'value' (float): The numeric value of the metric on this date/dimension/slice.

Additionally, you will provide:
    - metric_id (str): The metric for which to run anomaly analysis.
    - grain (str): One of ['day','week','month'] specifying the time grain to filter.
    - analysis_date (str): The date for which the analysis is relevant (e.g., '2025-02-15').
    - evaluation_time (str): A timestamp for when this analysis is executed (e.g., '2025-02-15 10:00:00').

Output Format (JSON-serializable dict):
---------------------------------------
{
  "schemaVersion": "1.0.0",
  "patternName": "AnomalyAnalysis",
  "metricId": "string",
  "grain": "day" | "week" | "month",

  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",

  "anomalies": [
    {
      "date": "YYYY-MM-DD",
      "type": "Spike" | "Drop",

      // Which method we used: z-score or SPC
      "method": "z-score",
      // For z-score method:
      "zScore": 3.5,

      // For SPC method (if method="SPC"):
      // "absoluteDeviationFromControlLimit": 20.0,

      // The actual real magnitude of the difference (not just %)
      "realMagnitude": 100.0,
      "magnitudePercent": 15.0
    },
    // Example for SPC method
    {
      "date": "YYYY-MM-DD",
      "type": "Spike",
      "method": "SPC",
      "absoluteDeviationFromControlLimit": 15.3,
      "realMagnitude": 50.0,
      "magnitudePercent": 5.2
    }
  ]
}

Algorithm Details:
------------------
1. Filter the ledger_df for rows matching:
   - the desired metric_id
   - the desired grain
2. For each dimension-slice subset:
   a) Run the SPC analysis (via the `process_control_analysis` function from the TrendAnalysis
      primitives). Identify days where the metric crosses the upper or lower control limit.
   b) Compute the z-score for each day (relative to the entire dimension-slice series). Mark
      any point with an absolute z-score above a threshold that corresponds roughly to a 1% tail
      (e.g. |z| >= 2.33) as an anomaly.
   c) Collect anomaly info. For SPC anomalies:
       - date => row date
       - type => "Spike" if value > UCL, "Drop" if value < LCL
       - method => "SPC"
       - absoluteDeviationFromControlLimit => how far above UCL (or below LCL) the value is
       - realMagnitude => the actual difference (same as above, effectively)
       - magnitudePercent => (difference / center_line) * 100, as an example
     For z-score anomalies:
       - date => row date
       - type => "Spike" if z-score > 0, "Drop" if z-score < 0
       - method => "z-score"
       - zScore => computed z-score
       - realMagnitude => (value - mean)
       - magnitudePercent => ((value - mean) / mean) * 100
3. Merge all anomalies into a single list (you can keep them sorted by date). 
4. Produce the final JSON.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# We'll reference your TrendAnalysis primitives for SPC:
# from intelligence_engine.primitives.TrendAnalysis import process_control_analysis
# But for this snippet, let's assume we've imported it as:
#   from trend_analysis_primitives import process_control_analysis
# and so on. (Adjust the actual import to match your file structure.)

# If you do not already have a straightforward z-score approach, we can use scipy:
from scipy.stats import zscore

def run_anomaly_analysis(
    ledger_df: pd.DataFrame,
    metric_id: str,
    grain: str,
    analysis_date: str,
    evaluation_time: str,
    date_col: str = "date",
    value_col: str = "value",
    dim_col: str = "dimension",
    slice_col: str = "slice"
) -> dict:
    """
    Execute the AnomalyAnalysis pattern.

    Parameters
    ----------
    ledger_df : pd.DataFrame
        In-memory ledger containing metric data for many metrics/dimensions, with columns:
            ['metric_id', 'time_grain', 'dimension', 'slice', 'date', 'value']
    metric_id : str
        The metric ID/name to analyze.
    grain : str
        The time grain to filter on, one of ['day','week','month'].
    analysis_date : str
        e.g. "2025-02-15"
    evaluation_time : str
        e.g. "2025-02-15 10:00:00"
    date_col : str, default "date"
    value_col : str, default "value"
    dim_col : str, default "dimension"
    slice_col : str, default "slice"

    Returns
    -------
    dict
        A JSON-serializable dictionary matching the specified Pattern output format.
    """

    # -------------------------------------------------------------------------
    # 1. Filter ledger by metric_id and grain
    # -------------------------------------------------------------------------
    df_filtered = ledger_df[
        (ledger_df["metric_id"] == metric_id) &
        (ledger_df["time_grain"] == grain)
    ].copy()
    if df_filtered.empty:
        # If there's no data after filtering, return a default structure
        return {
            "schemaVersion": "1.0.0",
            "patternName": "AnomalyAnalysis",
            "metricId": metric_id,
            "grain": grain,
            "analysisDate": analysis_date,
            "evaluationTime": evaluation_time,
            "anomalies": []
        }

    # -------------------------------------------------------------------------
    # 2. We'll want to do SPC analysis and z-score analysis per dimension slice
    # -------------------------------------------------------------------------
    # For dimensionless metrics, you might only have one dimension=="Total" or none at all.
    # We'll group by dimension/slice if present.
    group_cols = []
    if dim_col in df_filtered.columns and slice_col in df_filtered.columns:
        group_cols = [dim_col, slice_col]

    anomalies_list = []

    # define the z-score threshold that approximates a 1% tail for two-sided
    # (roughly ~2.33 for single-tailed 1%; ~2.58 for two-tailed 1%. Let's pick 2.33).
    ZSCORE_THRESHOLD = 2.33

    # We'll loop dimension-slice groups
    for group_key, sub_df in df_filtered.groupby(group_cols, dropna=False):

        # Sort ascending by date
        sub_df = sub_df.sort_values(by=date_col).reset_index(drop=True)

        # A. SPC Analysis
        #    Reference existing process_control_analysis from TrendAnalysis to get columns
        #    [central_line, ucl, lcl, etc.].
        #    If you do not have enough data points, it should handle gracefully.
        from trend_analysis_primitives import process_control_analysis  # hypothetical import
        spc_df = process_control_analysis(
            df=sub_df,
            date_col=date_col,
            value_col=value_col
        )

        # Identify points above UCL => "Spike", below LCL => "Drop".
        # We'll record the deviation from the limit as realMagnitude,
        # and also store magnitudePercent as (deviation / central_line)*100
        for i, row in spc_df.iterrows():
            val = row[value_col]
            cl  = row["central_line"]
            ucl = row["ucl"]
            lcl = row["lcl"]
            dte = row[date_col]

            if pd.isnull(val) or pd.isnull(ucl) or pd.isnull(lcl) or pd.isnull(cl):
                continue

            # spike if value > UCL
            if val > ucl:
                diff = val - ucl
                anomalies_list.append({
                    "date": dte.strftime("%Y-%m-%d"),
                    "type": "Spike",
                    "method": "SPC",
                    "absoluteDeviationFromControlLimit": float(diff),
                    "realMagnitude": float(diff),
                    "magnitudePercent": float((diff / cl)*100 if cl else 0.0)
                })
            # drop if value < LCL
            elif val < lcl:
                diff = lcl - val
                anomalies_list.append({
                    "date": dte.strftime("%Y-%m-%d"),
                    "type": "Drop",
                    "method": "SPC",
                    "absoluteDeviationFromControlLimit": float(diff),
                    "realMagnitude": float(diff),
                    "magnitudePercent": float((diff / cl)*100 if cl else 0.0)
                })

        # B. Z-score Analysis
        #    We'll compute the z-score for each data point across the entire sub_df,
        #    then mark anomalies if abs(zscore) >= ZSCORE_THRESHOLD
        #    We interpret "realMagnitude" as (value - mean) and "magnitudePercent" as
        #    ((value - mean)/mean)*100.
        vals = sub_df[value_col].values
        if len(vals) < 2:
            # skip if insufficient data
            continue

        zs = zscore(vals, nan_policy='omit')
        mean_ = np.nanmean(vals)

        for i, zval in enumerate(zs):
            if pd.isnull(zval):
                continue
            if abs(zval) >= ZSCORE_THRESHOLD:
                original_val = sub_df.loc[i, value_col]
                dte = sub_df.loc[i, date_col]
                sign = "Spike" if zval > 0 else "Drop"
                real_mag = original_val - mean_
                anomalies_list.append({
                    "date": dte.strftime("%Y-%m-%d"),
                    "type": sign,
                    "method": "z-score",
                    "zScore": float(zval),
                    "realMagnitude": float(real_mag),
                    "magnitudePercent": float((real_mag / mean_)*100 if mean_ else 0.0)
                })

    # -------------------------------------------------------------------------
    # 3. Prepare final output structure
    # -------------------------------------------------------------------------
    output = {
        "schemaVersion": "1.0.0",
        "patternName": "AnomalyAnalysis",
        "metricId": metric_id,
        "grain": grain,
        "analysisDate": analysis_date,
        "evaluationTime": evaluation_time,
        "anomalies": anomalies_list
    }

    return output


# Example usage (pseudo-code):
if __name__ == "__main__":
    data = {
        "metric_id": ["m1"]*10,
        "time_grain": ["day"]*10,
        "dimension": ["Total"]*10,
        "slice": ["Total"]*10,
        "date": pd.date_range("2025-02-01", periods=10, freq="D"),
        "value": [100, 105, 250, 108, 110, 300, 112, 95, 90, 400]
    }
    ledger_df_example = pd.DataFrame(data)

    analysis_result = run_anomaly_analysis(
        ledger_df=ledger_df_example,
        metric_id="m1",
        grain="day",
        analysis_date="2025-02-15",
        evaluation_time="2025-02-15 10:00:00"
    )

    # This 'analysis_result' is a dict, which you can then serialize to JSON.
    import json
    print(json.dumps(analysis_result, indent=2))
