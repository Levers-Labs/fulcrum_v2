# anomaly_analysis.py
"""
Pattern: AnomalyAnalysis
Version: 1.0

Purpose:
  Detects anomalies (spikes or drops) in a metric time series for each
  dimension-slice combination. Utilizes SPC (Statistical Process Control)
  and z-score methods for detection.

Input Format:
  ledger_df (pd.DataFrame): DataFrame with metric data. Required columns:
    - 'metric_id' (str): Metric identifier.
    - 'time_grain' (str): Granularity ('day', 'week', 'month').
    - 'dimension' (str): Dimension name (e.g., "region", or "Overall" if none).
    - 'slice' (str): Slice within the dimension (e.g., "North America", or "Overall" if none).
    - 'date' (datetime-like): Date of the metric value.
    - 'value' (float): Numeric metric value.
  metric_id (str): The specific metric to analyze.
  grain (str): The time_grain to filter and analyze.
  analysis_date (str): YYYY-MM-DD string indicating the date of analysis.
  evaluation_time (str): YYYY-MM-DD HH:MM:SS string for when the analysis was run.
  (Optional) date_col, value_col, dim_col, slice_col: Custom column names.

Output Format (JSON-serializable dict):
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
      "dimension": "string",
      "slice": "string",
      "type": "Spike" | "Drop",
      "method": "SPC" | "z-score",
      // SPC specific
      "absoluteDeviationFromControlLimit": "float", // If method == "SPC"
      // z-score specific
      "zScore": "float", // If method == "z-score"
      // Common
      "realMagnitude": "float",
      "magnitudePercent": "float" // Can be null if reference (mean/CL) is zero
    },
    // ... more anomalies
  ]
}
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore

# Assuming trend_analysis.py is in a location accessible by PYTHONPATH
# and contains process_control_analysis primitive.
from trend_analysis import process_control_analysis

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
    Executes the AnomalyAnalysis pattern.
    """

    # 1. Filter ledger_df for the relevant metric and grain
    df_metric_filtered = ledger_df[
        (ledger_df["metric_id"] == metric_id) &
        (ledger_df["time_grain"] == grain)
    ].copy()

    if df_metric_filtered.empty:
        return {
            "schemaVersion": "1.0.0",
            "patternName": "AnomalyAnalysis",
            "metricId": metric_id,
            "grain": grain,
            "analysisDate": analysis_date,
            "evaluationTime": evaluation_time,
            "anomalies": []
        }

    # Ensure date column is datetime
    df_metric_filtered[date_col] = pd.to_datetime(df_metric_filtered[date_col])

    anomalies_list = []
    ZSCORE_THRESHOLD = 2.33 # Corresponds to roughly 1% tail probability (one-sided)

    # 2. Group by each dimension-slice combination
    # Assumes dim_col and slice_col are always present, e.g., with "Overall" for dimensionless metrics.
    for (current_dimension, current_slice), series_df in df_metric_filtered.groupby([dim_col, slice_col], dropna=False):
        
        if series_df.empty:
            continue

        # Sort by date for time series operations
        series_df = series_df.sort_values(by=date_col).reset_index(drop=True)

        # a) SPC Analysis
        # process_control_analysis primitive calculates central_line, ucl, lcl
        try:
            spc_results_df = process_control_analysis(
                df=series_df, # Pass the already sorted and filtered series_df
                date_col=date_col,
                value_col=value_col
                # Other parameters for process_control_analysis use their defaults
            )
        except Exception as e:
            # If SPC primitive fails for a slice, log it and skip SPC for this slice
            print(f"Warning: SPC analysis failed for metric '{metric_id}', dimension '{current_dimension}', slice '{current_slice}'. Error: {e}")
            spc_results_df = series_df.copy() # ensure columns exist
            spc_results_df['central_line'] = np.nan
            spc_results_df['ucl'] = np.nan
            spc_results_df['lcl'] = np.nan


        for _, row in spc_results_df.iterrows():
            val = row[value_col]
            cl = row.get("central_line") # .get in case columns aren't created due to error
            ucl = row.get("ucl")
            lcl = row.get("lcl")
            dte = row[date_col]

            if pd.isnull(val) or pd.isnull(cl) or pd.isnull(ucl) or pd.isnull(lcl):
                continue

            anomaly_detail = None
            if val > ucl:
                diff = val - ucl
                anomaly_detail = {
                    "type": "Spike",
                    "absoluteDeviationFromControlLimit": float(diff),
                    "realMagnitude": float(diff),
                    "magnitudePercent": (float(diff / cl) * 100.0) if cl != 0 else np.nan,
                }
            elif val < lcl:
                diff = lcl - val # Difference is positive: how far it dropped below LCL
                anomaly_detail = {
                    "type": "Drop",
                    "absoluteDeviationFromControlLimit": float(diff),
                    "realMagnitude": float(diff), # Magnitude of the drop from LCL
                    "magnitudePercent": (float(diff / cl) * 100.0) if cl != 0 else np.nan,
                }

            if anomaly_detail:
                anomaly_detail.update({
                    "date": dte.strftime("%Y-%m-%d"),
                    "dimension": str(current_dimension),
                    "slice": str(current_slice),
                    "method": "SPC",
                })
                anomalies_list.append(anomaly_detail)

        # b) Z-score Analysis
        # Applied to the entire series for this dimension-slice
        metric_values = series_df[value_col].astype(float) # Ensure numeric
        if len(metric_values.dropna()) >= 2: # zscore needs at least one non-NaN value, practically more
            # scipy.stats.zscore returns NaN for all-NaN or single-value inputs if ddof=0
            # and can handle internal NaNs with nan_policy='omit'
            # It will return 0 for constant series if ddof=0, or nan if ddof=1 (default)
            try:
                z_scores = zscore(metric_values, nan_policy='omit', ddof=1) # ddof=1 for sample std dev
            except Exception as e:
                 print(f"Warning: Z-score calculation failed for metric '{metric_id}', dimension '{current_dimension}', slice '{current_slice}'. Error: {e}")
                 z_scores = np.full(len(metric_values), np.nan)


            series_mean = np.nanmean(metric_values)

            for i, z_val in enumerate(z_scores):
                if pd.isnull(z_val) or pd.isnull(series_mean): # Skip if z-score or mean is NaN
                    continue

                if abs(z_val) >= ZSCORE_THRESHOLD:
                    original_val = metric_values.iloc[i]
                    dte = series_df[date_col].iloc[i]
                    
                    anomaly_type = "Spike" if z_val > 0 else "Drop"
                    real_magnitude = original_val - series_mean
                    
                    magnitude_pct = np.nan
                    if series_mean != 0:
                        magnitude_pct = (real_magnitude / series_mean) * 100.0
                    
                    anomalies_list.append({
                        "date": dte.strftime("%Y-%m-%d"),
                        "dimension": str(current_dimension),
                        "slice": str(current_slice),
                        "type": anomaly_type,
                        "method": "z-score",
                        "zScore": float(z_val),
                        "realMagnitude": float(real_magnitude),
                        "magnitudePercent": float(magnitude_pct) if pd.notna(magnitude_pct) else None,
                    })

    # 3. Sort anomalies by date (optional, but good for consistency)
    if anomalies_list:
        anomalies_list.sort(key=lambda x: (x["date"], x["dimension"], x["slice"], x["method"]))

    # 4. Produce the final JSON-serializable dictionary
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

if __name__ == '__main__':
    # Create a more comprehensive dummy ledger_df for testing
    data_rows = []
    base_dates_m1 = pd.to_datetime(pd.date_range("2024-01-01", periods=60, freq="D"))
    base_dates_m2 = pd.to_datetime(pd.date_range("2024-01-01", periods=60, freq="D"))

    # Metric m1 - Region R1 - Slice S1 (with some anomalies)
    for i, date in enumerate(base_dates_m1):
        value = 100 + i*0.5 + np.random.normal(0, 5)
        if i == 20: value += 50 # Spike
        if i == 40: value -= 40 # Drop
        data_rows.append(["m1", "day", "Region", "R1", date, value])

    # Metric m1 - Region R1 - Slice S2 (stable)
    for i, date in enumerate(base_dates_m1):
        data_rows.append(["m1", "day", "Region", "R2", date, 70 + np.random.normal(0, 2)])

    # Metric m1 - Overall (dimensionless representation)
    for i, date in enumerate(base_dates_m1):
        data_rows.append(["m1", "day", "Overall", "Overall", date, 170 + i*0.2 + np.random.normal(0, 7)])


    # Metric m2 - Different behavior
    for i, date in enumerate(base_dates_m2):
        data_rows.append(["m2", "day", "Product", "P1", date, 200 - i*0.3 + np.random.normal(0,10)])
        if i == 30: data_rows[-1][5] = 50 # Sharp drop for m2 P1

    ledger_df_example = pd.DataFrame(data_rows, columns=["metric_id", "time_grain", "dimension", "slice", "date", "value"])
    
    print("--- Testing AnomalyAnalysis for m1 (day grain) ---")
    analysis_date_str = datetime.now().strftime("%Y-%m-%d")
    evaluation_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    analysis_result_m1 = run_anomaly_analysis(
        ledger_df=ledger_df_example,
        metric_id="m1",
        grain="day",
        analysis_date=analysis_date_str,
        evaluation_time=evaluation_time_str
    )

    import json
    print(json.dumps(analysis_result_m1, indent=2, default=str)) # Use default=str for np.nan

    # Test for a metric/grain with no data
    print("\n--- Testing AnomalyAnalysis for m3 (non-existent) ---")
    analysis_result_m3 = run_anomaly_analysis(
        ledger_df=ledger_df_example,
        metric_id="m3", # This metric does not exist
        grain="day",
        analysis_date=analysis_date_str,
        evaluation_time=evaluation_time_str
    )
    print(json.dumps(analysis_result_m3, indent=2, default=str))

    # Test for m2
    print("\n--- Testing AnomalyAnalysis for m2 (day grain) ---")
    analysis_result_m2 = run_anomaly_analysis(
        ledger_df=ledger_df_example,
        metric_id="m2",
        grain="day",
        analysis_date=analysis_date_str,
        evaluation_time=evaluation_time_str
    )
    print(json.dumps(analysis_result_m2, indent=2, default=str))