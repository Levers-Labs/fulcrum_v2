"""
ComparativeAnalysis Pattern

Purpose:
--------
This Pattern focuses on identifying whether a given day-grain metric Granger-causes
other metrics within the same dataset. It also estimates the correlation at the
best Granger-causality lag. Only the metric-pairs that exhibit a significant
Granger causality (p < 0.05) are included in the final output.

Input Format:
-------------
1) A pandas DataFrame ("ledger_df") containing daily metric values for at least the
   past 6 months. It must have the following columns:
      - 'metric_name': str
      - 'date': a date/datetime column
      - 'dimension': str (e.g., "Total" or other dimension slice)
      - 'value': numeric value
   We only care about rows where dimension == "Total" for this Pattern.

2) The specific 'metricId' (string) for which we want to investigate if it
   Granger-causes other metrics.

3) (Optional) Additional parameters like how far back we look (default ~6 months),
   or the maximum lag to test for Granger causality (default 14 days), etc.

Output Format (JSON structure):
-------------------------------
{
  "schemaVersion": "1.0.0",
  "patternName": "ComparativeAnalysis",
  "analysisDate": "YYYY-MM-DD",            # date of analysis
  "evaluationTime": "YYYY-MM-DD HH:mm:ss", # precise timestamp
  "grain": "day",
  "metricId": "<the input metricId>",
  "pairs": [
    {
      "comparisonMetric": "<some_other_metric_name>",
      "correlationCoefficient": <float>,
      "lagDaysForMaxCorrelation": <integer>
    },
    ...
  ]
}

Implementation Details:
-----------------------
1. Filter ledger_df to only include:
   - dimension == "Total"
   - date >= (today - 6 months)  [or some cutoff]
   - relevant metrics (the target metric, plus all others)

2. Pivot/merge the data so that each metric is a separate column, with a shared date index.

3. For each other metric (besides metricId), run the Granger causality test:
   - We use the `perform_granger_causality_test` function from the
     `intelligence_engine.primitives.ComparativeAnalysis` module. We do NOT
     rewrite that function; we just reference it.
   - In the context of this Pattern, we set:
       target = other_metric
       exog   = metricId
     This way, we check if "metricId" helps predict (Granger-cause) the other metric.

4. The `perform_granger_causality_test` function returns a DataFrame of p-values
   per lag. We pick the lag with the smallest p-value. If that p-value < 0.05,
   we consider the pair "significant".

5. The "lagDaysForMaxCorrelation" and "correlationCoefficient" are derived
   by shifting the “metricId” column by that best lag, then computing the
   Pearson correlation with the other metric.

6. We collect all significantly-caused metrics into the "pairs" list. Each entry
   contains:
   - "comparisonMetric" (the other metric name)
   - "correlationCoefficient" (float)
   - "lagDaysForMaxCorrelation" (integer lag that produced the best p-value)

7. Return a Python dict that matches the final JSON schema shown above. This
   dict can then be serialized to JSON for storage or downstream usage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# We reference the existing primitive:
# from intelligence_engine.primitives.ComparativeAnalysis import perform_granger_causality_test

def run_comparative_analysis_pattern(
    ledger_df: pd.DataFrame,
    metricId: str,
    lookback_days: int = 180,
    max_lag: int = 14
) -> dict:
    """
    Execute the ComparativeAnalysis pattern for 'metricId' on daily data.

    Parameters
    ----------
    ledger_df : pd.DataFrame
        Columns: ['metric_name','date','dimension','value'] (plus anything else).
        Contains daily metric records for at least the last 6 months (default).
    metricId : str
        The metric whose potential Granger-causality effect on other metrics we want to analyze.
    lookback_days : int, optional
        Number of days to look back from the current date. Default=180 (~6 months).
    max_lag : int, optional
        Maximum lag (in days) to test for Granger causality. Default=14.

    Returns
    -------
    dict
        Conforming to the JSON structure described in the docstring.
    """
    # 1) Filter to day-grain "Total" dimension within the last `lookback_days`.
    cutoff_date = pd.to_datetime("today").normalize() - pd.Timedelta(days=lookback_days)
    mask = (
        (ledger_df["dimension"] == "Total") &
        (ledger_df["date"] >= cutoff_date)
    )
    df_filtered = ledger_df.loc[mask].copy()

    # 2) Pivot so each metric is a separate column
    #    We'll pivot on metric_name -> columns, date -> index, values -> 'value'.
    #    Make sure to drop duplicates or handle pivot conflicts if needed.
    pivot_df = (
        df_filtered
        .pivot_table(index="date", columns="metric_name", values="value", aggfunc="mean")  # typically sum/mean is moot for "Total"
    )
    pivot_df.sort_index(inplace=True)

    # Ensure our 'metricId' is present in pivot
    if metricId not in pivot_df.columns:
        # If not present, return an empty result
        return {
            "schemaVersion": "1.0.0",
            "patternName": "ComparativeAnalysis",
            "analysisDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "evaluationTime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "grain": "day",
            "metricId": metricId,
            "pairs": []
        }

    # 3) For each other metric, run the Granger test => pick best lag => compute correlation
    #    We'll store results in a list
    pairs_list = []
    all_metric_columns = [m for m in pivot_df.columns if m != metricId]

    for comparison_metric in all_metric_columns:
        # We want to see if metricId Granger-causes comparison_metric
        sub_df = pivot_df[[metricId, comparison_metric]].dropna()
        if len(sub_df) < (max_lag + 5):
            # Not enough data points to run the test reliably
            continue

        try:
            # -- Step A: run the Granger test up to max_lag
            #    We'll reuse "perform_granger_causality_test(df, target, exog, maxlag, verbose=False)"
            from intelligence_engine.primitives.ComparativeAnalysis import perform_granger_causality_test

            gc_results = perform_granger_causality_test(
                df=sub_df.rename(columns={
                    metricId: "exog_col",
                    comparison_metric: "target_col"
                }),
                target="target_col",
                exog="exog_col",
                maxlag=max_lag,
                verbose=False
            )
            if gc_results.empty:
                continue

            # -- Step B: find the row with the lowest p-value
            gc_results_sorted = gc_results.sort_values("p_value", ascending=True).reset_index(drop=True)
            best_row = gc_results_sorted.iloc[0]
            best_pval = best_row["p_value"]
            best_lag = int(best_row["lag"])

            # If p_value < 0.05 => significant
            if (not pd.isna(best_pval)) and (best_pval < 0.05):
                # -- Step C: compute correlation at that best lag
                # Shift the 'exog_col' (our metricId) by `best_lag` days
                corr_df = sub_df.copy()
                corr_df["exog_shifted"] = corr_df["exog_col"].shift(best_lag)
                # Drop any NA introduced by shifting
                corr_df.dropna(inplace=True)
                # Pearson correlation
                corr_val = corr_df["exog_shifted"].corr(corr_df["target_col"])

                pair_entry = {
                    "comparisonMetric": comparison_metric,
                    "correlationCoefficient": float(corr_val) if corr_val is not None else 0.0,
                    "lagDaysForMaxCorrelation": best_lag
                }
                pairs_list.append(pair_entry)

        except Exception as ex:
            # We could log or handle errors, but for now we skip
            continue

    # 4) Build the final JSON-like dict
    result_dict = {
        "schemaVersion": "1.0.0",
        "patternName": "ComparativeAnalysis",
        "analysisDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "evaluationTime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grain": "day",
        "metricId": metricId,
        "pairs": pairs_list
    }

    return result_dict
