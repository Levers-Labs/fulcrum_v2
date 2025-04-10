import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the relevant primitive
from intelligence_engine.primitives.ComparativeAnalysis import perform_granger_causality_test

def run_comparative_analysis_pattern(
    ledger_df: pd.DataFrame,
    metricId: str,
    lookback_days: int = 180,
    max_lag: int = 14
) -> dict:
    """
    Executes the ComparativeAnalysis pattern to test whether `metricId` Granger-causes 
    other metrics on a daily grain and calculates the Pearson correlation at the best lag.

    Parameters
    ----------
    ledger_df : pd.DataFrame
        Must contain the following columns:
            - 'metric_name': str
            - 'date': date/datetime
            - 'dimension': str
            - 'value': numeric
        Only rows with dimension == 'Total' are considered. Data should cover ~6 months (or as given by lookback_days).
    metricId : str
        The metric whose potential Granger-causality effect on other metrics is tested.
    lookback_days : int, optional
        Lookback window in days (defaults to 180).
    max_lag : int, optional
        Maximum daily lag to test (defaults to 14).

    Returns
    -------
    dict
        {
          "schemaVersion": "1.0.0",
          "patternName": "ComparativeAnalysis",
          "analysisDate": "<YYYY-MM-DD>",
          "evaluationTime": "<YYYY-MM-DD HH:mm:ss>",
          "grain": "day",
          "metricId": "<the input metricId>",
          "pairs": [
             {
               "comparisonMetric": "<other_metric>",
               "correlationCoefficient": float,
               "lagDaysForMaxCorrelation": int
             },
             ...
          ]
        }
    """

    # Define cutoff for time filtering
    cutoff_date = pd.to_datetime("today").normalize() - pd.Timedelta(days=lookback_days)

    # Filter to dimension='Total' and within the lookback period
    df_filtered = ledger_df.loc[
        (ledger_df["dimension"] == "Total") &
        (ledger_df["date"] >= cutoff_date)
    ].copy()

    # Pivot to get one column per metric, indexed by date
    pivot_df = (
        df_filtered
        .pivot_table(index="date", columns="metric_name", values="value", aggfunc="mean")
        .sort_index()
    )

    # If the main metric isn't in the data, return an empty result
    if metricId not in pivot_df.columns:
        return {
            "schemaVersion": "1.0.0",
            "patternName": "ComparativeAnalysis",
            "analysisDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "evaluationTime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "grain": "day",
            "metricId": metricId,
            "pairs": []
        }

    pairs_list = []
    all_metrics = [m for m in pivot_df.columns if m != metricId]

    for comparison_metric in all_metrics:
        sub_df = pivot_df[[metricId, comparison_metric]].dropna()

        # Need enough observations to be meaningful at max_lag
        if len(sub_df) < (max_lag + 5):
            continue

        try:
            # Rename columns for compatibility with perform_granger_causality_test
            renamed_df = sub_df.rename(columns={
                metricId: "exog_col",
                comparison_metric: "target_col"
            })

            # Perform Granger causality test: 
            # if p-value < 0.05 at some lag, consider it 'significant'
            gc_results = perform_granger_causality_test(
                df=renamed_df,
                target="target_col",
                exog="exog_col",
                maxlag=max_lag,
                verbose=False
            )
            if gc_results.empty:
                continue

            # Sort by p-value ascending
            gc_sorted = gc_results.sort_values("p_value", ascending=True).reset_index(drop=True)
            best_row = gc_sorted.iloc[0]
            best_pval = best_row["p_value"]
            best_lag = int(best_row["lag"])

            # If significant, shift exog_col by best_lag and compute correlation
            if pd.notna(best_pval) and (best_pval < 0.05):
                corr_df = sub_df.copy()
                corr_df["exog_shifted"] = corr_df[metricId].shift(best_lag)
                corr_df.dropna(inplace=True)
                corr_val = corr_df["exog_shifted"].corr(corr_df[comparison_metric])

                pairs_list.append({
                    "comparisonMetric": comparison_metric,
                    "correlationCoefficient": float(corr_val) if corr_val is not None else 0.0,
                    "lagDaysForMaxCorrelation": best_lag
                })
        except Exception:
            # If a failure occurs, skip this comparison
            continue

    return {
        "schemaVersion": "1.0.0",
        "patternName": "ComparativeAnalysis",
        "analysisDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "evaluationTime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grain": "day",
        "metricId": metricId,
        "pairs": pairs_list
    }
