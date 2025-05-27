# comparative_analysis.py
"""
Pattern: ComparativeAnalysis
Version: 1.0
Author: Levers Labs Engineering

Purpose:
  Identifies if a given daily metric (metricId) Granger-causes other daily
  metrics from the 'Total' dimension. Estimates Pearson correlation at the
  best (most significant) Granger causality lag. Only pairs with
  Granger p-value < 0.05 are included.

Input Format:
  ledger_df (pd.DataFrame): DataFrame with daily metric data. Required columns:
    - 'metric_name' (str): Metric identifier.
    - 'date' (datetime-like): Date of the metric value.
    - 'dimension' (str): Dimension name; pattern filters for "Total".
    - 'value' (float): Numeric metric value.
    Data should span at least `lookback_days`.
  metricId (str): The primary metric to test as a potential cause.
  lookback_days (int): How many days of history to consider. Default: 180.
  max_lag (int): Max lag in days for Granger causality. Default: 14.

Output Format (JSON-serializable dict):
{
  "schemaVersion": "1.0.0",
  "patternName": "ComparativeAnalysis",
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",
  "grain": "day",
  "metricId": "<input metricId>",
  "pairs": [
    {
      "comparisonMetric": "<other_metric_name>",
      "correlationCoefficient": "float", // Can be 0.0 if not computable
      "lagDaysForMaxCorrelation": "int"   // Lag where p-value was lowest
    },
    // ... more significantly related pairs
  ]
}
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# This pattern relies on a primitive for Granger causality testing.
# The import path reflects the structure provided in the prompt.
from intelligence_engine.primitives.ComparativeAnalysis import perform_granger_causality_test

def run_comparative_analysis_pattern(
    ledger_df: pd.DataFrame,
    metricId: str,
    lookback_days: int = 180,
    max_lag: int = 14
) -> dict:
    """
    Executes the ComparativeAnalysis pattern to find metrics Granger-caused by metricId.
    """

    # 1. Filter ledger_df for "Total" dimension and date range
    # Ensure 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(ledger_df['date']):
        ledger_df['date'] = pd.to_datetime(ledger_df['date'])

    cutoff_date = pd.to_datetime(datetime.now().date()) - pd.Timedelta(days=lookback_days)
    
    mask = (
        (ledger_df["dimension"] == "Total") &
        (ledger_df["date"] >= cutoff_date)
    )
    df_filtered = ledger_df.loc[mask].copy()

    if df_filtered.empty:
        # Not enough data or "Total" dimension missing
        return {
            "schemaVersion": "1.0.0",
            "patternName": "ComparativeAnalysis",
            "analysisDate": datetime.now().strftime("%Y-%m-%d"),
            "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "grain": "day",
            "metricId": metricId,
            "pairs": []
        }

    # 2. Pivot data: each metric as a column, date as index
    try:
        pivot_df = df_filtered.pivot_table(
            index="date",
            columns="metric_name",
            values="value",
            aggfunc="mean" # Use mean for robustness, though sum/first often same for "Total"
        )
    except Exception as e:
        # Handle cases where pivot might fail (e.g., duplicate entries not intended for mean)
        print(f"Error pivoting data: {e}. Check for duplicate metric_name/date entries for dimension 'Total'.")
        return {
            "schemaVersion": "1.0.0",
            "patternName": "ComparativeAnalysis",
            "analysisDate": datetime.now().strftime("%Y-%m-%d"),
            "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "grain": "day",
            "metricId": metricId,
            "pairs": [{"error": "Data pivoting failed."}] # Or empty pairs
        }

    pivot_df.sort_index(inplace=True)

    # Ensure metricId is in the pivoted data
    if metricId not in pivot_df.columns:
        return {
            "schemaVersion": "1.0.0",
            "patternName": "ComparativeAnalysis",
            "analysisDate": datetime.now().strftime("%Y-%m-%d"),
            "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "grain": "day",
            "metricId": metricId,
            "pairs": [] # metricId not found after filtering and pivoting
        }

    pairs_list = []
    # Iterate through other metrics to test against metricId
    all_other_metrics = [m for m in pivot_df.columns if m != metricId]

    for comparison_metric in all_other_metrics:
        # Prepare data for Granger causality: metricId (exog) vs comparison_metric (target)
        # Ensure both series have values for the same dates by dropping NaNs after selection
        sub_df = pivot_df[[metricId, comparison_metric]].dropna()

        # Check if sufficient data remains after aligning dates
        if len(sub_df) < (max_lag + 10): # Added a slightly larger buffer
            continue

        try:
            # The primitive perform_granger_causality_test expects specific column names
            # or operates on series. Assuming it takes a DataFrame and column names:
            renamed_sub_df = sub_df.rename(columns={
                metricId: "exog_col",          # metricId is the potential cause
                comparison_metric: "target_col" # comparison_metric is the potential effect
            })

            gc_results_df = perform_granger_causality_test(
                df=renamed_sub_df,
                target="target_col", # We are testing if exog_col -> target_col
                exog="exog_col",
                maxlag=max_lag,
                verbose=False # Keep primitive non-verbose
            )

            if gc_results_df.empty:
                continue

            # Find the lag with the minimum p-value
            # Primitive is expected to return columns like 'lag' and 'p_value'
            best_result_row = gc_results_df.loc[gc_results_df['p_value'].idxmin()]
            best_pval = best_result_row["p_value"]
            best_lag = int(best_result_row["lag"]) # Ensure lag is integer

            # If p-value is significant, calculate correlation at this best lag
            if pd.notna(best_pval) and best_pval < 0.05:
                # Calculate correlation on the original sub_df (with original column names)
                # Shift the metricId series by the best_lag
                # A positive lag means metricId's past values influence comparison_metric's current values
                temp_corr_df = sub_df.copy()
                temp_corr_df[f"{metricId}_lagged"] = temp_corr_df[metricId].shift(best_lag)
                
                # Drop NaNs introduced by shifting before calculating correlation
                temp_corr_df.dropna(inplace=True)

                if not temp_corr_df.empty and len(temp_corr_df) >= 2: # Need at least 2 points for correlation
                    correlation_coefficient = temp_corr_df[f"{metricId}_lagged"].corr(temp_corr_df[comparison_metric])
                else:
                    correlation_coefficient = 0.0 # Fallback if too few points after lag

                pairs_list.append({
                    "comparisonMetric": comparison_metric,
                    "correlationCoefficient": float(correlation_coefficient) if pd.notna(correlation_coefficient) else 0.0,
                    "lagDaysForMaxCorrelation": best_lag
                })

        except Exception as e:
            # Log or handle specific errors from the primitive or calculations
            print(f"Could not perform comparative analysis for {metricId} vs {comparison_metric}: {e}")
            continue # Skip to the next comparison_metric

    # Construct the final output dictionary
    output_dict = {
        "schemaVersion": "1.0.0",
        "patternName": "ComparativeAnalysis",
        "analysisDate": datetime.now().strftime("%Y-%m-%d"),
        "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grain": "day", # This pattern is designed for daily grain
        "metricId": metricId,
        "pairs": pairs_list
    }

    return output_dict

if __name__ == '__main__':
    # Example Usage (requires the primitive to be available)
    # Mocking the primitive if not actually present for standalone testing
    try:
        from intelligence_engine.primitives.ComparativeAnalysis import perform_granger_causality_test
    except ImportError:
        print("Mocking perform_granger_causality_test primitive for example.")
        def perform_granger_causality_test(df, target, exog, maxlag, verbose=False):
            # Mock implementation: returns some plausible results
            results = []
            for lag in range(1, maxlag + 1):
                # Simulate p-values, making some significant
                p_val = np.random.uniform(0.01, 0.5)
                if lag % 3 == 0: p_val = np.random.uniform(0.01, 0.04) # Make some lags significant
                results.append({'lag': lag, 'p_value': p_val, 'f_value': np.random.rand()*10})
            return pd.DataFrame(results)
        # This mock needs to be accessible via the import path used in the pattern
        import sys
        import os
        # Create dummy structure for the mock to be importable
        if not os.path.exists("intelligence_engine/primitives"):
            os.makedirs("intelligence_engine/primitives")
        with open("intelligence_engine/primitives/__init__.py", "w") as f: f.write("")
        with open("intelligence_engine/primitives/ComparativeAnalysis.py", "w") as f:
            f.write("import pandas as pd\nimport numpy as np\n") # Add necessary imports for the mock
            f.write(f"\n{perform_granger_causality_test.__name__} = {perform_granger_causality_test.__code__}\n")


    # Sample Ledger Data
    num_days = 200
    dates = pd.to_datetime([datetime.now().date() - timedelta(days=i) for i in range(num_days)])
    dates = sorted(dates)

    data = {
        'date': [],
        'metric_name': [],
        'dimension': [],
        'value': []
    }

    np.random.seed(42)
    metric_a_series = np.random.rand(num_days) * 100
    metric_b_series = metric_a_series * 0.5 + np.random.rand(num_days) * 50 # B somewhat depends on A
    metric_c_series = np.random.rand(num_days) * 100 # C is independent

    for i in range(num_days):
        data['date'].extend([dates[i]] * 3)
        data['metric_name'].extend(['MetricA', 'MetricB', 'MetricC'])
        data['dimension'].extend(['Total'] * 3)
        data['value'].extend([
            metric_a_series[i],
            metric_b_series[i] + (metric_a_series[i-3]*0.3 if i >=3 else 0), # Lagged effect for B
            metric_c_series[i]
        ])

    sample_ledger_df = pd.DataFrame(data)

    print("--- Testing ComparativeAnalysis for MetricA ---")
    result = run_comparative_analysis_pattern(
        ledger_df=sample_ledger_df,
        metricId='MetricA',
        lookback_days=180,
        max_lag=10
    )
    import json
    print(json.dumps(result, indent=2))

    print("\n--- Testing ComparativeAnalysis for MetricC (likely no significant pairs) ---")
    result_c = run_comparative_analysis_pattern(
        ledger_df=sample_ledger_df,
        metricId='MetricC',
        lookback_days=180,
        max_lag=10
    )
    print(json.dumps(result_c, indent=2))

    print("\n--- Testing with metricId not in data ---")
    result_missing = run_comparative_analysis_pattern(
        ledger_df=sample_ledger_df,
        metricId='MetricX', # Does not exist
        lookback_days=180,
        max_lag=10
    )
    print(json.dumps(result_missing, indent=2))

    # Clean up mock
    if 'intelligence_engine.primitives.ComparativeAnalysis' in sys.modules and "Mocking" in sys.modules['intelligence_engine.primitives.ComparativeAnalysis'].__file__:
        del sys.modules['intelligence_engine.primitives.ComparativeAnalysis']
        del sys.modules['intelligence_engine.primitives']
        del sys.modules['intelligence_engine']
        if os.path.exists("intelligence_engine/primitives/ComparativeAnalysis.py"): os.remove("intelligence_engine/primitives/ComparativeAnalysis.py")
        if os.path.exists("intelligence_engine/primitives/__init__.py"): os.remove("intelligence_engine/primitives/__init__.py")
        if os.path.exists("intelligence_engine/primitives"): os.rmdir("intelligence_engine/primitives")
        if os.path.exists("intelligence_engine"): os.rmdir("intelligence_engine")