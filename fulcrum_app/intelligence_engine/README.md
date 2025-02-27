# Patterns

This section covers **what** each Pattern does, **why** we do it, **which Primitives** it calls, **inputs/outputs** (including how we compose the final `PatternOutput`), and any additional best-practice notes.

---

## Overview of Pattern Architecture

Each **Pattern** is a Python class with:

- A `PATTERN_NAME` and `PATTERN_VERSION`.
- A single main `run(...)` method that:
    1. Accepts relevant inputs (like `data: pd.DataFrame`, `analysis_window`, parameters).
    2. Calls **Primitives** from our `fulcrum_app/intelligence_engine/primitives/` modules.
    3. Assembles the results into a `PatternOutput` object, containing `pattern_name`, `pattern_version`, `metric_id`, `analysis_window`, and a final `results` dictionary.

**Key**: This ensures each Pattern is reusable, consistent, and easily tested. The “Storytelling Engine” or any other consumer can parse the final `PatternOutput` to generate textual or visual stories.

---

## Pattern Summaries

Below is a table summarizing each pattern at a high level:

| **Pattern** | **Purpose** | **Key Primitives** | **Output** |
| --- | --- | --- | --- |
| **PerformanceStatusPattern** | Classifies the metric as on/off-track vs. final target | `classify_metric_status` (performance.py) | Status label `"on_track"/"off_track"/"no_target"/"no_data"`, plus final value/target |
| **HistoricalPerformancePattern** | Summarizes historical data (growth, rolling averages, stats) | `calculate_pop_growth`, `calculate_rolling_averages`, `calculate_descriptive_stats` | Growth series, rolling averages, descriptive stats. |
| **DimensionAnalysisPattern** | Aggregates a metric by dimension slices, computing shares & top slices | `calculate_slice_metrics`, `compute_slice_shares`, `rank_metric_slices` | A breakdown of each slice’s aggregated_value & share, plus the top N slices. |
| **MetricDetailsPattern** | Fetches metadata (definition, owners, disclaimers) — minimal numeric work | No heavy numeric primitives; just references config or DB | Returns a dictionary of metadata fields: `definition`, `owner_team`, `targets`, `disclaimers`, etc. |
| **DataQualityPattern** | Checks data completeness, outlier counts, suspicious volatility | `calculate_descriptive_stats`, `detect_anomaly_with_variance` | Basic stats, anomaly_count, data_completeness (# of rows). |
| **MetricGraphPattern** | Explores the upstream/downstream relationships in a metric graph | GraphService calls (e.g. `get_upstream_metrics`) | A list of upstream/downstream metric IDs for lineage or dependency. |
| **RootCausePattern** | Explains T0→T1 change via dimension breakdown, driver-based attribution | `analyze_dimension_impact`, `influence_attribution`, etc. | Summaries of dimension-level deltas, driver-based partial effects, plus any residual. |
| **ForecastPattern** | Generates short- or long-range forecasts (ARIMA, etc.) | `simple_forecast` (forecasting.py) | A time-series of future forecasted points, plus method metadata. |
| **LeveragePattern** | Identifies which drivers produce the best “bang for buck” | `rank_drivers_by_leverage` (leverage_scenario.py) | A list of drivers sorted by ROI or sensitivity/cost ratio. |
| **ImpactAnalysisPattern** | Evaluates how a change in Metric A influenced B, C, etc. (downstream) | Possibly `GraphService` or partial driver approach | Currently a stub returning `downstream_impact`, `events`. |
| **GoalSettingPattern** | Finds required growth rate or driver shift to hit a future target | `calculate_required_growth` (performance.py) or driver approach | Returns the needed compound growth rate, or partial driver shift. |
| **ComparativeAnalysisPattern** | Compares correlations, synergy among multiple metrics | `compare_metrics_correlation` (comparative_analysis.py) | A correlation matrix (or synergy info) in dictionary form. |
| **EventAnalysisPattern** | Maps external events to metric changes (pre/post analysis) | `quantify_event_impact` (root_cause.py) | A list of events with approximate “impact.” |
| **AnomalyAnalysisPattern** | A one-stop shop for detecting outliers via variance, SPC, ML | `detect_anomalies` (which calls `detect_anomaly_with_variance`, `detect_spc_anomalies`, `detect_anomaly_ml`) | # of anomalies, plus a record of each flagged row. |
| **CohortAnalysisPattern** | Tracks user or entity retention over time-based cohorts | `perform_cohort_analysis` (cohort_analysis.py) | A pivot table of [cohort_label x period_index], showing count/sum usage. |

---

### **PerformanceStatusPattern**

1. **Purpose**: Evaluate the metric’s final row vs. a target, classify “on_track” or “off_track.”
2. **Inputs**:
    - `data` with columns `[date, value, target]` (target can be absent or NaN).
    - `analysis_window`.
3. **Calls**: `classify_metric_status(...)` from `performance.py`.
4. **Output**:
    - `results["status"]` = “on_track”/“off_track”/“no_target”/“no_data”.
    - `results["final_value"]`, `results["final_target"]`, `results["threshold"]`.

### **HistoricalPerformancePattern**

1. **Purpose**: Summarize historical data with PoP growth, rolling averages, descriptive stats.
2. **Inputs**: `data` with `[date, value]`.
3. **Calls**:
    - `calculate_pop_growth(...)`
    - `calculate_rolling_averages(...)`
    - `calculate_descriptive_stats(...)`
4. **Output**:
    - Growth series list, rolling average list, dictionary of summary stats (min, max, mean, median, etc.).

### **DimensionAnalysisPattern**

1. **Purpose**: Aggregates metric by dimension slices, computing top slices, share distribution.
2. **Inputs**: `data` with `[slice_col, value_col]`.
3. **Calls**:
    - `calculate_slice_metrics(...)`
    - `compute_slice_shares(...)`
    - `rank_metric_slices(...)`
4. **Output**:
    - A dictionary with three main arrays: `[slice_metrics]`, `[slice_shares]`, `[top_slices]`.

### **MetricDetailsPattern**

1. **Purpose**: Surfaces non-numeric metadata: definition, owners, disclaimers.
2. **Inputs**: A `metadata` dict (often from a config store).
3. **Calls**: *No heavy numeric primitives—just references.*
4. **Output**:
    - `definition`, `owner_team`, `targets`, `disclaimers`.

### **DataQualityPattern**

1. **Purpose**: Evaluate the reliability of the data, checking outliers or missing coverage.
2. **Inputs**: `data` with `[date, value]`.
3. **Calls**:
    - `calculate_descriptive_stats(...)`
    - `detect_anomaly_with_variance(...)`
4. **Output**:
    - Summaries: `descriptive_stats`, `anomaly_count`, and `data_completeness` (# of rows).

### **MetricGraphPattern**

1. **Purpose**: Explore the metric’s upstream/downstream in the global metric graph.
2. **Inputs**: Typically a `graph_service` with a method like `get_upstream_metrics(metric_id)`.
3. **Calls**: A `GraphService` method, not a direct primitive.
4. **Output**:
    - `results["upstream_metrics"]`, `results["downstream_metrics"]`.

### **RootCausePattern**

1. **Purpose**: Explains T0→T1 change with dimension-level or driver-based decomposition.
2. **Inputs**: `data_t0`, `data_t1`, optional driver model & driver data.
3. **Calls**:
    - `analyze_dimension_impact(...)`
    - `influence_attribution(...)`
4. **Output**:
    - `dimension_impact`: a list of slice deltas,
    - `driver_attribution`: partial effect from each driver,
    - possibly `seasonality_effect` or `residual`.

### **ForecastPattern**

1. **Purpose**: Generate short/long-range forecast.
2. **Inputs**: `data` with `[date, value]`, plus method (e.g., “ses”, “holtwinters”).
3. **Calls**: `simple_forecast(...)` from `forecasting.py`.
4. **Output**:
    - A list of future points `[{"date":..., "forecast":...}, ...]`.

### **LeveragePattern**

1. **Purpose**: Identifies driver ROI or “bang for buck.”
2. **Inputs**: A fitted driver model, a cost map for each driver.
3. **Calls**:
    - `rank_drivers_by_leverage(...)` from `leverage_scenario.py`.
4. **Output**:
    - A sorted list of drivers with `coefficient, cost_per_unit, roi`.

### **ImpactAnalysisPattern**

1. **Purpose**: Inverse of Root Cause—if Metric A changed, who felt it?
2. **Inputs**: Possibly `data`, `event_data`, or a graph approach.
3. **Calls**: Usually a partial approach or graph-based approach.
4. **Output**:
    - `downstream_impact` array, `events` array. (Currently a stub in this codebase.)

### **GoalSettingPattern**

1. **Purpose**: Compute the required compound growth or driver change to meet a future target.
2. **Inputs**: `current_value`, `target_value`, `periods_left`.
3. **Calls**: `calculate_required_growth(...)` from `performance.py`.
4. **Output**:
    - `needed_growth_rate` plus summary of `current_value`, `target_value`, `periods_left`.

### **ComparativeAnalysisPattern**

1. **Purpose**: Compare multiple metrics for correlation, synergy, or predictive significance.
2. **Inputs**: `df` with columns from the `metrics` list.
3. **Calls**: `compare_metrics_correlation(...)` from `comparative_analysis.py`.
4. **Output**:
    - `results["correlation_matrix"]` in a dictionary form.

### **EventAnalysisPattern**

1. **Purpose**: Evaluate external events and approximate their impact on the metric.
2. **Inputs**: `data` with `[date, value]`, plus `events` with `[event_date]`.
3. **Calls**: `quantify_event_impact(...)`.
4. **Output**:
    - A list of events with before/after “impact.”

### **AnomalyAnalysisPattern**

1. **Purpose**: A single pattern that calls multiple anomaly detection methods (variance, SPC, ML).
2. **Inputs**: `data` with `[date, value]`.
3. **Calls**: `detect_anomalies(...)` from `trend.py`.
4. **Output**:
    - `results["num_anomalies"]`, plus a list of anomaly rows.

### **CohortAnalysisPattern**

1. **Purpose**: Build a pivot table of usage or retention across cohorts.
2. **Inputs**: Typically `[user_id, signup_date, activity_date, measure_col]`.
3. **Calls**: `perform_cohort_analysis(...)`.
4. **Output**:
    - A pivot table stored in dictionary form (`.to_dict("split")`).

---

## Common Pattern I/O Conventions

**Pattern** classes share the same structure:

- **Name/Version**: `PATTERN_NAME` string, `PATTERN_VERSION` string.
- **`run(...)`** method:
    - **Inputs**:
        1. **`metric_id: str`** – identifies the metric we’re analyzing.
        2. **`data: pd.DataFrame`** – the numeric data or relevant slice data. Some patterns also require `driver_model`, `events`, or dimension columns.
        3. **`analysis_window: Dict[str, str]`** – typically `{"start_date":..., "end_date":...}`.
        4. **Pattern-Specific** parameters (e.g. `threshold=0.05`, `window_size=7`).
    - **Process**: The pattern calls **Primitives** or references a service (like `GraphService`).
    - **Output**: A `PatternOutput` instance with:
        - `pattern_name`, `pattern_version`, `metric_id`, `analysis_window`, `results` (a dictionary of final analysis).

**Why** we do it this way:

- Ensures each pattern is easily tested or replaced.
- The final `PatternOutput` is machine-readable and consistent, letting the **Storytelling Engine** produce narratives without re-running computations.

# Primitives

This section details each set of **Primitives**. We list them by domain, describing **what** they do, **why** they exist, **inputs/outputs**, and any special notes. Where relevant, we show each function’s **short purpose** and **inputs/outputs** in a table.

---

## Performance Primitives

File: **`performance.py`**

**Why**: These handle goal-vs-actual computations, status classification, and minimal “growth to target” logic.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `calculate_metric_gva` | Compute Goal vs. Actual difference at a single time point. | `(actual, target, allow_negative_target=False)` | `{'abs_diff': float, 'pct_diff': float or None}` |
| `calculate_historical_gva` | Merge two time-series (actual vs. target), compute daily or row-wise GvA. | `df_actual, df_target, date_col='date', value_col='value'` etc. | Returns a DF with columns `['date', 'abs_gva', 'pct_gva']`. |
| `classify_metric_status` | Assign on_track/off_track if `value >= (1-threshold)*target`. | `row_val, row_target, threshold=0.05, allow_negative_target=False` | Returns a string status: `"on_track"`, `"off_track"`, or `"no_target"`. |
| `detect_status_changes` | Identify rows where status flips from one to another. | A DF with `status_col`; we add `status_flip=True/False`. | Returns an augmented DF with `prev_status` and `status_flip` columns. |
| `track_status_durations` | Calculate consecutive runs of the same status. | A DF with `status_col` (optionally a date_col). | Returns a DF of `[status, start_idx, end_idx, run_length, (dates if needed)]`. |
| `monitor_threshold_proximity` | Check if a metric is near flipping from on/off-track by margin. | `(val, target, margin=0.05, allow_negative_target=False)` | Returns a boolean. |
| `calculate_required_growth` | Solve for needed compound growth rate to reach future target in `periods_left`. | `(current_value, target_value, periods_left, allow_negative=False)` | Returns a float growth rate (e.g. 0.02 => 2%). If impossible, returns 0 or None. |

---

## Time Series Growth Primitives

File: **`time_series_growth.py`**

**Why**: Best for capturing incremental changes, partial-to-date comparisons, and trending logic over an entire time series.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `calculate_pop_growth` | Calculate row-by-row period-over-period % growth. | `df, value_col='value', sort_by_date=None, fill_method=None` | Returns a DF with a new `pop_growth` column in `%`, ignoring rows where the previous value is 0 or NaN. |
| `calculate_to_date_growth_rates` | Compares partial-to-date vs. a prior partial window for MTD, WTD, etc. | `df, date_col='date', value_col='value', freq='M' or 'W' approach, aggregator=...` | Typically returns some pivot or daily cumsum approach. The code might be more custom depending on your partial alignment. |
| `calculate_average_growth` | Derives a single “average growth rate” across entire DF. Could be arithmetic, geometric, or regression-based. | `df, value_col='value', method='arithmetic'` | Returns a float growth rate. For example, 0.05 => 5%. |
| `calculate_rolling_averages` | Generates a rolling average column for smoothing. | `df, value_col='value', window=7, min_periods=None, center=False, new_col_name=None` | Returns a DF with an added column, e.g. `"rolling_avg_{window}"`. |
| `calculate_slope_of_time_series` | A linear regression slope over the entire DF, or using date_col for x if provided. | `df, value_col='value', date_col=None` | Returns a float slope. If date_col is used, slope is “value per day.” |
| `calculate_cumulative_growth` | Summation or product approach for cumulative or indexed growth. | `df, value_col='value', sort_by_date=None, method='sum | product |

---

## Dimension Analysis Primitives

File: **`dimension_analysis.py`**

**Why**: Dimension analysis is crucial for slicing a metric by categories (region, product line), detecting segment anomalies, and measuring concentration risk.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `calculate_slice_metrics` | Groups DF by `slice_col` and aggregates `value_col` by sum, mean, etc. | `(df, slice_col, value_col, agg='sum', top_n=None, other_label='Other', dropna_slices=True)` | Returns a DF `[slice_col, aggregated_value]`, optionally combining slices beyond top_n into “Other.” |
| `compute_slice_shares` | Adds a `share_pct` = slice_value / total_value * 100. | `(agg_df, slice_col, val_col='aggregated_value')` | Returns a DF with a `share_pct` column. |
| `rank_metric_slices` | Sorts slices by their aggregated_value, returning top or bottom N. | `(agg_df, val_col='aggregated_value', top_n=5, ascending=False)` | Returns the top (or bottom) slices in a new DF. |
| `analyze_composition_changes` | Compare T0 vs. T1 dimension shares. | `(df_t0, df_t1, slice_col='segment', val_col='aggregated_value')` | Merges T0/T1, computes `share_diff = share_pct_t1 - share_pct_t0`. |
| `detect_anomalies_in_slices` | For each slice, check if it’s out of range historically (z-score approach, etc.). | `(df, slice_col, value_col, date_col=None, z_thresh=3.0)` | Typically returns a new DF or some anomaly labeling. |
| `compare_dimension_slices_over_time` | Provides a side-by-side or pivot view of T0 vs. T1 by slice. | `(df, slice_col, date_col='date', value_col='value', t0, t1, agg='sum')` | Produces a DF `[slice, val_t0, val_t1, abs_diff, pct_diff]`. |
| `calculate_concentration_index` | Computes Herfindahl-Hirschman (HHI) or Gini if desired, measuring how concentrated the distribution is. | `(df, val_col='aggregated_value', method='HHI' or 'gini')` | Returns a float index. HHI => [1/N..1], Gini => [0..1]. |

---

## Root Cause Primitives

File: **`root_cause.py`**

**Why**: This suite handles “why did the metric change?” from dimension, driver, or event/seasonality angles.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `decompose_metric_change` | Splits an overall T0→T1 delta across multiple factors in an **additive** or **multiplicative** formula-based approach. | `(val_t0, val_t1, factors, relationship='additive' | 'multiplicative')` |
| `calculate_component_drift` | Evaluates how each operand in a formula changed from T0→T1. | `(df, formula, id_col='component', value_col_t0='value_t0', value_col_t1='value_t1')` | Returns a DF with `delta`, `partial_effect` columns for each component row. |
| `analyze_dimension_impact` | Summation of dimension-level changes from T0→T1 if metric= sum of slices. | `(df_t0, df_t1, slice_col='segment', value_col='value')` | Returns a DF `[slice_col, val_t0, val_t1, delta, pct_of_total_delta]`. |
| `get_regression_model` | Fits a simple regression linking driver metrics to output (may allow lasso/ridge, etc.). | `(X, y, model_type='linear', fit_intercept=True, **kwargs)` | Returns a fitted model with `.coef_` and `.intercept_`. |
| `influence_attribution` | Uses the model’s coefficients to attribute a T0→T1 change in y across driver changes. | `(model, X_t0, X_t1, y_change, driver_names)` | Returns a dict with `'estimated_delta'`, `'residual'`, `'drivers':{driver:{delta_x, partial_effect,pct_of_est, pct_of_act}}` |
| `influence_drift` | Compares two sets of driver coefficients from two different time windows. | `(model_t0, model_t1, driver_names)` | Returns a dict driver-> {coef_t0, coef_t1, delta_coef}. |
| `evaluate_seasonality_effect` | Runs an STL decomposition to see how much T0→T1 difference might be from seasonal components. | `(df, date_col='date', value_col='value', period=None)` | Returns a dict: `'seasonal_diff','total_diff','fraction_of_total_diff'`. |
| `quantify_event_impact` | Approximates how external events impacted the metric by comparing pre vs. post windows. | `(df, event_df, date_col='date', value_col='value', window_before=7, window_after=7)` | Returns a DF with each event’s `before_avg, after_avg, impact`. |

---

## Forecasting Primitives

File: **`forecasting.py`**

**Why**: Covers everything from naive forecast to auto_arima, merges with targets, handles intervals, calculates accuracy, and does advanced driver-based decomposition.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `simple_forecast` | A minimal interface for multiple forecast methods (`naive`, `ses`, `holtwinters`, `auto_arima`). | `(df, value_col='value', periods=7, method='ses', date_col=None, freq=None, **kwargs)` | Returns a DF with columns `[date, forecast]` for future periods. |
| `forecast_upstream_metrics` | For each driver (dict of driver_id->DF), calls `simple_forecast` and returns driver_id-> forecast DF. | `(drivers, periods=7, method='ses', date_col=None, freq=None, **kwargs)` | Returns a dict driver_id-> forecast DataFrame. |
| `forecast_metric_dimensions` | Forecast each dimension slice individually. | `(df, slice_col='segment', date_col='date', value_col='value', periods=7, method='ses', freq=None, **kwargs)` | Returns a dict slice_value-> forecast DF. |
| `forecast_best_and_worst_case` | Creates ±X% scenarios around an existing forecast DF. | `(forecast_df, buffer_pct=10.0, forecast_col='forecast')` | Returns the same DF plus columns `best_case` and `worst_case`. |
| `forecast_target_achievement` | Merges a forecast DF with a target DF, computing differences or on_track flags. | `(forecast_df, target_df, forecast_col='forecast', target_col='target', date_col='date')` | Returns a merged DF with `abs_diff, pct_diff, on_track`. |
| `calculate_forecast_accuracy` | Compares past forecast vs. actual for error metrics (RMSE, MAE, MAPE). | `(actual_df, forecast_df, date_col='date', actual_col='actual', forecast_col='forecast')` | Returns a dict `{'rmse':..., 'mae':..., 'mape':..., 'n':...}` |
| `assess_forecast_uncertainty` | Summarizes how wide the forecast intervals are if the DF includes `[lower, upper]` columns. | `(forecast_df, lower_col='lower', upper_col='upper', forecast_col='forecast')` | Returns e.g. `{'mean_interval_width':..., 'max_interval_width':..., 'mean_relative_width':...}` |
| `decompose_forecast_drivers` | If we have a driver-based forecast, show partial effect from each driver in the final forecast. | `(driver_forecasts, driver_coefs, base_intercept=0.0)` | Returns a DF of `[date, total_forecast, driver_X_contribution, ...]`. |

---

## Leverage Scenario Primitives

File: **`leverage_scenario.py`**

**Why**: This domain is crucial for scenario planning and resource allocation—**“Which driver do we tweak to maximize outcome for minimal cost?”**

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `calculate_driver_sensitivity` | For a linear model, sensitivity ~ coefficient. For non-linear, partial derivative at `current_point`. | `(model, current_point: dict)` | Returns a dict driver-> sensitivity float. |
| `simulate_driver_scenarios` | Re-run the model with certain driver shifts. If `relative=True`, scenario[driver]= fraction => `x_j_new= x_j*(1 + fraction)`. | `(model, current_point, scenario, relative=True)` | Returns a single new predicted y. |
| `backcalculate_driver_targets` | Solve how much a single driver must change to reach a target metric. For linear: delta_j = (target - intercept - sum_others - b_j*x_j)/ b_j. | `(model, current_point, target_y, driver_name, relative=False)` | Returns a float shift or fraction. |
| `evaluate_driver_adjustment_costs` | Multiplies scenario[driver] by `cost_map[driver]` to produce cost. | `(scenario, cost_map, relative=True)` | Returns dict driver-> cost_of_that_shift. |
| `rank_drivers_by_leverage` | Sort drivers by (coefficient / cost_per_unit). | `(model, current_point, cost_map)` | Returns a sorted DF of `[driver, coefficient, cost_per_unit, roi]`. |
| `analyze_cross_driver_effects` | If the model has interaction terms, check synergy or conflict. Typically for statsmodels with param names like `X1:X2`. | `(model, drivers: list)` | Returns a dict of interaction param-> coefficient. |
| `identify_improvement_headroom` | Compare current driver levels to `max_feasible` => headroom= max_feasible[drv]- current_value. | `(current_point, max_feasible)` | Returns dict driver-> headroom float. |
| `evaluate_implementation_constraints` | If scenario shift> constraints => clamp or skip. | `(scenario, constraints, relative=True)` | Returns a feasible scenario dict, possibly clamped. |
| `rank_improvement_opportunities` | Enumerates single-driver scenarios, calculates effect vs. cost => ROI, sorts descending. | `(model, current_point, cost_map, feasible_shifts, default_shift=0.1)` | Returns a DF `[driver, shift, new_y, delta_y, cost, roi]` sorted by ROI. |

---

## Comparative Analysis Primitives

File: **`comparative_analysis.py`**

**Why**: For cross-metric analysis, synergy detection, and significance testing.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `compare_metrics_correlation` | Computes correlation matrix among multiple columns in a DF. | `(df, metrics: list, method='pearson')` | Returns a correlation matrix as a DataFrame. |
| `detect_metric_predictive_significance` | Uses Granger causality to see if x_col leads y_col in a time-series sense. | `(df, x_col, y_col, max_lag=5, add_const=True)` | Returns a dict with p-values for each lag, plus best lag & best p_value. |
| `analyze_metric_interactions` | Fit a regression with interaction terms (like X1:X2) using patsy/statsmodels. | `(df, target_col, driver_cols, interaction_pairs)` | Returns `model_summary`, and each interaction term’s coefficient/p_value. |
| `benchmark_metrics_against_peers` | Merge two series (my metric vs. peer), compute ratio or difference. | `(df, df_peer, date_col='date', my_val_col='my_value', peer_val_col='peer_value', method='ratio' | 'difference')` |
| `detect_statistical_significance` | Generic function for t-tests or chi-square to see if two groups differ. | `(groupA, groupB, test_type='t-test', equal_var=True)` | Returns `{'test_stat':..., 'p_value':..., 'significant': bool, 'df':...}` |
| `test_cointegration` | Engle-Granger test to see if two series are cointegrated. | `(df, colA, colB)` | Returns `{'p_value':..., 'test_stat':..., 'critical_values':...}`. |
| `detect_lagged_influence` | Cross-correlation approach to find the lag at which colA best correlates with colB. | `(df, colA, colB, max_lag=10)` | Returns `{'best_lag':..., 'best_corr':..., 'all_lags': {...}}`. |

---

## Cohort Analysis Primitives

File: **`cohort_analysis.py`**

**Why**: Cohort analysis is crucial for retention/churn patterns or usage expansions over time.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `perform_cohort_analysis` | Groups data by a “cohort” (based on signup_date truncated to monthly/weekly) and then sees activity in subsequent periods | `(df, entity_id_col='user_id', signup_date_col='signup_date', activity_date_col='activity_date', time_grain='M', max_periods=12, measure_col=None, measure_method='count')` | Returns a pivot table with index=cohort_label, columns=period_index (0..max_periods), values= measure (count, sum, etc.). |
| *Helper: `compute_retention_rates`* | (Often separate) Divides each row in the pivot by its period 0 to get retention fraction | `(pivot_table: pd.DataFrame)` | Returns the same shape DF with each row normalized. |

---

## Trend & Anomaly Primitives

File: **`trend.py`**

**Why**: This suite is essential for identifying overall trend directions, breakpoints, performance plateaus, record extremes, and outliers or anomalies.

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `analyze_metric_trend` | Overall linear regression slope => up/down/stable classification. | `(df, value_col='value', date_col=None, slope_threshold=0.0)` | Returns `{ 'trend':'up' |
| `detect_trend_changes` | Rolling-window slope approach to see sign flips or large slope changes. | `(df, value_col='value', date_col=None, window_size=5, slope_change_threshold=0.0)` | Returns a DF with `slope`, `prev_slope`, `trend_change=True/False`. |
| `detect_new_trend_direction` | If the slope recently flipped from negative to positive => “new_upward,” or opposite => “new_downward.” | `(df, slope_col='slope')` | Returns a string: `"new_upward" |
| `detect_performance_plateau` | Check if the last `window` points have minimal range `(max-min)/mean < tolerance`. | `(df, value_col='value', tolerance=0.01, window=7)` | Returns `True` if plateau, else `False`. |
| `detect_record_high` | Check if the latest value is the highest on record. | `(df, value_col='value')` | Returns boolean. |
| `detect_record_low` | Check if the latest value is the lowest on record. | `(df, value_col='value')` | Returns boolean. |
| `detect_anomaly_with_variance` | Rolling mean/std-based outlier detection. Mark if abs(value - mean) > z_thresh * std. | `(df, value_col='value', window=7, z_thresh=3.0)` | Returns a DF with columns `[rolling_mean, rolling_std, is_anomaly=True/False]`. |
| `detect_spc_anomalies` | Basic ±3 sigma control chart approach (UCL, LCL). | `(df, value_col='value', window=7)` | Returns a DF with `[UCL, LCL, spc_anomaly=True/False]`. |
| `detect_anomaly_ml` | Single-feature outlier detection using `IsolationForest`. | `(df, value_col='value', contamination=0.05)` | Returns a DF with `is_anomaly_ml=True/False`. |
| `detect_anomalies` | Ensemble approach combining variance, SPC, ML => final_anomaly if any method flags. | `(df, value_col='value')` | Returns a DF with columns `[is_anomaly_variance, spc_anomaly, is_anomaly_ml, final_anomaly]`. |
| `detect_volatility_spike` | If rolling std dev jumps by a factor `ratio_thresh` from one step to next. | `(df, value_col='value', window=7, ratio_thresh=2.0)` | Returns a DF with `[rolling_std, prev_std, vol_spike=True/False]`. |

---

## Descriptive Stats, Data Quality, Misc.

File: **`descriptive_stats.py`** (and possibly some “misc” files)

| **Function** | **Short Purpose** | **Inputs** | **Outputs** |
| --- | --- | --- | --- |
| `calculate_descriptive_stats` | Computes a robust set of summary statistics: count, nulls, min, max, mean, std, mad, skew, kurtosis, outlier counts, CV, etc. | `(df, value_col='value', zscore_threshold=3.0, iqr_multiplier=1.5, etc.)` | Returns a dict with fields like `count, null_count, inf_count, min, max, mean, median, std, iqr, mad, skew, kurtosis, outlier_count_z, outlier_count_iqr, etc.` |