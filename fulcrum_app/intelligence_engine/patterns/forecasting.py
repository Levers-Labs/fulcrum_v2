# forecasting.py
"""
Pattern: Forecasting
Version: 1.0

Purpose:
  Generates forecasts for a given metric, including projections for specific
  future period end-dates (e.g., end of week, month, quarter) and a detailed
  daily forecast. For each period, it provides statistical forecast details,
  pacing projections, and required performance metrics.

Input Format:
  ledger_df (pd.DataFrame): Historical metric data. Required columns:
    - 'metric_id' (str)
    - 'time_grain' (str)
    - 'date' (datetime-like)
    - 'value' (float)
    (Assumes 'dimension' and 'slice' are filtered to 'Overall'/'Total' before calling)
  targets_df (pd.DataFrame, optional): Future targets. Required columns:
    - 'metric_id' (str)
    - 'date' (datetime-like, representing end of target period)
    - 'target_value' (float)
  metric_id (str): The metric to forecast.
  grain (str): The primary time_grain of the metric ('day', 'week', 'month').
  analysis_date (str or pd.Timestamp): The date from which forecasts are made.
  evaluation_time (str or pd.Timestamp): Timestamp of when the analysis is run.
  forecast_horizon_days (int): How many days out for the detailed daily forecast.
  confidence_level (float): Confidence level for prediction intervals (e.g., 0.95).

Output Format (JSON-serializable dict):
  As specified in the initial problem description, including:
  "schemaVersion", "patternName", "metricId", "grain", "analysisDate",
  "evaluationTime", "forecastPeriods" (array), "dailyForecast" (array).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Assuming primitives are accessible
from intelligence_engine.primitives.forecasting_analysis import simple_forecast # Needs enhancement for intervals
from intelligence_engine.primitives.performance_analysis import calculate_required_growth, classify_metric_status
from intelligence_engine.primitives.time_series_growth import calculate_pop_growth

# --- Helper Functions for Date Calculations ---
def get_period_end_date(analysis_dt: pd.Timestamp, period_name: str) -> pd.Timestamp:
    """Calculates the end date for named periods relative to analysis_dt."""
    if period_name == "endOfWeek":
        return (analysis_dt + pd.offsets.Week(weekday=6)).normalize() # Sunday
    elif period_name == "endOfMonth":
        return (analysis_dt + pd.offsets.MonthEnd(0)).normalize()
    elif period_name == "endOfQuarter":
        return (analysis_dt + pd.offsets.QuarterEnd(0)).normalize()
    elif period_name == "endOfNextMonth":
        return (analysis_dt + pd.offsets.MonthEnd(1)).normalize()
    raise ValueError(f"Unknown period_name: {period_name}")

def get_period_start_date(target_end_dt: pd.Timestamp, period_name_or_grain: str) -> pd.Timestamp:
    """Calculates the start date for a period ending on target_end_dt."""
    # For pacing, we need the start of the period that analysis_date falls into,
    # if period_name refers to the current period.
    # If it refers to a future period (like endOfNextMonth), this logic changes.
    # This helper is more for finding the start of a period given its end.
    if period_name_or_grain == "endOfWeek" or period_name_or_grain == "week":
        return (target_end_dt - pd.offsets.Week(weekday=0) + pd.offsets.Day(0)).normalize() # Monday
    elif period_name_or_grain == "endOfMonth" or period_name_or_grain == "month":
        return target_end_dt.replace(day=1).normalize()
    elif period_name_or_grain == "endOfQuarter" or period_name_or_grain == "quarter":
        return (target_end_dt - pd.offsets.QuarterBegin(startingMonth=1) + pd.offsets.Day(0)).normalize()
    elif period_name_or_grain == "day":
         return target_end_dt.normalize()
    # endOfNextMonth would require knowing which month target_end_dt is in
    elif period_name_or_grain == "endOfNextMonth": # Assumes target_end_dt IS end of next month
         return target_end_dt.replace(day=1).normalize()
    raise ValueError(f"Cannot determine start for: {period_name_or_grain}")


def get_current_period_boundaries(analysis_dt: pd.Timestamp, grain_for_pacing: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get start and end of the current period (week, month, quarter) that analysis_dt is in."""
    if grain_for_pacing == "week": # Assuming grain_for_pacing matches the main forecast period like endOfWeek
        start = (analysis_dt - pd.offsets.Week(weekday=0) + pd.offsets.Day(0)).normalize() # Monday
        end = (start + pd.offsets.Week(weekday=6) - pd.offsets.Day(0)).normalize()     # Sunday
    elif grain_for_pacing == "month":
        start = analysis_dt.replace(day=1).normalize()
        end = (analysis_dt + pd.offsets.MonthEnd(0)).normalize()
    elif grain_for_pacing == "quarter":
        start = (analysis_dt - pd.offsets.QuarterBegin(startingMonth=1) + pd.offsets.Day(0)).normalize()
        end = (analysis_dt + pd.offsets.QuarterEnd(0)).normalize()
    else: # Default to day or if not a standard pacing period
        start = analysis_dt.normalize()
        end = analysis_dt.normalize()
    return start, end

def get_grain_timedelta(grain: str, num_units: int = 1) -> pd.Timedelta:
    if grain == 'day':
        return pd.Timedelta(days=num_units)
    elif grain == 'week':
        return pd.Timedelta(weeks=num_units)
    elif grain == 'month':
        # Approximate for counting, exact dates handled by offsets
        return pd.Timedelta(days=num_units * 30)
    return pd.Timedelta(days=0)

def _mock_simple_forecast_with_intervals(
    series_df: pd.DataFrame, value_col: str, date_col: str,
    periods_to_forecast: int, confidence_level: float, freq: str
) -> pd.DataFrame:
    """ Mocks the forecasting primitive to include intervals. """
    if series_df.empty:
        return pd.DataFrame(columns=[date_col, 'forecastedValue', 'lowerBound', 'upperBound'])

    last_date = series_df[date_col].max()
    last_value = series_df.iloc[-1][value_col]
    
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1 if freq=='D' else 7 if freq=='W-MON' else 30), periods=periods_to_forecast, freq=freq) # Simplistic date stepping
    
    forecast_values = []
    lower_bounds = []
    upper_bounds = []

    for i in range(periods_to_forecast):
        # Simple projection with increasing uncertainty
        fc_val = last_value * (1 + 0.005 * (i+1)) # Slight upward trend
        uncertainty_factor = 0.05 * (i+1)**0.5 # Increasing uncertainty
        lower = fc_val * (1 - uncertainty_factor * (1 + (1-confidence_level))) # Wider for higher confidence
        upper = fc_val * (1 + uncertainty_factor * (1 + (1-confidence_level)))
        forecast_values.append(fc_val)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        
    return pd.DataFrame({
        date_col: forecast_dates,
        'forecastedValue': forecast_values,
        'lowerBound': lower_bounds,
        'upperBound': upper_bounds
    })


def run_forecasting_pattern(
    ledger_df: pd.DataFrame,
    metric_id: str,
    grain: str, # 'day', 'week', 'month' - primary grain of the input metric
    analysis_date: Union[str, pd.Timestamp],
    evaluation_time: Union[str, pd.Timestamp],
    targets_df: Optional[pd.DataFrame] = None, # Columns: metric_id, date (period end), target_value
    forecast_horizon_days: int = 90, # For detailed daily forecast
    confidence_level: float = 0.95,
    pacing_status_threshold_pct: float = 5.0, # e.g., on track if within 5% of target
    num_past_periods_for_growth: int = 4 # For calculating pastPoPGrowthPercent
) -> Dict[str, Any]:
    """
    Executes the Forecasting pattern.
    """
    analysis_dt = pd.to_datetime(analysis_date)
    eval_time_str = pd.to_datetime(evaluation_time).strftime("%Y-%m-%d %H:%M:%S")
    analysis_date_str = analysis_dt.strftime("%Y-%m-%d")

    output = {
        "schemaVersion": "1.0.0",
        "patternName": "Forecasting",
        "metricId": metric_id,
        "grain": grain, # Reflects the primary grain of the input metric & high-level periods
        "analysisDate": analysis_date_str,
        "evaluationTime": eval_time_str,
        "forecastPeriods": [],
        "dailyForecast": []
    }

    # 1. Prepare historical data for the metric
    hist_df = ledger_df[
        (ledger_df['metric_id'] == metric_id) &
        (ledger_df['time_grain'] == grain) & # Filter by main grain for training model
        (pd.to_datetime(ledger_df['date']) <= analysis_dt)
    ].copy()
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values(by='date').set_index('date')['value'].resample(get_grain_timedelta(grain).freqstr).sum().reset_index() # Ensure regular frequency

    if hist_df.empty or len(hist_df) < 2: # Not enough data to forecast
        print(f"Warning: Insufficient historical data for {metric_id} at {grain} grain to generate forecast.")
        return output # Return empty shell

    # 2. Generate Detailed Daily Forecast
    # We always generate a daily forecast for the `dailyForecast` field,
    # regardless of the input `grain`. The model might be trained on `grain` data then interpolated/resampled.
    # For simplicity, if grain is not 'day', we might need a daily version of hist_df or forecast daily.
    # Here, we use a MOCK forecast primitive that outputs daily.
    
    # Determine frequency string for daily forecast generation.
    # The model should ideally be trained on `grain` data, then forecast at that `grain`,
    # and then daily data can be interpolated if needed, or a daily model used.
    # For this example, we'll mock a daily forecast directly.
    
    # To use the provided `simple_forecast`, we need to adapt it for intervals or use a mock.
    # Let's assume `_mock_simple_forecast_with_intervals` exists and gives daily forecasts.
    # For training, it should ideally use `hist_df` which is at the specified `grain`.
    # The mock below will simplify and just project from the last point of daily data.
    
    daily_hist_df_for_mock = ledger_df[
        (ledger_df['metric_id'] == metric_id) &
        (pd.to_datetime(ledger_df['date']) <= analysis_dt)
    ].copy()
    daily_hist_df_for_mock['date'] = pd.to_datetime(daily_hist_df_for_mock['date'])
    daily_hist_df_for_mock = daily_hist_df_for_mock.sort_values(by='date')
    # If main grain is not day, we might need to convert daily_hist_df_for_mock to daily sums
    if grain != 'day':
         daily_hist_df_for_mock = daily_hist_df_for_mock.set_index('date')['value'].resample('D').sum().reset_index()


    daily_fc_df = _mock_simple_forecast_with_intervals(
        series_df=daily_hist_df_for_mock, # Pass daily data to mock
        value_col='value',
        date_col='date',
        periods_to_forecast=forecast_horizon_days,
        confidence_level=confidence_level,
        freq='D' # Daily forecast
    )

    for _, row in daily_fc_df.iterrows():
        output["dailyForecast"].append({
            "date": row['date'].strftime("%Y-%m-%d"),
            "forecastedValue": round(row['forecastedValue'], 2) if pd.notna(row['forecastedValue']) else None,
            "lowerBound": round(row['lowerBound'], 2) if pd.notna(row['lowerBound']) else None,
            "upperBound": round(row['upperBound'], 2) if pd.notna(row['upperBound']) else None,
            "confidenceLevel": confidence_level
        })

    # 3. Process Each Defined Forecast Period
    period_names_to_process = ["endOfWeek", "endOfMonth", "endOfQuarter", "endOfNextMonth"]
    
    # Get latest actual value for "current_value" in required_growth
    latest_actual_value = hist_df['value'].iloc[-1] if not hist_df.empty else 0

    for period_name in period_names_to_process:
        period_obj = {"periodName": period_name}
        forecast_target_date = get_period_end_date(analysis_dt, period_name)

        # --- Statistical Forecast Section ---
        fc_row = daily_fc_df[daily_fc_df['date'] == forecast_target_date]
        if not fc_row.empty:
            period_obj["statisticalForecastedValue"] = round(fc_row.iloc[0]['forecastedValue'], 2)
            period_obj["statisticalLowerBound"] = round(fc_row.iloc[0]['lowerBound'], 2)
            period_obj["statisticalUpperBound"] = round(fc_row.iloc[0]['upperBound'], 2)
        else: # Forecast date might be beyond daily_fc_df horizon or not align
            period_obj["statisticalForecastedValue"] = None # Or extrapolate from main grain forecast
            period_obj["statisticalLowerBound"] = None
            period_obj["statisticalUpperBound"] = None
        
        period_obj["statisticalConfidenceLevel"] = confidence_level
        
        current_target_value = None
        if targets_df is not None and not targets_df.empty:
            target_row = targets_df[
                (targets_df['metric_id'] == metric_id) &
                (pd.to_datetime(targets_df['date']) == forecast_target_date)
            ]
            if not target_row.empty:
                current_target_value = target_row.iloc[0]['target_value']
        period_obj["statisticalTargetValue"] = current_target_value

        if period_obj["statisticalForecastedValue"] is not None and current_target_value is not None and current_target_value != 0:
            gap_pct = (period_obj["statisticalForecastedValue"] / current_target_value - 1) * 100
            period_obj["statisticalForecastedGapPercent"] = round(gap_pct, 2)
            # Using classify_metric_status primitive
            period_obj["statisticalForecastStatus"] = classify_metric_status(
                period_obj["statisticalForecastedValue"], current_target_value, threshold_ratio=pacing_status_threshold_pct / 100.0
            )
        else:
            period_obj["statisticalForecastedGapPercent"] = None
            period_obj["statisticalForecastStatus"] = "no_target_data" if current_target_value is None else "no_forecast"


        # --- Pacing Projection Section ---
        # Determine the grain for pacing based on the period_name
        pacing_period_grain = ""
        if period_name == "endOfWeek": pacing_period_grain = "week"
        elif period_name == "endOfMonth": pacing_period_grain = "month"
        elif period_name == "endOfQuarter": pacing_period_grain = "quarter"
        # "endOfNextMonth" is tricky for current period pacing; skip pacing for it or define specific logic
        
        if pacing_period_grain: # Only do pacing for current week/month/quarter
            current_period_start, current_period_end = get_current_period_boundaries(analysis_dt, pacing_period_grain)

            if analysis_dt >= current_period_start : # analysis_dt must be within or after the period start for pacing
                elapsed_days = (analysis_dt - current_period_start).days + 1 # Inclusive of analysis_dt
                total_days_in_period = (current_period_end - current_period_start).days + 1
                
                period_obj["percentOfPeriodElapsed"] = round((elapsed_days / total_days_in_period) * 100.0, 2) if total_days_in_period > 0 else 0.0

                # Cumulative value from hist_df (which is at `grain`)
                # This needs careful handling of `grain` vs daily `analysis_dt`
                actuals_in_current_period = hist_df[
                    (hist_df['date'] >= current_period_start) & (hist_df['date'] <= analysis_dt) # Sum actuals up to analysis_dt
                ]
                current_cumulative_value = actuals_in_current_period['value'].sum()
                period_obj["currentCumulativeValue"] = round(current_cumulative_value, 2)

                if period_obj["percentOfPeriodElapsed"] > 0 and period_obj["percentOfPeriodElapsed"] < 100 : # Avoid division by zero and pacing if period is over
                    period_obj["pacingProjectedValue"] = round((current_cumulative_value / period_obj["percentOfPeriodElapsed"]) * 100.0, 2)
                elif period_obj["percentOfPeriodElapsed"] >= 100: # Period complete
                     period_obj["pacingProjectedValue"] = round(current_cumulative_value, 2)
                else: # 0% elapsed
                    period_obj["pacingProjectedValue"] = None


                if period_obj["pacingProjectedValue"] is not None and current_target_value is not None and current_target_value !=0:
                    pacing_gap_pct = (period_obj["pacingProjectedValue"] / current_target_value - 1) * 100
                    period_obj["pacingGapPercent"] = round(pacing_gap_pct, 2)
                    period_obj["pacingStatus"] = classify_metric_status(
                        period_obj["pacingProjectedValue"], current_target_value, threshold_ratio=pacing_status_threshold_pct / 100.0
                    )
                else:
                    period_obj["pacingGapPercent"] = None
                    period_obj["pacingStatus"] = "no_target_data" if current_target_value is None else "not_yet_pacing"
            else: # analysis_date is before the current period start (e.g. for endOfNextMonth if it's start of current month)
                period_obj["percentOfPeriodElapsed"] = 0.0
                period_obj["currentCumulativeValue"] = 0.0
                period_obj["pacingProjectedValue"] = None
                period_obj["pacingGapPercent"] = None
                period_obj["pacingStatus"] = "not_yet_started"

        # --- Required Performance Section ---
        if current_target_value is not None:
            # Calculate remaining grains based on the main `grain` of the metric
            remaining_grains_count = 0
            next_grain_start_date = analysis_dt # Default, will be incremented
            if grain == 'day':
                next_grain_start_date = analysis_dt + pd.Timedelta(days=1)
                remaining_grains_count = (forecast_target_date - next_grain_start_date).days + 1
            elif grain == 'week':
                # Start of next week after analysis_dt's week
                next_grain_start_date = (analysis_dt + pd.offsets.Week(weekday=0) + pd.offsets.Day(0) + pd.Timedelta(weeks=1)).normalize()
                # Count weeks by checking how many week starts are there
                if next_grain_start_date <= forecast_target_date:
                     # Number of full weeks from next_grain_start_date up to forecast_target_date's week start + potentially one more if forecast_target_date is in that week
                    temp_date = next_grain_start_date
                    while temp_date <= forecast_target_date :
                        remaining_grains_count +=1
                        temp_date += pd.Timedelta(weeks=1)

            elif grain == 'month':
                next_grain_start_date = (analysis_dt.replace(day=1) + pd.offsets.MonthBegin(1)).normalize()
                if next_grain_start_date <= forecast_target_date:
                    temp_date = next_grain_start_date
                    while temp_date <= forecast_target_date:
                        remaining_grains_count +=1
                        # Move to the start of the next month
                        temp_date = (temp_date.replace(day=1) + pd.offsets.MonthBegin(1)).normalize()


            period_obj["remainingGrainsCount"] = max(0, remaining_grains_count)

            if period_obj["remainingGrainsCount"] > 0:
                req_growth = calculate_required_growth(
                    current_value=latest_actual_value, # Latest known value at `grain`
                    target_value=current_target_value,
                    periods_left=period_obj["remainingGrainsCount"]
                )
                period_obj["requiredPopGrowthPercent"] = round(req_growth * 100, 2) if req_growth is not None else None
            else:
                period_obj["requiredPopGrowthPercent"] = None # Target date is in the past or current grain

            # Past PoP Growth (at the specified 'grain')
            if len(hist_df) >= num_past_periods_for_growth + 1:
                past_df_for_growth = hist_df.tail(num_past_periods_for_growth + 1).copy()
                # calculate_pop_growth needs date and value columns
                past_df_for_growth = calculate_pop_growth(past_df_for_growth, date_col='date', value_col='value', periods=1)
                avg_past_growth = past_df_for_growth['pop_growth'].mean() # This is already a percentage
                period_obj["pastPopGrowthPercent"] = round(avg_past_growth, 2) if pd.notna(avg_past_growth) else None
            else:
                period_obj["pastPopGrowthPercent"] = None

            if period_obj["requiredPopGrowthPercent"] is not None and period_obj["pastPopGrowthPercent"] is not None:
                period_obj["deltaFromHistoricalGrowth"] = round(period_obj["requiredPopGrowthPercent"] - period_obj["pastPopGrowthPercent"], 2)
            else:
                period_obj["deltaFromHistoricalGrowth"] = None
        else: # No target value
            period_obj["remainingGrainsCount"] = None
            period_obj["requiredPopGrowthPercent"] = None
            period_obj["pastPopGrowthPercent"] = None
            period_obj["deltaFromHistoricalGrowth"] = None
            
        output["forecastPeriods"].append(period_obj)

    return output

if __name__ == '__main__':
    # Sample Ledger Data (daily)
    dates_daily = pd.to_datetime([datetime(2024, 1, 1) + timedelta(days=i) for i in range(120)])
    np.random.seed(42)
    values_daily = 100 + np.arange(120) * 0.5 + np.random.normal(0, 10, 120)
    values_daily[80:90] = values_daily[80:90] * 1.2 # Recent surge
    
    ledger_data_daily = []
    for i, date_val in enumerate(dates_daily):
        ledger_data_daily.append(['sales_total', 'day', date_val, values_daily[i]])
        # Simulate weekly data by taking first day of week's value (simplified)
        if date_val.dayofweek == 0: # Monday
             ledger_data_daily.append(['sales_total', 'week', date_val, values_daily[i:i+7].sum()])
        if date_val.day == 1: # First of month
             ledger_data_daily.append(['sales_total', 'month', date_val, values_daily[i:i+30].sum()])


    sample_ledger = pd.DataFrame(ledger_data_daily, columns=['metric_id', 'time_grain', 'date', 'value'])

    # Sample Targets Data
    sample_targets = pd.DataFrame([
        ['sales_total', pd.to_datetime('2024-05-05'), 180], # End of current week for analysis_date 2024-05-01
        ['sales_total', pd.to_datetime('2024-05-31'), 2500], # End of current month
        ['sales_total', pd.to_datetime('2024-06-30'), 2800], # End of next month
        ['sales_total', pd.to_datetime('2024-06-30'), 3000], # End of quarter (Q2)
    ], columns=['metric_id', 'date', 'target_value'])
    
    analysis_date_to_test = pd.to_datetime('2024-05-01') # Wednesday
    evaluation_time_to_test = datetime.now()

    print("--- Testing Forecasting Pattern (grain: day) ---")
    result_day = run_forecasting_pattern(
        ledger_df=sample_ledger,
        metric_id='sales_total',
        grain='day',
        analysis_date=analysis_date_to_test,
        evaluation_time=evaluation_time_to_test,
        targets_df=sample_targets,
        forecast_horizon_days=60, # Shorter for test output
        num_past_periods_for_growth=8 # For pastPoPGrowthPercent
    )
    import json
    print(json.dumps(result_day, indent=2, default=str))

    print("\n--- Testing Forecasting Pattern (grain: week) ---")
    # Ensure target dates align with week endings (Sunday for get_period_end_date)
    sample_targets_weekly = pd.DataFrame([
        ['sales_total', pd.to_datetime('2024-05-05'), 1000], # End of current week
        ['sales_total', pd.to_datetime('2024-05-31').to_period('M').end_time.normalize(), 4500], # End of current month (using month end)
    ], columns=['metric_id', 'date', 'target_value'])

    result_week = run_forecasting_pattern(
        ledger_df=sample_ledger,
        metric_id='sales_total',
        grain='week', # Model trains on weekly aggregated data
        analysis_date=analysis_date_to_test,
        evaluation_time=evaluation_time_to_test,
        targets_df=sample_targets_weekly,
        forecast_horizon_days=60,
        num_past_periods_for_growth=4 # e.g. last 4 weeks
    )
    print(json.dumps(result_week, indent=2, default=str))