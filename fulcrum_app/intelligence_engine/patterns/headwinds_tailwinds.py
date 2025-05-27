# headwinds_tailwinds.py
"""
Pattern: HeadwindsTailwinds
Version: 1.0

Purpose:
  Assesses potential near-future risks (headwinds) and opportunities (tailwinds)
  for a given metric. It includes analysis of leading indicator trends,
  upcoming seasonal changes, volatility shifts, and concentration risks.

Input Format:
  ledger_df (pd.DataFrame): Historical metric data. Required columns:
    - 'metric_id' (str)
    - 'time_grain' (str)
    - 'date' (datetime-like)
    - 'value' (float)
    - 'dimension' (str, optional for concentration)
    - 'slice_value' (str, optional for concentration)
  metric_id (str): The primary metric to analyze.
  grain (str): The primary time_grain of the metric ('day', 'week', 'month').
  analysis_date (str or pd.Timestamp): The date from which to assess.
  evaluation_time (str or pd.Timestamp): Timestamp of when the analysis is run.
  driver_metrics_info (List[Dict], optional): Information about potential leading indicators.
    Each dict: {'driver_metric_id': str, 'relationship_strength': float (e.g., elasticity/beta)}
  dimensions_for_concentration (List[str], optional): List of dimension column names
    to check for concentration risk (e.g., ['region', 'product_category']).
  seasonal_lookback_years (int): Number of past years for seasonal outlook. Default 2.
  volatility_window_grains (int): Number of grains for current/prior volatility calc. Default 30 for day, 12 for week, 6 for month.
  concentration_top_n_segments (int): Number of top segments for concentration. Default 3.

Output Format (JSON-serializable dict):
  As specified in the problem description, including "leadingIndicators",
  "seasonalOutlook", "volatility", and "concentration".
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Union, List, Optional, Tuple

# Assuming primitives are accessible
from intelligence_engine.primitives.time_series_growth import calculate_pop_growth, calculate_average_growth
from intelligence_engine.primitives.descriptive_stats import calculate_descriptive_stats
from intelligence_engine.primitives.dimensional_analysis import calculate_slice_metrics, compute_slice_shares
# For leading indicators, correlation/Granger might come from ComparativeAnalysis primitives
# from intelligence_engine.primitives.ComparativeAnalysis import ...


def _get_volatility(series: pd.Series, method: str = "cov") -> Optional[float]:
    """Calculates volatility. CoV = (std_dev / mean) * 100. """
    if len(series.dropna()) < 2:
        return None
    if method == "cov": # Coefficient of Variation
        mean = series.mean()
        std_dev = series.std()
        if pd.isna(mean) or pd.isna(std_dev) or abs(mean) < 1e-9:
            return None
        return (std_dev / abs(mean)) * 100.0
    # Can add other methods like std dev of PoP changes
    return None

def _get_grain_offset(grain: str, num_periods: int = 1):
    if grain == 'day':
        return pd.DateOffset(days=num_periods)
    elif grain == 'week':
        return pd.DateOffset(weeks=num_periods)
    elif grain == 'month':
        return pd.DateOffset(months=num_periods)
    raise ValueError(f"Unsupported grain for offset: {grain}")


def run_headwinds_tailwinds_pattern(
    ledger_df: pd.DataFrame,
    metric_id: str,
    grain: str,
    analysis_date: Union[str, pd.Timestamp],
    evaluation_time: Union[str, pd.Timestamp],
    driver_metrics_info: Optional[List[Dict[str, Any]]] = None,
    dimensions_for_concentration: Optional[List[str]] = None,
    seasonal_lookback_years: int = 2,
    volatility_window_grains_map: Optional[Dict[str, int]] = None,
    concentration_top_n_segments: int = 3,
    leading_indicator_trend_periods: int = 3 # e.g., 3 months PoP trend for a monthly driver
) -> Dict[str, Any]:
    analysis_dt = pd.to_datetime(analysis_date)
    eval_time_str = pd.to_datetime(evaluation_time).strftime("%Y-%m-%d %H:%M:%S")
    analysis_date_str = analysis_dt.strftime("%Y-%m-%d")

    output = {
        "schemaVersion": "1.0.0",
        "patternName": "HeadwindsTailwinds",
        "metricId": metric_id,
        "grain": grain,
        "analysisDate": analysis_date_str,
        "evaluationTime": eval_time_str,
        "leadingIndicators": [],
        "seasonalOutlook": {},
        "volatility": {},
        "concentration": []
    }

    # --- Prepare historical data for the primary metric ---
    metric_hist_df = ledger_df[
        (ledger_df['metric_id'] == metric_id) &
        (ledger_df['time_grain'] == grain) &
        (pd.to_datetime(ledger_df['date']) <= analysis_dt)
    ].copy()
    if not pd.api.types.is_datetime64_any_dtype(metric_hist_df['date']):
        metric_hist_df['date'] = pd.to_datetime(metric_hist_df['date'])
    metric_hist_df = metric_hist_df.sort_values(by='date')


    # --- 1. Leading Indicators ---
    if driver_metrics_info:
        for driver_info in driver_metrics_info:
            driver_metric_id = driver_info.get('driver_metric_id')
            # Assuming driver_info also contains 'relationship_strength' (e.g. elasticity)
            # and 'driver_grain' if different from main metric grain
            relationship_strength = driver_info.get('relationship_strength', 0.1) # Default impact factor
            driver_grain = driver_info.get('driver_grain', grain) # Assume same grain if not specified

            driver_hist_df = ledger_df[
                (ledger_df['metric_id'] == driver_metric_id) &
                (ledger_df['time_grain'] == driver_grain) &
                (pd.to_datetime(ledger_df['date']) <= analysis_dt)
            ].copy()
            if not pd.api.types.is_datetime64_any_dtype(driver_hist_df['date']):
                driver_hist_df['date'] = pd.to_datetime(driver_hist_df['date'])
            driver_hist_df = driver_hist_df.sort_values(by='date')

            if len(driver_hist_df) >= leading_indicator_trend_periods + 1:
                # Calculate PoP trend for the driver
                # Use calculate_pop_growth from time_series_growth primitive
                driver_pop_df = calculate_pop_growth(
                    df=driver_hist_df.tail(leading_indicator_trend_periods + 1), # Use recent periods for trend
                    date_col='date', value_col='value', periods=1
                )
                avg_pop_trend_pct = driver_pop_df['pop_growth'].mean() # This is already a percentage

                if pd.notna(avg_pop_trend_pct):
                    trend_direction = "positive" if avg_pop_trend_pct > 0 else "negative" if avg_pop_trend_pct < 0 else "stable"
                    # Simplified potential impact: driver_trend * relationship_strength
                    potential_impact = avg_pop_trend_pct * relationship_strength
                    
                    output["leadingIndicators"].append({
                        "driverMetric": driver_metric_id,
                        "method": driver_info.get("method", "assumed_relationship"), # e.g. correlation, regression
                        "trendDirection": trend_direction,
                        "popTrendPercent": round(avg_pop_trend_pct, 2),
                        "potentialImpactPercent": round(potential_impact, 2)
                    })

    # --- 2. Seasonal Outlook ---
    if not metric_hist_df.empty:
        # Determine upcoming period (e.g., next month if grain is month)
        # For simplicity, look at the period starting one grain after analysis_date
        upcoming_period_start = analysis_dt + _get_grain_offset(grain, 1)
        upcoming_period_end = upcoming_period_start + _get_grain_offset(grain, 1) - _get_grain_offset('day',1) # End of that grain period
        
        # Identify the corresponding month/week_of_year for the upcoming period
        target_month = upcoming_period_start.month
        target_week = upcoming_period_start.isocalendar().week if grain == 'week' else None
        
        historical_changes = []
        for year_offset in range(1, seasonal_lookback_years + 1):
            # Look at the same period in previous years
            prev_year_upcoming_start = upcoming_period_start - pd.DateOffset(years=year_offset)
            prev_year_upcoming_end = upcoming_period_end - pd.DateOffset(years=year_offset)
            
            # And the period before that in the previous year, for PoP change
            prev_year_period_before_start = prev_year_upcoming_start - _get_grain_offset(grain,1)
            prev_year_period_before_end = prev_year_upcoming_start - _get_grain_offset('day',1)

            val_target_period_prev_year_df = metric_hist_df[
                (metric_hist_df['date'] >= prev_year_upcoming_start) & (metric_hist_df['date'] <= prev_year_upcoming_end)
            ]
            val_before_period_prev_year_df = metric_hist_df[
                (metric_hist_df['date'] >= prev_year_period_before_start) & (metric_hist_df['date'] <= prev_year_period_before_end)
            ]

            if not val_target_period_prev_year_df.empty and not val_before_period_prev_year_df.empty:
                val_target = val_target_period_prev_year_df['value'].sum() # Assuming sum for the period
                val_before = val_before_period_prev_year_df['value'].sum()
                if val_before != 0:
                    historical_changes.append((val_target / val_before - 1) * 100.0)
        
        if historical_changes:
            avg_hist_change = np.mean(historical_changes)
            output["seasonalOutlook"] = {
                "historicalChangePercent": round(avg_hist_change, 2),
                "periodLengthGrains": 1, # Assuming we're looking at one grain ahead
                "expectedEndDate": upcoming_period_end.strftime("%Y-%m-%d"),
                "direction": "favorable" if avg_hist_change > 0 else "unfavorable" if avg_hist_change < 0 else "neutral"
            }

    # --- 3. Volatility ---
    default_vol_window = {'day': 30, 'week': 12, 'month': 6}
    volatility_window = volatility_window_grains_map.get(grain, default_vol_window.get(grain,6)) if volatility_window_grains_map else default_vol_window.get(grain,6)

    if len(metric_hist_df) >= 2 * volatility_window:
        current_period_series = metric_hist_df['value'].tail(volatility_window)
        previous_period_series = metric_hist_df['value'].iloc[-(2 * volatility_window):-volatility_window]

        current_vol = _get_volatility(current_period_series)
        previous_vol = _get_volatility(previous_period_series)

        if current_vol is not None and previous_vol is not None:
            output["volatility"]["currentVolatilityPercent"] = round(current_vol, 2)
            output["volatility"]["previousVolatilityPercent"] = round(previous_vol, 2)
            if previous_vol != 0:
                change_pct = (current_vol / previous_vol - 1) * 100.0
                output["volatility"]["volatilityChangePercent"] = round(change_pct, 2)
            else:
                 output["volatility"]["volatilityChangePercent"] = None # Or a large number if current_vol is non-zero
        else: # Not enough data or zero mean for CoV
             output["volatility"]["currentVolatilityPercent"] = None
             output["volatility"]["previousVolatilityPercent"] = None
             output["volatility"]["volatilityChangePercent"] = None


    # --- 4. Concentration Risk ---
    if dimensions_for_concentration and not metric_hist_df.empty:
        # Use data from the most recent complete grain period ending on or before analysis_dt
        # For simplicity, let's use the period ending on analysis_dt if data exists,
        # otherwise the most recent available period.
        last_metric_date = metric_hist_df['date'].max()
        concentration_period_end = analysis_dt
        # Find the start of the grain period that analysis_dt (or last_metric_date) falls into
        # This logic ensures we are looking at a consistent period for all slices
        
        # For concentration, we typically look at the most recent *complete* period or current state.
        # Let's use the values from the period defined by analysis_dt.
        # To do this, we need the start of the period analysis_dt is in.
        
        temp_current_start, temp_current_end = get_current_period_boundaries(analysis_dt, grain)
        
        # Filter ledger_df (which has all dimensions/slices) for the concentration period
        concentration_data_df = ledger_df[
            (ledger_df['metric_id'] == metric_id) &
            (ledger_df['time_grain'] == grain) & # Data should be at the correct grain already for this metric
            (pd.to_datetime(ledger_df['date']) >= temp_current_start) &
            (pd.to_datetime(ledger_df['date']) <= temp_current_end)
        ].copy()
        if not pd.api.types.is_datetime64_any_dtype(concentration_data_df['date']):
             concentration_data_df['date'] = pd.to_datetime(concentration_data_df['date'])


        for dim_name_col in dimensions_for_concentration:
            if dim_name_col in concentration_data_df.columns and 'slice_value' in concentration_data_df.columns: # Assuming general structure
                # The user's ledger_df has `dimension` and `slice_value`.
                # `dim_name_col` here refers to values within the `dimension` column like 'region'.
                # So we filter for that dimension type first.
                dim_specific_data_df = concentration_data_df[concentration_data_df['dimension'] == dim_name_col]

                if not dim_specific_data_df.empty:
                    # Now, aggregate by the actual slices in 'slice_value'
                    # Using calculate_slice_metrics primitive
                    slice_agg_df = calculate_slice_metrics(
                        df=dim_specific_data_df,
                        slice_col='slice_value', # This is the column with actual segment names
                        value_col='value',
                        agg_func='sum' # Sum values within the period for each slice
                    )
                    # slice_agg_df has 'slice_value' and 'aggregated_value'

                    if not slice_agg_df.empty and slice_agg_df['aggregated_value'].sum() > 0:
                        # Compute shares for these aggregated slices
                        # Primitive expects 'slice_col', 'val_col'
                        shared_df = compute_slice_shares(
                            agg_df=slice_agg_df,
                            slice_col='slice_value', # from calculate_slice_metrics output
                            val_col='aggregated_value' # from calculate_slice_metrics output
                        )
                        # shared_df has 'slice_value', 'aggregated_value', 'share_pct'

                        # Sort by share to get top N
                        shared_df_sorted = shared_df.sort_values(by='share_pct', ascending=False)
                        top_n = shared_df_sorted.head(concentration_top_n_segments)

                        if not top_n.empty:
                            output["concentration"].append({
                                "dimensionName": dim_name_col, # This is the type of dimension, e.g., "region"
                                "topSegmentSlices": top_n['slice_value'].tolist(),
                                "topSegmentSharePercent": round(top_n['share_pct'].sum(), 2)
                            })
    return output


if __name__ == '__main__':
    # Sample Ledger Data
    num_days_hist = 365 * 2 # Two years of daily data
    base_dt = datetime(2025, 2, 1)
    dates = pd.to_datetime([base_dt - timedelta(days=i) for i in range(num_days_hist)])
    dates = sorted(dates)

    data = {
        'date': [], 'metric_id': [], 'time_grain': [], 'value': [],
        'dimension': [], 'slice_value': []
    }
    np.random.seed(0)

    # --- active_users (monthly) ---
    # Generate monthly dates for active_users
    monthly_dates = pd.date_range(start=dates[0].replace(day=1), end=base_dt, freq='MS')
    active_users_series = 1000 + np.arange(len(monthly_dates)) * 10 + np.random.normal(0, 50, len(monthly_dates))
    # Seasonality: dip in Feb, peak in Nov
    for i, d_m in enumerate(monthly_dates):
        val = active_users_series[i]
        if d_m.month == 2: val *= 0.85 # Feb dip
        if d_m.month == 7: val *= 0.9 # Summer dip
        if d_m.month == 11: val *= 1.15 # Nov peak
        data['date'].append(d_m)
        data['metric_id'].append('active_users')
        data['time_grain'].append('month')
        data['value'].append(round(max(200, val)))
        data['dimension'].append('Overall') # For main metric series
        data['slice_value'].append('Overall')
        # For concentration for active_users by region
        for region in ['NA', 'EU', 'APAC', 'LATAM', 'Other']:
            share = {'NA': 0.4, 'EU': 0.3, 'APAC': 0.15, 'LATAM': 0.1, 'Other': 0.05}[region]
            data['date'].append(d_m)
            data['metric_id'].append('active_users')
            data['time_grain'].append('month')
            data['value'].append(round(max(20, val * share * (1 + np.random.uniform(-0.1,0.1)))))
            data['dimension'].append('region') # Dimension type
            data['slice_value'].append(region)  # Actual slice

    # --- website_traffic (monthly, leading indicator for active_users) ---
    website_traffic_series = 8000 + np.arange(len(monthly_dates)) * 60 + np.random.normal(0, 300, len(monthly_dates))
    # Let's make its recent trend significant
    website_traffic_series[-6:] = website_traffic_series[-6:] * (1 + np.linspace(0.05, 0.15, 6)) # Accelerating growth recently

    for i, d_m in enumerate(monthly_dates):
        data['date'].append(d_m)
        data['metric_id'].append('website_traffic')
        data['time_grain'].append('month')
        data['value'].append(round(max(1000, website_traffic_series[i])))
        data['dimension'].append('Overall')
        data['slice_value'].append('Overall')

    sample_ledger = pd.DataFrame(data)
    sample_ledger['date'] = pd.to_datetime(sample_ledger['date'])

    # Driver info for leading indicators
    drivers = [
        {
            'driver_metric_id': 'website_traffic',
            'relationship_strength': 0.02, # e.g., 1% change in traffic -> 0.02% change in active users
            'driver_grain': 'month',
            'method': 'correlation_derived_strength'
        }
    ]
    
    # Dimensions for concentration check
    dims_conc = ['region'] # This means we look for rows where ledger_df['dimension'] == 'region'

    print("--- Testing HeadwindsTailwinds Pattern for 'active_users' (month grain) ---")
    analysis_date_test = base_dt # Use the last date of generated data as analysis date
    eval_time_test = datetime.now()

    result = run_headwinds_tailwinds_pattern(
        ledger_df=sample_ledger,
        metric_id='active_users',
        grain='month',
        analysis_date=analysis_date_test,
        evaluation_time=eval_time_test,
        driver_metrics_info=drivers,
        dimensions_for_concentration=dims_conc,
        seasonal_lookback_years=1, # Use 1 year for this data
        concentration_top_n_segments=3,
        leading_indicator_trend_periods=3 # PoP trend over last 3 months for driver
    )
    import json
    print(json.dumps(result, indent=2, default=str)) # default=str for datetime/numpy if any