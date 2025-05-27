# File: intelligence_engine/patterns/historical_performance.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Assuming these are correctly structured base classes from your system
# For this example, I'll define simple placeholders if they are not strictly needed for the logic itself.
try:
    from fulcrum_app.intelligence_engine.data_structures import PatternOutput
    from .base_pattern import Pattern # Assuming base_pattern.py is in the same directory
except ImportError:
    # Placeholder for local execution if actual base classes are not available
    class Pattern:
        PATTERN_NAME = "unknown_pattern"
        PATTERN_VERSION = "0.0.0"
        def validate_data(self, data, required_cols):
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Data missing one of required columns: {required_cols}")
        def validate_analysis_window(self, window):
            if not all(k in window for k in ["start_date", "end_date"]):
                raise ValueError("Analysis window must contain 'start_date' and 'end_date'")
        def handle_empty_data(self, metric_id, analysis_window):
            # This would typically call _minimal_output or similar
            return self._minimal_output(metric_id, analysis_window, analysis_window.get("grain", "day"),
                                       pd.to_datetime(analysis_window["start_date"]), pd.to_datetime(analysis_window["end_date"]), 0)


    class PatternOutput:
        def __init__(self, pattern_name, pattern_version, metric_id, analysis_window, results, error_message=None):
            self.pattern_name = pattern_name
            self.pattern_version = pattern_version
            self.metric_id = metric_id
            self.analysis_window = analysis_window
            self.results = results
            self.error_message = error_message

# Primitives modules (assuming they are in the same directory or accessible via PYTHONPATH)
# Adjust import paths as per your actual project structure.
from intelligence_engine.primitives.time_series_growth import (
    calculate_pop_growth,
    _convert_grain_to_freq # Assuming this helper is made accessible or replicated
)
from intelligence_engine.primitives.trend_analysis import (
    analyze_metric_trend
    # Other trend primitives like detect_record_high/low might be used if preferred
)


class HistoricalPerformancePattern(Pattern):
    """
    Analyzes a metric's historical performance over a specified lookback window.
    It computes period-over-period growth, acceleration, trend analysis, record
    highs/lows, basic seasonality checks, benchmark comparisons, and flags
    exceptions such as spikes/drops.
    """

    PATTERN_NAME = "HistoricalPerformance" # Corrected to match JSON output
    PATTERN_VERSION = "1.0.0" # Corrected to match JSON output

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame, # Historical data for the metric
        analysis_window: Dict[str, str], # Defines lookback_start, lookback_end, grain
        num_periods_to_analyze: int = 12, # Number of resampled periods for detailed PoP, accel etc.
        trend_exception_std_devs: float = 2.0,
        trend_exception_window: int = 5,
        seasonality_deviation_threshold_pct: float = 5.0 # Max dev from expected YoY for "following pattern"
    ) -> PatternOutput:
        """
        Execute the HistoricalPerformancePattern.
        """
        try:
            self.validate_data(data, ["date", "value"])
            self.validate_analysis_window(analysis_window)

            if data.empty:
                return self.handle_empty_data(metric_id, analysis_window)

            data_copy = data.copy()
            data_copy["date"] = pd.to_datetime(data_copy["date"])
            data_copy = data_copy.sort_values("date").reset_index(drop=True)

            lookback_start_dt = pd.to_datetime(analysis_window["start_date"])
            lookback_end_dt = pd.to_datetime(analysis_window["end_date"])
            grain = analysis_window.get("grain", "day").lower()
            current_analysis_date_for_output = lookback_end_dt # The "as of" date for the analysis

            # 1. Filter data to the overall lookback window
            data_in_window = data_copy[
                (data_copy["date"] >= lookback_start_dt) & (data_copy["date"] <= lookback_end_dt)
            ].copy()

            if data_in_window.empty:
                return self._minimal_output(metric_id, analysis_window, grain, lookback_start_dt, lookback_end_dt, 0)

            # 2. Resample data to the specified grain for PoP, trend, and recent exception analysis
            # We'll take the sum for the grain period. Other aggregations like 'last' or 'mean' can be chosen.
            freq = _convert_grain_to_freq(grain)
            resampled_data = data_in_window.set_index('date')['value'].resample(freq).sum().dropna().reset_index()
            
            # Ensure we only use up to lookback_end_dt after resampling
            resampled_data = resampled_data[resampled_data['date'] <= lookback_end_dt]


            if len(resampled_data) < 2: # Need at least 2 periods for most analyses
                return self._minimal_output(metric_id, analysis_window, grain, lookback_start_dt, lookback_end_dt, len(resampled_data))

            # 3. Focus on the most recent `num_periods_to_analyze` for detailed PoP/acceleration
            grouped_for_pop = resampled_data.tail(num_periods_to_analyze).copy()
            if len(grouped_for_pop) < 2: # Still need at least 2 for PoP
                 grouped_for_pop = resampled_data.tail(2).copy() if len(resampled_data) >=2 else resampled_data.copy()


            # --- PoP Growth and Acceleration (on `grouped_for_pop`) ---
            pop_growth_df = calculate_pop_growth(
                df=grouped_for_pop, date_col="date", value_col="value", periods=1, growth_col_name="pop_growth_pct"
            )

            pop_growth_rates_list = []
            if len(pop_growth_df) > 1:
                for i in range(1, len(pop_growth_df)): # Starts from the first period with growth
                    current_period = pop_growth_df.iloc[i]
                    # Find the start of the current period (which is end of prior for popGrowth)
                    # This depends on grain. For simplicity, use the date from resampled_data.
                    prior_period_end_date = pop_growth_df.iloc[i-1]['date']
                    current_period_end_date = current_period['date']
                    
                    pop_growth_rates_list.append({
                        "periodStart": prior_period_end_date.strftime("%Y-%m-%d"),
                        "periodEnd": current_period_end_date.strftime("%Y-%m-%d"),
                        "popGrowthPercent": round(current_period["pop_growth_pct"], 2) if pd.notna(current_period["pop_growth_pct"]) else None
                    })
            
            acceleration_rates_list = []
            if len(pop_growth_rates_list) > 1:
                for i in range(1, len(pop_growth_rates_list)):
                    current_growth = pop_growth_rates_list[i]["popGrowthPercent"]
                    previous_growth = pop_growth_rates_list[i-1]["popGrowthPercent"]
                    accel = None
                    if current_growth is not None and previous_growth is not None:
                        accel = round(current_growth - previous_growth, 2)
                    acceleration_rates_list.append({
                        "periodStart": pop_growth_rates_list[i-1]["periodEnd"], # Accel is between two growth periods
                        "periodEnd": pop_growth_rates_list[i]["periodEnd"],
                        "popAccelerationPercent": accel
                    })

            current_pop = pop_growth_rates_list[-1]["popGrowthPercent"] if pop_growth_rates_list else None
            valid_growths = [g["popGrowthPercent"] for g in pop_growth_rates_list if g["popGrowthPercent"] is not None]
            avg_pop = round(np.mean(valid_growths), 2) if valid_growths else None
            current_accel = acceleration_rates_list[-1]["popAccelerationPercent"] if acceleration_rates_list else None
            
            num_accel, num_slow = 0, 0
            if acceleration_rates_list:
                for accel_rate_info in reversed(acceleration_rates_list):
                    rate = accel_rate_info["popAccelerationPercent"]
                    if rate is None: break
                    if rate > 0.01: # Threshold for acceleration
                        if num_slow > 0: break # Streak broken
                        num_accel += 1
                    elif rate < -0.01: # Threshold for slowing
                        if num_accel > 0: break # Streak broken
                        num_slow += 1
                    else: # Stable, breaks streak
                        break
            
            # --- Trend Analysis (on full `resampled_data` within lookback) ---
            trend_analysis_input_df = resampled_data # Use all resampled periods for robust trend
            trend_results = analyze_metric_trend(
                df=trend_analysis_input_df, value_col="value", date_col="date",
                window_size=min(7, len(trend_analysis_input_df)) # Window for recent direction
            )
            trend_direction_map = {"up": "New Upward", "down": "New Downward", "stable": "Stable", "insufficient_data": "None"}
            trend_type_str = trend_direction_map.get(trend_results.get("trend_direction", "None"), "None")
            if trend_results.get("is_plateaued", False): trend_type_str = "Plateau"
            
            trend_start_dt_str = trend_analysis_input_df['date'].iloc[0].strftime("%Y-%m-%d") if not trend_analysis_input_df.empty else lookback_start_dt.strftime("%Y-%m-%d")
            
            # Calculate average PoP growth for the trend_analysis_input_df period
            trend_period_pop_df = calculate_pop_growth(trend_analysis_input_df, date_col="date", value_col="value", periods=1, growth_col_name="pop_growth_pct")
            trend_avg_pop_val = trend_period_pop_df["pop_growth_pct"].mean() if not trend_period_pop_df["pop_growth_pct"].dropna().empty else None
            trend_avg_pop_output = round(trend_avg_pop_val,2) if pd.notna(trend_avg_pop_val) else None


            # --- Previous Trend Info ---
            # Analyze trend on all but the last period of `trend_analysis_input_df`
            prev_trend_type_str, prev_trend_start_dt_str, prev_trend_avg_pop_output, prev_trend_duration = "None", trend_start_dt_str, None, 0
            if len(trend_analysis_input_df) > 1:
                prev_trend_input_df = trend_analysis_input_df.iloc[:-1]
                if len(prev_trend_input_df) > 1:
                    prev_trend_results = analyze_metric_trend(
                        df=prev_trend_input_df, value_col="value", date_col="date",
                        window_size=min(7, len(prev_trend_input_df))
                    )
                    prev_trend_type_str = trend_direction_map.get(prev_trend_results.get("trend_direction", "None"), "None")
                    if prev_trend_results.get("is_plateaued", False): prev_trend_type_str = "Plateau"
                    prev_trend_start_dt_str = prev_trend_input_df['date'].iloc[0].strftime("%Y-%m-%d")
                    
                    prev_trend_pop_df = calculate_pop_growth(prev_trend_input_df, date_col="date", value_col="value", periods=1, growth_col_name="pop_growth_pct")
                    prev_trend_avg_pop_val = prev_trend_pop_df["pop_growth_pct"].mean() if not prev_trend_pop_df["pop_growth_pct"].dropna().empty else None
                    prev_trend_avg_pop_output = round(prev_trend_avg_pop_val,2) if pd.notna(prev_trend_avg_pop_val) else None
                    prev_trend_duration = len(prev_trend_input_df)

            # --- Record High/Low (on `data_in_window` - original grain) ---
            def _calculate_record_info(df_window, value_col, is_high=True):
                if df_window.empty: return {}
                sorted_df = df_window.sort_values(by=value_col, ascending=not is_high)
                record_val = sorted_df[value_col].iloc[0]
                record_date = sorted_df['date'].iloc[0]
                rank = 1 # By definition, it's the 1st highest/lowest in this window
                
                prior_record_val, prior_record_date_str, abs_delta, rel_delta = None, None, None, None
                if len(sorted_df) > 1:
                    prior_record_val = sorted_df[value_col].iloc[1]
                    prior_record_date_str = sorted_df['date'].iloc[1].strftime("%Y-%m-%d")
                    abs_delta = record_val - prior_record_val
                    rel_delta = (abs_delta / abs(prior_record_val) * 100.0) if prior_record_val != 0 and pd.notna(prior_record_val) else None
                
                return {
                    "value": round(record_val,2), "date": record_date.strftime("%Y-%m-%d"), "rank": rank,
                    "numPeriodsCompared": len(df_window), # Number of original data points in window
                    "priorRecordValue": round(prior_record_val,2) if pd.notna(prior_record_val) else None,
                    "priorRecordDate": prior_record_date_str,
                    "absoluteDeltaFromPriorRecord": round(abs_delta,2) if pd.notna(abs_delta) else None,
                    "relativeDeltaFromPriorRecord": round(rel_delta,2) if pd.notna(rel_delta) else None
                }
            record_high_info = _calculate_record_info(data_in_window, 'value', is_high=True)
            record_low_info = _calculate_record_info(data_in_window, 'value', is_high=False)
            # Adjust keys for JSON
            if "priorRecordValue" in record_high_info: record_high_info["priorRecordHighValue"] = record_high_info.pop("priorRecordValue")
            if "priorRecordDate" in record_high_info: record_high_info["priorRecordHighDate"] = record_high_info.pop("priorRecordDate")
            if "priorRecordValue" in record_low_info: record_low_info["priorRecordLowValue"] = record_low_info.pop("priorRecordValue")
            if "priorRecordDate" in record_low_info: record_low_info["priorRecordLowDate"] = record_low_info.pop("priorRecordDate")


            # --- Basic Seasonality (YoY comparison using `data_in_window`) ---
            seasonality_output = {}
            if not resampled_data.empty:
                last_resampled_period_val = resampled_data['value'].iloc[-1]
                last_resampled_period_date = resampled_data['date'].iloc[-1]
                
                # Find corresponding value from one year ago (at same grain)
                # This requires looking at the original `data_copy` before windowing, then resampling that prior year data.
                # For simplicity, let's find the point in `data_in_window` closest to 1 year before `last_resampled_period_date`.
                date_one_year_ago = last_resampled_period_date - pd.DateOffset(years=1)
                
                # Find the actual data point in the original grain closest to this date last year WITHIN the lookback window
                prior_year_points_in_window = data_in_window[data_in_window['date'] <= date_one_year_ago]
                
                actual_yoy_change_pct = None
                if not prior_year_points_in_window.empty:
                    # Use the value from the same 'grain' period last year.
                    # This means we need to resample the part of data_in_window that corresponds to last year's period.
                    # Simplified: take the last point in data_in_window on or before date_one_year_ago
                    val_one_year_ago = prior_year_points_in_window['value'].iloc[-1]
                    if val_one_year_ago != 0 and pd.notna(val_one_year_ago):
                        actual_yoy_change_pct = (last_resampled_period_val / val_one_year_ago - 1) * 100.0
                
                # Expected change: average YoY changes within the data_in_window
                yoy_diffs_pct = []
                for _, row in data_in_window.iterrows():
                    current_val_point = row['value']
                    current_date_point = row['date']
                    date_last_year_point = current_date_point - pd.DateOffset(years=1)
                    
                    # Find closest point in data_in_window to date_last_year_point
                    historical_match_df = data_in_window[data_in_window['date'] <= date_last_year_point]
                    if not historical_match_df.empty:
                        historical_val_point = historical_match_df['value'].iloc[-1]
                        if historical_val_point != 0 and pd.notna(historical_val_point):
                            yoy_diffs_pct.append((current_val_point / historical_val_point - 1) * 100.0)
                
                expected_yoy_change_pct = np.mean(yoy_diffs_pct) if yoy_diffs_pct else 0.0
                
                deviation_pct = None
                is_following_pattern = False
                if actual_yoy_change_pct is not None:
                    deviation_pct = actual_yoy_change_pct - expected_yoy_change_pct
                    is_following_pattern = abs(deviation_pct) <= seasonality_deviation_threshold_pct
                
                seasonality_output = {
                    "isFollowingExpectedPattern": is_following_pattern,
                    "expectedChangePercent": round(expected_yoy_change_pct,2) if pd.notna(expected_yoy_change_pct) else None,
                    "actualChangePercent": round(actual_yoy_change_pct,2) if pd.notna(actual_yoy_change_pct) else None,
                    "deviationPercent": round(deviation_pct,2) if pd.notna(deviation_pct) else None,
                }


            # --- Benchmark Comparisons ---
            benchmarks_list = []
            # 1. Current period (last in resampled_data) vs. Previous period (second to last)
            if len(resampled_data) >= 2:
                current_period_val = resampled_data['value'].iloc[-1]
                prev_period_val = resampled_data['value'].iloc[-2]
                abs_chg = current_period_val - prev_period_val
                pct_chg = (abs_chg / abs(prev_period_val) * 100.0) if prev_period_val != 0 and pd.notna(prev_period_val) else None
                benchmarks_list.append({
                    "referencePeriod": f"prior {grain}",
                    "absoluteChange": round(abs_chg,2), "changePercent": round(pct_chg,2) if pd.notna(pct_chg) else None
                })
            # 2. Current period vs. Same period last year (using values from seasonality)
            if seasonality_output.get("actualChangePercent") is not None:
                 # Need absolute change for this benchmark
                if actual_yoy_change_pct is not None and yoy_ref_value_for_actual is not None : # From seasonality section
                    abs_change_yoy = last_resampled_period_val - yoy_ref_value_for_actual
                    benchmarks_list.append({
                        "referencePeriod": f"same {grain} last year",
                        "absoluteChange": round(abs_change_yoy,2),
                        "changePercent": seasonality_output["actualChangePercent"] # Already calculated
                    })


            # --- Trend Exceptions (on `resampled_data`) ---
            exceptions_list = []
            if len(resampled_data) >= trend_exception_window:
                recent_periods_for_exc = resampled_data.tail(trend_exception_window)['value']
                mean_recent = recent_periods_for_exc.mean()
                std_recent = recent_periods_for_exc.std()
                
                if pd.notna(mean_recent) and pd.notna(std_recent) and std_recent > 1e-6:
                    upper_b = mean_recent + trend_exception_std_devs * std_recent
                    lower_b = mean_recent - trend_exception_std_devs * std_recent
                    last_val_resampled = resampled_data['value'].iloc[-1]

                    exception_type = None
                    delta_from_norm = None
                    if last_val_resampled > upper_b:
                        exception_type = "Spike"
                        delta_from_norm = last_val_resampled - upper_b
                    elif last_val_resampled < lower_b:
                        exception_type = "Drop"
                        delta_from_norm = last_val_resampled - lower_b # Will be negative

                    if exception_type:
                        # Magnitude relative to the exceeded bound
                        ref_for_mag_pct = upper_b if exception_type == "Spike" else lower_b
                        mag_pct = (delta_from_norm / abs(ref_for_mag_pct) * 100.0) if ref_for_mag_pct !=0 and pd.notna(ref_for_mag_pct) else None
                        exceptions_list.append({
                            "type": exception_type,
                            "currentValue": round(last_val_resampled,2),
                            "normalRangeLow": round(lower_b,2),
                            "normalRangeHigh": round(upper_b,2),
                            "absoluteDeltaFromNormalRange": round(abs(delta_from_norm),2), # Absolute magnitude of deviation
                            "magnitudePercent": round(mag_pct,2) if pd.notna(mag_pct) else None
                        })
            
            # --- Assemble Final Results ---
            final_results = {
                "schemaVersion": self.PATTERN_VERSION, # Using class attribute
                "patternName": self.PATTERN_NAME,    # Using class attribute
                "metricId": metric_id,
                "grain": grain,
                "analysisDate": current_analysis_date_for_output.strftime("%Y-%m-%d"),
                "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "lookbackStart": lookback_start_dt.strftime("%Y-%m-%d"),
                "lookbackEnd": lookback_end_dt.strftime("%Y-%m-%d"),
                "numPeriodsAnalyzed": len(grouped_for_pop), # Num periods used for PoP/Accel calculations
                
                "popGrowthRatesOverWindow": pop_growth_rates_list,
                "accelerationRatesOverWindow": acceleration_rates_list,
                "currentPopGrowthPercent": current_pop,
                "averagePopGrowthPercentOverWindow": avg_pop,
                "currentGrowthAcceleration": current_accel,
                "numPeriodsAccelerating": num_accel,
                "numPeriodsSlowing": num_slow,

                "trendType": trend_type_str,
                "trendStartDate": trend_start_dt_str,
                "trendAveragePopGrowth": trend_avg_pop_output,
                
                "previousTrendType": prev_trend_type_str,
                "previousTrendStartDate": prev_trend_start_dt_str,
                "previousTrendAveragePopGrowth": prev_trend_avg_pop_output,
                "previousTrendDurationGrains": prev_trend_duration,

                "recordHigh": record_high_info,
                "recordLow": record_low_info,
                "seasonality": seasonality_output,
                "benchmarkComparisons": benchmarks_list,
                "trendExceptions": exceptions_list,
            }

            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window, # Pass original window dict
                results=final_results
            )

        except Exception as e:
            # Log the error
            # import traceback
            # print(f"Error in HistoricalPerformancePattern: {e}\n{traceback.format_exc()}")
            error_results = {
                "error": f"An unexpected error occurred: {str(e)}",
                "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results=error_results,
                error_message=str(e)
            )

    def _minimal_output(
        self,
        metric_id: str,
        analysis_window: Dict[str, str],
        grain: str,
        lookback_start_dt: pd.Timestamp,
        lookback_end_dt: pd.Timestamp,
        actual_periods_analyzed: int
    ) -> PatternOutput:
        minimal_results = {
            "schemaVersion": self.PATTERN_VERSION,
            "patternName": self.PATTERN_NAME,
            "metricId": metric_id,
            "grain": grain,
            "analysisDate": lookback_end_dt.strftime("%Y-%m-%d"),
            "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lookbackStart": lookback_start_dt.strftime("%Y-%m-%d"),
            "lookbackEnd": lookback_end_dt.strftime("%Y-%m-%d"),
            "numPeriodsAnalyzed": actual_periods_analyzed,
            "popGrowthRatesOverWindow": [], "accelerationRatesOverWindow": [],
            "currentPopGrowthPercent": None, "averagePopGrowthPercentOverWindow": None,
            "currentGrowthAcceleration": None, "numPeriodsAccelerating": 0, "numPeriodsSlowing": 0,
            "trendType": "None", "trendStartDate": lookback_start_dt.strftime("%Y-%m-%d"),
            "trendAveragePopGrowth": None, "previousTrendType": "None",
            "previousTrendStartDate": lookback_start_dt.strftime("%Y-%m-%d"),
            "previousTrendAveragePopGrowth": None, "previousTrendDurationGrains": 0,
            "recordHigh": {}, "recordLow": {}, "seasonality": {},
            "benchmarkComparisons": [], "trendExceptions": []
        }
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results=minimal_results,
            error_message="Insufficient data for full analysis." if actual_periods_analyzed < 2 else None
        )

if __name__ == '__main__':
    # Example Usage
    
    # Create Sample Data
    base_dt_main = datetime(2024, 5, 15)
    date_rng_main = pd.to_datetime([base_dt_main - timedelta(days=x) for x in range(365*2)]) # 2 years of daily data
    date_rng_main = sorted(date_rng_main)
    
    np.random.seed(42)
    values_main = 100 + np.arange(len(date_rng_main)) * 0.1 + \
                  np.sin(np.arange(len(date_rng_main)) / (365/12) * np.pi) * 20 + \
                  np.random.normal(0, 5, len(date_rng_main))
    values_main[-30:] = values_main[-30:] + 15 # Recent jump
    values_main[100] = values_main[100] * 0.3 # A dip
    values_main[200] = values_main[200] * 1.8 # A spike

    sample_data_df = pd.DataFrame({'date': date_rng_main, 'value': values_main})
    sample_data_df['value'] = sample_data_df['value'].round(2)

    # Define analysis window for the last 90 days, daily grain
    analysis_window_daily = {
        "start_date": (base_dt_main - timedelta(days=89)).strftime("%Y-%m-%d"), # 90 days total
        "end_date": base_dt_main.strftime("%Y-%m-%d"),
        "grain": "day"
    }
    
    pattern_instance_daily = HistoricalPerformancePattern()
    print(f"--- Running HistoricalPerformancePattern (Daily for last 90 days) ---")
    output_daily = pattern_instance_daily.run(
        metric_id="daily_sales",
        data=sample_data_df,
        analysis_window=analysis_window_daily,
        num_periods_to_analyze=30 # Analyze PoP for last 30 days out of 90
    )
    import json
    print(json.dumps(output_daily.results, indent=2, default=str))


    # Define analysis window for the last 12 months, monthly grain
    analysis_window_monthly = {
        "start_date": (base_dt_main - pd.DateOffset(months=11)).replace(day=1).strftime("%Y-%m-%d"),
        "end_date": base_dt_main.strftime("%Y-%m-%d"),
        "grain": "month"
    }
    pattern_instance_monthly = HistoricalPerformancePattern()
    print(f"\n--- Running HistoricalPerformancePattern (Monthly for last 12 months) ---")
    output_monthly = pattern_instance_monthly.run(
        metric_id="monthly_sales",
        data=sample_data_df, # Use same raw daily data, it will be resampled
        analysis_window=analysis_window_monthly,
        num_periods_to_analyze=12
    )
    print(json.dumps(output_monthly.results, indent=2, default=str))

    # Test with insufficient data
    short_data_df = sample_data_df.tail(1)
    print(f"\n--- Running with insufficient data (1 day) ---")
    output_short = pattern_instance_daily.run(
        metric_id="daily_sales_short",
        data=short_data_df,
        analysis_window=analysis_window_daily, # Window is still 90 days
        num_periods_to_analyze=30
    )
    print(json.dumps(output_short.results, indent=2, default=str))