# File: intelligence_engine/patterns/historical_performance.py

"""
Historical Performance Pattern

This module implements the HistoricalPerformancePattern which analyzes a metric's
historical performance over a defined lookback window. It computes period-over-period
growth rates, acceleration (i.e., changes in growth rate), trends, record highs/lows,
a basic seasonality check, benchmark comparisons, and potential trend exceptions like
spikes or drops.

Output Format (JSON-like):
{
  "schemaVersion": "1.0.0",
  "patternName": "HistoricalPerformance",
  "metricId": "string",
  "grain": "day" | "week" | "month",

  // Key dates/times
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",

  // Lookback info
  "lookbackStart": "YYYY-MM-DD",
  "lookbackEnd": "YYYY-MM-DD",
  "numPeriodsAnalyzed": 12,

  // PoP growth details
  "popGrowthRatesOverWindow": [
    {
      "periodStart": "YYYY-MM-DD",
      "periodEnd": "YYYY-MM-DD",
      "popGrowthPercent": 3.2
    },
    ...
  ],

  // Acceleration (second derivative of growth)
  "accelerationRatesOverWindow": [
    {
      "periodStart": "YYYY-MM-DD",
      "periodEnd": "YYYY-MM-DD",
      "popAccelerationPercent": 1.1
    },
    ...
  ],

  // Summaries for current vs. average growth
  "currentPopGrowthPercent": 8.5,
  "averagePopGrowthPercentOverWindow": 5.0,
  "currentGrowthAcceleration": 3.5,
  "numPeriodsAccelerating": 2,
  "numPeriodsSlowing": 0,

  // Trend classification
  "trendType": "Stable" | "New Upward" | "New Downward" | "Plateau" | "None",
  "trendStartDate": "YYYY-MM-DD",
  "trendAveragePopGrowth": 4.2,

  // Previous trend info
  "previousTrendType": "Stable",
  "previousTrendStartDate": "YYYY-MM-DD",
  "previousTrendAveragePopGrowth": 2.1,
  "previousTrendDurationGrains": 6,

  // Record values
  "recordHigh": {
    "value": 1000,
    "rank": 1,
    "numPeriodsCompared": 36,
    "priorRecordHighValue": 950,
    "priorRecordHighDate": "YYYY-MM-DD",
    "absoluteDeltaFromPriorRecord": 50,
    "relativeDeltaFromPriorRecord": 5.26
  },
  "recordLow": {
    "value": 300,
    "rank": 2,
    "numPeriodsCompared": 36,
    "priorRecordLowValue": 305,
    "priorRecordLowDate": "YYYY-MM-DD",
    "absoluteDeltaFromPriorRecord": -5,
    "relativeDeltaFromPriorRecord": -1.64
  },

  // Seasonal analysis
  "seasonality": {
    "isFollowingExpectedPattern": true,
    "expectedChangePercent": 10.0,
    "actualChangePercent": 9.2,
    "deviationPercent": -0.8
  },

  // Benchmark comparisons
  "benchmarkComparisons": [
    {
      "referencePeriod": "priorWTD",
      "absoluteChange": 30.0,
      "changePercent": 5.5
    },
    ...
  ],

  // Trend exceptions (spike/drop)
  "trendExceptions": [
    {
      "type": "Spike" | "Drop",
      "currentValue": 950,
      "normalRangeLow": 800,
      "normalRangeHigh": 900,
      "absoluteDeltaFromNormalRange": 50,
      "magnitudePercent": 5.6
    }
  ]
}
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Base data structures and pattern class
from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from .base_pattern import Pattern

# Primitives modules (time series + trend analysis)
from .time_series_growth import (
    calculate_pop_growth,
    _convert_grain_to_freq
)
from .trend_analysis import (
    analyze_metric_trend
)


class HistoricalPerformancePattern(Pattern):
    """
    Analyzes a metric's historical performance over a specified lookback window.
    It computes period-over-period growth, acceleration, trend analysis, record
    highs/lows, basic seasonality checks, benchmark comparisons, and flags
    exceptions such as spikes/drops.
    """

    PATTERN_NAME = "historical_performance"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str],
        num_periods: int = 12
    ) -> PatternOutput:
        """
        Execute the HistoricalPerformancePattern.

        Parameters
        ----------
        metric_id : str
            ID of the metric being analyzed.
        data : pd.DataFrame
            Must include at least ['date', 'value'] columns.
        analysis_window : Dict[str, str]
            Keys:
               - 'start_date' (YYYY-MM-DD)
               - 'end_date'   (YYYY-MM-DD)
               - 'grain'      (day|week|month) (optional)
        num_periods : int, default=12
            Number of recent resampled periods to analyze.

        Returns
        -------
        PatternOutput
            A PatternOutput containing the JSON results with the structure above.
        """
        try:
            # Validate input data
            self.validate_data(data, ["date", "value"])
            self.validate_analysis_window(analysis_window)

            if data.empty:
                return self.handle_empty_data(metric_id, analysis_window)

            data = data.copy()
            data["date"] = pd.to_datetime(data["date"])
            data.sort_values("date", inplace=True)

            lookback_start = pd.to_datetime(analysis_window["start_date"])
            lookback_end = pd.to_datetime(analysis_window["end_date"])
            grain = analysis_window.get("grain", "day").lower()

            # Filter to lookback
            mask = (data["date"] >= lookback_start) & (data["date"] <= lookback_end)
            data_window = data.loc[mask].copy()
            if data_window.empty:
                return self._minimal_output(metric_id, analysis_window, grain, lookback_start, lookback_end, num_periods)

            # Resample by grain, taking the last value in each period
            freq = _convert_grain_to_freq(grain)  # e.g., "D", "W-MON", "MS"
            data_window.set_index("date", inplace=True)
            grouped = (
                data_window
                .resample(freq)
                .last()[["value"]]
                .dropna(subset=["value"])
            )
            grouped = grouped.reset_index()

            if len(grouped) < 2:
                return self._minimal_output(metric_id, analysis_window, grain, lookback_start, lookback_end, num_periods)

            # Keep only the last num_periods
            if len(grouped) > num_periods:
                grouped = grouped.iloc[-num_periods:].copy()

            # Calculate period-over-period growth
            grouped_growth = calculate_pop_growth(
                df=grouped,
                date_col="date",
                value_col="value",
                periods=1,
                fill_method=None,
                annualize=False,
                growth_col_name="pop_growth"
            )

            pop_growth_rates = []
            for i in range(1, len(grouped_growth)):
                curr = grouped_growth.iloc[i]
                prev = grouped_growth.iloc[i - 1]
                pop_growth_rates.append({
                    "periodStart": prev["date"].strftime("%Y-%m-%d"),
                    "periodEnd": curr["date"].strftime("%Y-%m-%d"),
                    "popGrowthPercent": float(curr["pop_growth"]) if not pd.isna(curr["pop_growth"]) else None
                })

            # Compute acceleration as difference of consecutive growth rates
            acceleration_rates = []
            for i in range(1, len(pop_growth_rates)):
                this_growth = pop_growth_rates[i]["popGrowthPercent"]
                prev_growth = pop_growth_rates[i - 1]["popGrowthPercent"]
                accel_val = None
                if this_growth is not None and prev_growth is not None:
                    accel_val = this_growth - prev_growth
                acceleration_rates.append({
                    "periodStart": pop_growth_rates[i]["periodStart"],
                    "periodEnd": pop_growth_rates[i]["periodEnd"],
                    "popAccelerationPercent": accel_val
                })

            current_pop_growth = pop_growth_rates[-1]["popGrowthPercent"] if pop_growth_rates else None
            valid_g_list = [x["popGrowthPercent"] for x in pop_growth_rates if x["popGrowthPercent"] is not None]
            avg_pop_growth = float(np.mean(valid_g_list)) if valid_g_list else None
            current_growth_acceleration = acceleration_rates[-1]["popAccelerationPercent"] if acceleration_rates else None

            # Count consecutive accelerating/slowing
            num_periods_accelerating = 0
            num_periods_slowing = 0
            for ar in reversed(acceleration_rates):
                val = ar["popAccelerationPercent"]
                if val is None:
                    break
                if val > 0:
                    if num_periods_slowing > 0:
                        break
                    num_periods_accelerating += 1
                elif val < 0:
                    if num_periods_accelerating > 0:
                        break
                    num_periods_slowing += 1
                else:
                    break

            # Analyze trend on the grouped data
            trend_res = analyze_metric_trend(
                df=grouped,
                value_col="value",
                date_col="date",
                window_size=min(5, len(grouped))
            )
            direction_map = {
                "up": "New Upward",
                "down": "New Downward",
                "stable": "Stable",
                "insufficient_data": "None"
            }
            raw_dir = trend_res["trend_direction"]
            if trend_res.get("is_plateaued"):
                trend_type = "Plateau"
            else:
                trend_type = direction_map.get(raw_dir, "None")

            trend_start_date = grouped["date"].iloc[0].strftime("%Y-%m-%d")
            trend_avg_growth = avg_pop_growth if avg_pop_growth is not None else 0.0

            # Previous trend: drop last row if we have enough data
            if len(grouped) > 2:
                prev_subset = grouped.iloc[:-1].copy()
                prev_trend_res = analyze_metric_trend(
                    df=prev_subset,
                    value_col="value",
                    date_col="date"
                )
                if prev_trend_res.get("is_plateaued"):
                    previous_trend_type = "Plateau"
                else:
                    prev_raw_dir = prev_trend_res["trend_direction"]
                    previous_trend_type = direction_map.get(prev_raw_dir, "None")

                previous_trend_start_date = prev_subset["date"].iloc[0].strftime("%Y-%m-%d")
                # Recompute average growth in that subset
                temp_growth = calculate_pop_growth(
                    df=prev_subset,
                    date_col="date",
                    value_col="value",
                    periods=1,
                    fill_method=None,
                    annualize=False,
                    growth_col_name="pop_growth"
                )
                if len(temp_growth) > 1:
                    g_vals = temp_growth["pop_growth"].dropna()
                    pavg = float(np.mean(g_vals)) if not g_vals.empty else None
                else:
                    pavg = None

                previous_trend_average_pop_growth = pavg
                previous_trend_duration_grains = len(prev_subset)
            else:
                previous_trend_type = "None"
                previous_trend_start_date = trend_start_date
                previous_trend_average_pop_growth = None
                previous_trend_duration_grains = 0

            # Evaluate record highs/lows in data_window
            data_window.reset_index(inplace=True)
            data_window.sort_values("value", ascending=False, inplace=True)

            def _compute_record_stats(sorted_df, is_high=True):
                if sorted_df.empty:
                    return {}
                num_compared = len(sorted_df)
                record_val = float(sorted_df.iloc[0]["value"])
                record_date = sorted_df.iloc[0]["date"]

                if len(sorted_df) > 1:
                    prior_val = float(sorted_df.iloc[1]["value"])
                    prior_date = sorted_df.iloc[1]["date"]
                    abs_delta = record_val - prior_val
                    rel_delta = None
                    if prior_val != 0:
                        rel_delta = (abs_delta / prior_val) * 100.0
                    rank = 1
                    prior_date_str = prior_date.strftime("%Y-%m-%d")
                else:
                    prior_val = None
                    abs_delta = None
                    rel_delta = None
                    rank = 1
                    prior_date_str = None

                out = {
                    "value": record_val,
                    "rank": rank,
                    "numPeriodsCompared": num_compared,
                    (
                        "priorRecordHighValue" if is_high else "priorRecordLowValue"
                    ): prior_val,
                    (
                        "priorRecordHighDate" if is_high else "priorRecordLowDate"
                    ): prior_date_str,
                    "absoluteDeltaFromPriorRecord": abs_delta,
                    "relativeDeltaFromPriorRecord": rel_delta
                }
                return out

            record_high = _compute_record_stats(data_window, is_high=True)
            data_window.sort_values("value", ascending=True, inplace=True)
            record_low = _compute_record_stats(data_window, is_high=False)

            # Check seasonality (simple year-over-year)
            seasonality_data = {}
            data_window.sort_values("date", inplace=True)
            yoy_date = lookback_end - timedelta(days=365)
            yoy_df = data_window[data_window["date"] <= yoy_date]
            if not yoy_df.empty:
                yoy_ref_value = float(yoy_df.iloc[-1]["value"])
                current_val = float(data_window.iloc[-1]["value"])
                if yoy_ref_value != 0:
                    actual_change_percent = ((current_val - yoy_ref_value) / yoy_ref_value) * 100.0
                else:
                    actual_change_percent = None

                yoy_changes = []
                for idx, row in data_window.iterrows():
                    this_date = row["date"]
                    ref_date = this_date - timedelta(days=365)
                    subset = data_window[data_window["date"] <= ref_date]
                    if not subset.empty:
                        ref_val = float(subset.iloc[-1]["value"])
                        cur_val = float(row["value"])
                        if ref_val != 0:
                            yoy_changes.append((cur_val - ref_val) / ref_val * 100.0)

                if yoy_changes:
                    expected_change = float(np.mean(yoy_changes))
                else:
                    expected_change = 0.0

                if actual_change_percent is not None:
                    deviation_percent = actual_change_percent - expected_change
                    is_following = abs(deviation_percent) <= 2.0
                    seasonality_data = {
                        "isFollowingExpectedPattern": is_following,
                        "expectedChangePercent": expected_change,
                        "actualChangePercent": actual_change_percent,
                        "deviationPercent": deviation_percent
                    }

            # Benchmark comparisons (example: current week vs. prior week if daily)
            benchmark_comparisons = []
            if grain == "day":
                last_date = data_window["date"].max()
                current_week_monday = last_date - pd.Timedelta(days=last_date.dayofweek)
                c_start = current_week_monday
                c_end = last_date

                p_start = c_start - pd.Timedelta(days=7)
                p_end = c_end - pd.Timedelta(days=7)

                c_mask = (data_window["date"] >= c_start) & (data_window["date"] <= c_end)
                p_mask = (data_window["date"] >= p_start) & (data_window["date"] <= p_end)

                current_sum = data_window.loc[c_mask, "value"].sum()
                prior_sum = data_window.loc[p_mask, "value"].sum()
                abs_change = current_sum - prior_sum
                change_percent = None
                if prior_sum != 0:
                    change_percent = (abs_change / prior_sum) * 100.0

                benchmark_comparisons.append({
                    "referencePeriod": "priorWTD",
                    "absoluteChange": abs_change,
                    "changePercent": change_percent
                })

            # Trend exceptions: simple spike/drop vs. +/- 2 std in last N points
            trend_exceptions = []
            N_FOR_EXCEPTION = min(5, len(grouped))
            recent_subset = grouped.iloc[-N_FOR_EXCEPTION:]["value"]
            mean_val = recent_subset.mean()
            std_val = recent_subset.std()
            if std_val is not None and not np.isnan(std_val) and len(recent_subset) > 1:
                upper_bound = mean_val + 2.0 * std_val
                lower_bound = mean_val - 2.0 * std_val
                last_val = grouped.iloc[-1]["value"]

                if last_val > upper_bound:
                    delta_from_range = last_val - upper_bound
                    magnitude_percent = None
                    if upper_bound != 0:
                        magnitude_percent = (delta_from_range / upper_bound) * 100.0
                    trend_exceptions.append({
                        "type": "Spike",
                        "currentValue": float(last_val),
                        "normalRangeLow": float(lower_bound),
                        "normalRangeHigh": float(upper_bound),
                        "absoluteDeltaFromNormalRange": float(delta_from_range),
                        "magnitudePercent": magnitude_percent
                    })
                elif last_val < lower_bound:
                    delta_from_range = lower_bound - last_val  # positive if last_val < lower_bound
                    magnitude_percent = None
                    if lower_bound != 0:
                        magnitude_percent = (delta_from_range / abs(lower_bound)) * 100.0
                    trend_exceptions.append({
                        "type": "Drop",
                        "currentValue": float(last_val),
                        "normalRangeLow": float(lower_bound),
                        "normalRangeHigh": float(upper_bound),
                        "absoluteDeltaFromNormalRange": float(delta_from_range),
                        "magnitudePercent": magnitude_percent
                    })

            # Final JSON results
            results = {
                "schemaVersion": "1.0.0",
                "patternName": "HistoricalPerformance",
                "metricId": metric_id,
                "grain": grain,
                "analysisDate": lookback_end.strftime("%Y-%m-%d"),
                "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                "lookbackStart": lookback_start.strftime("%Y-%m-%d"),
                "lookbackEnd": lookback_end.strftime("%Y-%m-%d"),
                "numPeriodsAnalyzed": num_periods,

                "popGrowthRatesOverWindow": pop_growth_rates,
                "accelerationRatesOverWindow": acceleration_rates,

                "currentPopGrowthPercent": current_pop_growth,
                "averagePopGrowthPercentOverWindow": avg_pop_growth,
                "currentGrowthAcceleration": current_growth_acceleration,
                "numPeriodsAccelerating": num_periods_accelerating,
                "numPeriodsSlowing": num_periods_slowing,

                "trendType": trend_type,
                "trendStartDate": trend_start_date,
                "trendAveragePopGrowth": trend_avg_growth,

                "previousTrendType": previous_trend_type,
                "previousTrendStartDate": previous_trend_start_date,
                "previousTrendAveragePopGrowth": previous_trend_average_pop_growth,
                "previousTrendDurationGrains": previous_trend_duration_grains,

                "recordHigh": record_high,
                "recordLow": record_low,

                "seasonality": seasonality_data,
                "benchmarkComparisons": benchmark_comparisons,
                "trendExceptions": trend_exceptions
            }

            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results=results
            )

        except Exception as e:
            error_results = {
                "error": str(e),
                "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results=error_results
            )

    def _minimal_output(
        self,
        metric_id: str,
        analysis_window: Dict[str, str],
        grain: str,
        lookback_start: pd.Timestamp,
        lookback_end: pd.Timestamp,
        num_periods: int
    ) -> PatternOutput:
        """
        Returns a minimal output object for scenarios with insufficient data.
        """
        minimal_results = {
            "schemaVersion": "1.0.0",
            "patternName": "HistoricalPerformance",
            "metricId": metric_id,
            "grain": grain,
            "analysisDate": lookback_end.strftime("%Y-%m-%d"),
            "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lookbackStart": lookback_start.strftime("%Y-%m-%d"),
            "lookbackEnd": lookback_end.strftime("%Y-%m-%d"),
            "numPeriodsAnalyzed": num_periods,

            "popGrowthRatesOverWindow": [],
            "accelerationRatesOverWindow": [],
            "currentPopGrowthPercent": None,
            "averagePopGrowthPercentOverWindow": None,
            "currentGrowthAcceleration": None,
            "numPeriodsAccelerating": 0,
            "numPeriodsSlowing": 0,

            "trendType": "None",
            "trendStartDate": lookback_start.strftime("%Y-%m-%d"),
            "trendAveragePopGrowth": None,

            "previousTrendType": "None",
            "previousTrendStartDate": lookback_start.strftime("%Y-%m-%d"),
            "previousTrendAveragePopGrowth": None,
            "previousTrendDurationGrains": 0,

            "recordHigh": {},
            "recordLow": {},
            "seasonality": {},
            "benchmarkComparisons": [],
            "trendExceptions": []
        }
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results=minimal_results
        )
