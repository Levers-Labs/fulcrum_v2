# intelligence_engine/patterns/performance_status.py

"""
Performance Status Pattern

This module implements the PerformanceStatusPattern which analyzes whether a metric
is on/off track (or exceeding) a given target. It classifies the current status,
tracks how that status has changed, and provides details about gaps or overperformance,
as well as streak information and a "hold steady" scenario.

It uses primitives from the Performance module where possible.

Output Format:
{
  "schemaVersion": "1.0.0",
  "patternName": "PerformanceStatus",
  "metricId": "string",
  "grain": "day" | "week" | "month",

  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",

  // Current vs. prior values
  "currentValue": float,
  "priorValue": float,
  "absoluteDeltaFromPrior": float,
  "popChangePercent": float,

  // Target-related fields
  "targetValue": float,
  "status": "on_track" | "off_track" | "overperforming" | "no_target",
  "absoluteGap": float,               // if off_track
  "absoluteGapDelta": float,          // how gap changed from prior period
  "percentGap": float,                // if off_track
  "percentGapDelta": float,           // how relative gap changed
  "absoluteOverperformance": float,   // if overperforming
  "percentOverperformance": float,    // if overperforming

  // Status change info
  "statusChange": {
    "hasFlipped": bool,
    "oldStatus": str,
    "newStatus": str,
    "oldStatusDurationGrains": int
  },

  // Streak info
  "streak": {
    "length": int,
    "status": str,
    "performanceChangePercentOverStreak": float,
    "absoluteChangeOverStreak": float,
    "averageChangePercentPerGrain": float,
    "averageChangeAbsolutePerGrain": float
  },

  // "Hold steady" scenario
  "holdSteady": {
    "isCurrentlyAtOrAboveTarget": bool,
    "timeToMaintainGrains": int,
    "currentMarginPercent": float
  }
}
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from .base_pattern import Pattern

# Import relevant primitives from the Performance module
from fulcrum_app.intelligence_engine.primitives.Performance import (
    calculate_metric_gva,
    classify_metric_status,
    detect_status_changes
)


class PerformanceStatusPattern(Pattern):
    """
    Analyzes a metric's performance against a target, classifying its status and
    providing detailed information about any gap or overperformance, status changes,
    streaks, and a simple "hold steady" scenario.
    """

    PATTERN_NAME = "PerformanceStatus"
    PATTERN_VERSION = "1.0.0"

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str],
        threshold: float = 0.05
    ) -> PatternOutput:
        """
        Execute the performance status analysis.

        Parameters
        ----------
        metric_id : str
            The ID of the metric being analyzed.
        data : pd.DataFrame
            DataFrame containing columns: date, value, target (optional).
        analysis_window : Dict[str, str]
            Dictionary with 'start_date', 'end_date', and optionally 'grain'.
        threshold : float, default 0.05
            The ratio used to classify a metric as on_track if:
                actual_value >= target_value * (1 - threshold).

        Returns
        -------
        PatternOutput
            The output object with the final results in `results`.
        """
        try:
            # Validate input columns
            required_cols = ["date", "value"]
            self.validate_data(data, required_cols)
            self.validate_analysis_window(analysis_window)

            # Handle empty data
            if data.empty:
                return self.handle_empty_data(metric_id, analysis_window)

            # Convert and sort by date
            df = data.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)

            # Extract grain and analysis date from the window
            grain = analysis_window.get("grain", "day").lower()
            analysis_date = analysis_window.get("end_date", df["date"].iloc[-1].strftime("%Y-%m-%d"))

            # Prepare final/current vs. prior values
            current_value, prior_value = self._get_current_and_prior_values(df)

            # Attempt to derive a target if present
            target_value = None
            has_target = "target" in df.columns
            if has_target:
                valid_targets = df["target"].dropna()
                if not valid_targets.empty:
                    target_value = float(valid_targets.iloc[-1])

            # Classify final status using a threshold-based approach
            # This primitive returns "on_track", "off_track", or "no_target".
            base_status = classify_metric_status(
                actual_value=current_value,
                target_value=target_value,
                threshold_ratio=threshold,
                allow_negative_target=False,
                status_if_no_target="no_target"
            )

            # Distinguish "overperforming" if the actual is strictly above target
            status = self._refine_overperformance_status(current_value, target_value, base_status, threshold)

            # Use GvA to measure gap or overperformance amounts
            gap_info = {}
            if status != "no_target" and target_value is not None:
                gva = calculate_metric_gva(actual_value=current_value, target_value=target_value)
                # gva has {'abs_diff': x, 'pct_diff': y} or None if invalid
                # If abs_diff < 0 => off track; if abs_diff > 0 => overperformance
                # This is purely numeric, not threshold-based logic.

                if gva["abs_diff"] is not None:
                    if gva["abs_diff"] < 0:  # negative => behind target
                        # Gap is positive magnitude of abs_diff
                        gap_info["absoluteGap"] = abs(gva["abs_diff"])
                        gap_info["percentGap"] = abs(gva["pct_diff"]) if gva["pct_diff"] else None

                    elif gva["abs_diff"] > 0:  # above target
                        gap_info["absoluteOverperformance"] = gva["abs_diff"]
                        gap_info["percentOverperformance"] = gva["pct_diff"] if gva["pct_diff"] else None

            # If we have a prior value, compute absolute and percent changes
            abs_delta_from_prior = None
            pop_change_percent = None
            if prior_value is not None:
                abs_delta_from_prior = current_value - prior_value
                if prior_value != 0:
                    pop_change_percent = (abs_delta_from_prior / prior_value) * 100.0

            # Compute gap deltas if we have prior data (for "absoluteGapDelta"/"percentGapDelta")
            # or overperformance deltas if "absoluteOverperformance" was set.
            gap_deltas = {}
            if target_value is not None and prior_value is not None:
                # The prior gap or overperformance
                prior_gap_gva = calculate_metric_gva(
                    actual_value=prior_value, target_value=target_value
                )
                if prior_gap_gva["abs_diff"] is not None:
                    if prior_gap_gva["abs_diff"] < 0:
                        # prior gap is positive magnitude
                        prev_abs_gap = abs(prior_gap_gva["abs_diff"])
                        prev_pct_gap = abs(prior_gap_gva["pct_diff"]) if prior_gap_gva["pct_diff"] else None

                        if "absoluteGap" in gap_info:
                            # new gap
                            new_abs_gap = gap_info["absoluteGap"]
                            gap_deltas["absoluteGapDelta"] = new_abs_gap - prev_abs_gap

                        if "percentGap" in gap_info and prev_pct_gap is not None:
                            new_pct_gap = gap_info["percentGap"]
                            gap_deltas["percentGapDelta"] = new_pct_gap - prev_pct_gap

                    elif prior_gap_gva["abs_diff"] > 0:
                        prev_over = prior_gap_gva["abs_diff"]
                        prev_pct_over = prior_gap_gva["pct_diff"] if prior_gap_gva["pct_diff"] else None
                        if "absoluteOverperformance" in gap_info:
                            new_over = gap_info["absoluteOverperformance"]
                            gap_deltas["absoluteOverperformanceDelta"] = new_over - prev_over

                        if "percentOverperformance" in gap_info and prev_pct_over is not None:
                            new_pct_over = gap_info["percentOverperformance"]
                            gap_deltas["percentOverperformanceDelta"] = new_pct_over - prev_pct_over

            # Determine status changes over time
            status_change_info = {}
            if has_target and len(df) > 1:
                status_change_info = self._compute_status_change_over_time(df, threshold)

            # Compute streak info
            streak_info = {}
            if len(df) > 1:
                streak_info = self._calculate_streak_info(df, status)

            # Compute hold steady scenario
            hold_steady_info = {}
            if target_value is not None and status != "no_target":
                hold_steady_info = self._calculate_hold_steady_scenario(current_value, target_value)

            # Assemble the final result
            results = {
                "schemaVersion": "1.0.0",
                "patternName": self.PATTERN_NAME,
                "metricId": metric_id,
                "grain": grain,
                "analysisDate": analysis_date,
                "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                "currentValue": current_value,
                "status": status
            }

            # Prior value portion
            if prior_value is not None:
                results["priorValue"] = prior_value
            if abs_delta_from_prior is not None:
                results["absoluteDeltaFromPrior"] = abs_delta_from_prior
            if pop_change_percent is not None:
                results["popChangePercent"] = pop_change_percent

            # Target portion
            if target_value is not None:
                results["targetValue"] = target_value

            # Gap or Overperformance portion
            results.update(gap_info)
            results.update(gap_deltas)

            # Status change portion
            if status_change_info:
                results["statusChange"] = status_change_info

            # Streak info
            if streak_info:
                results["streak"] = streak_info

            # Hold steady
            if hold_steady_info:
                results["holdSteady"] = hold_steady_info

            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results=results
            )

        except Exception as ex:
            error_msg = {
                "error": str(ex),
                "evaluationTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results=error_msg
            )

    def _get_current_and_prior_values(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, Optional[float]]:
        """
        Returns the most recent metric value and the second-most recent (if present).
        """
        final_value = float(df["value"].iloc[-1])
        prior_value = float(df["value"].iloc[-2]) if len(df) > 1 else None
        return final_value, prior_value

    def _refine_overperformance_status(
        self,
        current_value: float,
        target_value: Optional[float],
        base_status: str,
        threshold: float
    ) -> str:
        """
        If the base_status is 'on_track' and current_value is significantly
        above target, override status with 'overperforming'.
        Otherwise return base_status or keep 'no_target' or 'off_track'.
        This is purely optional logic to highlight that a metric is
        well above target, not just meeting it.
        """
        if target_value is None or np.isnan(target_value) or base_status == "no_target":
            return "no_target"

        if base_status == "off_track":
            return "off_track"

        # If base_status is "on_track" but the metric is above
        # the target by more than threshold*target, classify "overperforming"
        margin = current_value - target_value
        if margin > (threshold * target_value):
            return "overperforming"
        return "on_track"

    def _compute_status_change_over_time(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Generate a column of statuses over time, then use detect_status_changes
        to find if there's a flip from old to new.
        Returns a dict with the most recent flip info if present.
        """
        # Create a copy and classify row-by-row
        tmp = df.copy()
        tmp["status"] = tmp.apply(
            lambda row: classify_metric_status(
                actual_value=row["value"],
                target_value=row["target"] if "target" in row else None,
                threshold_ratio=threshold,
                allow_negative_target=False,
                status_if_no_target="no_target"
            ),
            axis=1
        )

        # Use the primitive to detect flips
        tmp2 = detect_status_changes(tmp, status_col="status", sort_by_date="date")

        # The last row will show whether there's a flip from prev row
        if tmp2.empty or "status_flip" not in tmp2.columns:
            return {}

        # Identify the last flipped row, if any
        flips = tmp2[tmp2["status_flip"] == True]
        if flips.empty:
            return {}

        last_flip_row = flips.iloc[-1]
        return {
            "hasFlipped": True,
            "oldStatus": last_flip_row["prev_status"],
            "newStatus": last_flip_row["status"],
            # We can approximate the oldStatusDurationGrains by looking at runs
            "oldStatusDurationGrains": self._count_final_run_length(tmp2, last_flip_row["prev_status"], flips.index[-1])
        }

    def _count_final_run_length(
        self,
        df_with_status: pd.DataFrame,
        old_status: str,
        flip_index: int
    ) -> int:
        """
        Counts how many consecutive rows prior to flip_index had the old_status.
        """
        count = 0
        # Move backward from flip_index - 1
        for i in range(flip_index - 1, -1, -1):
            if df_with_status["status"].iloc[i] == old_status:
                count += 1
            else:
                break
        return count

    def _calculate_streak_info(
        self,
        df: pd.DataFrame,
        current_status: str
    ) -> Dict[str, Any]:
        """
        Looks at the direction (increasing/decreasing/stable) of 'value' to see
        how many consecutive data points align with the latest direction.
        """
        values = df["value"].to_numpy()
        if len(values) < 2:
            return {}

        streak_length = 1
        # Determine the most recent single-step direction
        if values[-1] > values[-2]:
            direction = "increasing"
        elif values[-1] < values[-2]:
            direction = "decreasing"
        else:
            direction = "stable"

        # Extend backward
        for i in range(len(values) - 2, 0, -1):
            prev = values[i - 1]
            cur = values[i]
            if (cur > prev and direction == "increasing"):
                streak_length += 1
            elif (cur < prev and direction == "decreasing"):
                streak_length += 1
            elif (cur == prev and direction == "stable"):
                streak_length += 1
            else:
                break

        if streak_length < 2:
            return {}

        # Calculate total absolute and percentage change over that streak
        streak_start_val = values[-streak_length]
        streak_end_val = values[-1]
        absolute_change = streak_end_val - streak_start_val
        pct_change = None
        if streak_start_val != 0:
            pct_change = (absolute_change / abs(streak_start_val)) * 100

        return {
            "length": streak_length,
            "status": current_status,
            "performanceChangePercentOverStreak": pct_change,
            "absoluteChangeOverStreak": absolute_change,
            "averageChangePercentPerGrain": (pct_change / streak_length) if (pct_change is not None) else None,
            "averageChangeAbsolutePerGrain": absolute_change / streak_length
        }

    def _calculate_hold_steady_scenario(
        self,
        current_value: float,
        target_value: float
    ) -> Dict[str, Any]:
        """
        Simple 'hold steady' scenario for if the metric is already at/above target.
        This can be extended to show how many periods we can remain above target
        given a hypothetical average negative drift or the like.
        """
        if current_value < target_value:
            return {}

        # Example: If current_value is above target,
        # specify a short horizon to maintain.
        margin = None
        if target_value != 0:
            margin = ((current_value - target_value) / target_value) * 100

        return {
            "isCurrentlyAtOrAboveTarget": True,
            "timeToMaintainGrains": 3,  # Example / placeholder
            "currentMarginPercent": margin
        }
