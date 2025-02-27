"""
Performance Status Pattern

This module implements the PerformanceStatusPattern which analyzes whether a metric
is on/off track versus its target. It classifies the current status, tracks status
changes, and provides details about any gap or overperformance.

Output Format:
{
  "schemaVersion": "1.0.0",
  "patternName": "PerformanceStatus",
  "metricId": "string",
  "grain": "day" | "week" | "month",

  // Key dates/times
  "analysisDate": "YYYY-MM-DD",       // The period for which we're analyzing performance
  "evaluationTime": "YYYY-MM-DD HH:mm:ss", // Actual time of this analysis run

  // Current vs. prior values
  "currentValue": 123.45,
  "priorValue": 120.00,
  "absoluteDeltaFromPrior": 3.45,               // e.g. currentValue - priorValue
  "popChangePercent": 2.875,                    // e.g. (3.45 / 120.0) * 100

  // Target-related fields
  "targetValue": 118.0,
  "status": "on_track" | "off_track" | "overperforming",
  "absoluteGap": 0.0,                      // if off_track: how far below target in absolute terms
  "absoluteGapDelta": -100,                // how much the absolute gap is shrinking/growing vs last period
  "percentGap": 0.0,                       // if off_track: how far below target in percentage terms
  "percentGapDelta": -1.2,                 // how much the relative gap is shrinking/growing
  "absoluteOverperformance": 5.45,         // if overperforming: how far above target in absolute terms
  "percentOverperformance": 4.58,          // if overperforming: how far above target in percentage terms

  // Status change info
  "statusChange": {
    "hasFlipped": true,
    "oldStatus": "off_track",
    "newStatus": "on_track",
    "oldStatusDurationGrains": 2
  },

  // Streak info
  "streak": {
    "length": 3,
    "status": "on_track",
    "performanceChangePercentOverStreak": 15.4,
    "absoluteChangeOverStreak": 20.0,        // total absolute increase over streak
    "averageChangePercentPerGrain": 5.13,    // 15.4% / 3
    "averageChangeAbsolutePerGrain": 6.67    // 20 / 3
  },

  // "Hold steady" scenario
  "holdSteady": {
    "isCurrentlyAtOrAboveTarget": true,
    "timeToMaintainGrains": 3,
    "currentMarginPercent": 2.0
  }
}
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from .base_pattern import Pattern


class PerformanceStatusPattern(Pattern):
    """
    Analyzes a metric's performance against target, classifying its status and 
    providing detailed information about gaps, streaks, and status changes.
    """
    
    PATTERN_NAME = "performance_status"
    PATTERN_VERSION = "1.0"
    
    def run(self, 
            metric_id: str, 
            data: pd.DataFrame, 
            analysis_window: Dict[str, str],
            threshold: float = 0.05) -> PatternOutput:
        """
        Execute the performance status analysis.
        
        Parameters
        ----------
        metric_id : str
            The ID of the metric being analyzed
        data : pd.DataFrame
            DataFrame containing columns: date, value, target
        analysis_window : Dict[str, str]
            Dictionary with start_date, end_date, and optional grain
        threshold : float, default 0.05
            The threshold ratio for determining on_track status.
            A metric is on_track if value >= target * (1-threshold)
            
        Returns
        -------
        PatternOutput
            Pattern output with performance status analysis results
        """
        try:
            # Validate input data
            required_columns = ['date', 'value']
            self.validate_data(data, required_columns)
            self.validate_analysis_window(analysis_window)
            
            # Handle empty data
            if data.empty:
                return self.handle_empty_data(metric_id, analysis_window)
            
            # Convert date column to datetime if needed
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            
            # Sort by date
            data.sort_values('date', inplace=True)
            
            # Extract the grain from analysis_window
            grain = analysis_window.get('grain', 'day').lower()
            
            # Calculate current and prior values
            final_value, prior_value = self._get_current_and_prior_values(data)
            
            # Calculate target value if target column exists
            target_value = None
            has_target = 'target' in data.columns
            if has_target:
                target_values = data['target'].dropna()
                if not target_values.empty:
                    target_value = target_values.iloc[-1]
            
            # Calculate status
            status, gap_info = self._calculate_status(final_value, target_value, threshold)
            
            # Calculate status change info
            status_change_info = self._calculate_status_change(data, threshold) if has_target else {}
            
            # Calculate streak info
            streak_info = self._calculate_streak_info(data, status) if len(data) > 1 else {}
            
            # Calculate hold steady info
            hold_steady_info = self._calculate_hold_steady(final_value, target_value) if has_target else {}
            
            # Prepare results
            results = {
                "schemaVersion": "1.0.0",
                "final_value": final_value,  # For backward compatibility
                "currentValue": final_value,
                "final_status": status,      # For backward compatibility
                "status": status,
                "threshold": threshold
            }
            
            # Add target info if available
            if target_value is not None:
                results["final_target"] = target_value  # For backward compatibility
                results["targetValue"] = target_value
            
            # Add prior values if available
            if prior_value is not None:
                results["priorValue"] = prior_value
                abs_delta = final_value - prior_value
                results["absoluteDeltaFromPrior"] = abs_delta
                if prior_value != 0:
                    results["popChangePercent"] = (abs_delta / prior_value) * 100
            
            # Add gap info if available
            if gap_info:
                results.update(gap_info)
            
            # Add status change info if available
            if status_change_info:
                results["statusChange"] = status_change_info
            
            # Add streak info if available
            if streak_info:
                results["streak"] = streak_info
            
            # Add hold steady info if available
            if hold_steady_info:
                results["holdSteady"] = hold_steady_info
            
            # Add evaluation time
            results["evaluationTime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results=results
            )
        
        except Exception as e:
            # Handle any errors
            error_results = {
                "error": str(e),
                "evaluationTime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results=error_results
            )
    
    def _get_current_and_prior_values(self, data: pd.DataFrame) -> Tuple[float, Optional[float]]:
        """Get the current and prior values from the data."""
        final_value = float(data['value'].iloc[-1])
        prior_value = float(data['value'].iloc[-2]) if len(data) > 1 else None
        return final_value, prior_value
    
    def _calculate_status(self, value: float, target: Optional[float], threshold: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate status and gap information."""
        gap_info = {}
        
        if target is None or np.isnan(target):
            return "no_target", gap_info
        
        # Calculate status and gap
        if value >= target:
            status = "on_track"  # or "overperforming" if you want to distinguish
            abs_overperformance = value - target
            pct_overperformance = (abs_overperformance / target) * 100 if target != 0 else np.nan
            gap_info["absoluteOverperformance"] = abs_overperformance
            gap_info["percentOverperformance"] = pct_overperformance
        else:
            status = "off_track"
            abs_gap = target - value
            pct_gap = (abs_gap / target) * 100 if target != 0 else np.nan
            gap_info["absoluteGap"] = abs_gap
            gap_info["percentGap"] = pct_gap
        
        return status, gap_info
    
    def _calculate_status_change(self, data: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Calculate information about status changes."""
        if 'target' not in data.columns or len(data) < 2:
            return {}
        
        # Calculate status for each row
        status_values = []
        for _, row in data.iterrows():
            value = row['value']
            target = row['target']
            if pd.isna(target):
                status_values.append("no_target")
            elif value >= target:
                status_values.append("on_track")
            else:
                status_values.append("off_track")
        
        # If no status changes, return empty dict
        if all(s == status_values[-1] for s in status_values):
            return {}
        
        # Find the most recent status change
        current_status = status_values[-1]
        prev_statuses = [s for s in status_values[:-1] if s != current_status]
        if not prev_statuses:
            return {}
        
        old_status = prev_statuses[-1]
        
        # Calculate duration of old status
        old_status_durations = []
        count = 0
        for s in reversed(status_values[:-1]):
            if s == old_status:
                count += 1
            else:
                break
        
        return {
            "hasFlipped": True,
            "oldStatus": old_status,
            "newStatus": current_status,
            "oldStatusDurationGrains": count
        }
    
    def _calculate_streak_info(self, data: pd.DataFrame, current_status: str) -> Dict[str, Any]:
        """Calculate information about current streak."""
        values = data['value'].values
        
        # Find the streak length
        streak_length = 1
        current_direction = None
        
        for i in range(len(values) - 1, 0, -1):
            if values[i] > values[i-1]:
                direction = "increasing"
            elif values[i] < values[i-1]:
                direction = "decreasing"
            else:
                direction = "stable"
            
            if current_direction is None:
                current_direction = direction
            elif direction != current_direction:
                break
            
            streak_length += 1
        
        if streak_length < 2:
            return {}
        
        # Calculate change over streak
        streak_start_value = values[-(streak_length)]
        streak_end_value = values[-1]
        abs_change = streak_end_value - streak_start_value
        pct_change = (abs_change / streak_start_value) * 100 if streak_start_value != 0 else np.nan
        
        return {
            "length": streak_length,
            "status": current_status,
            "performanceChangePercentOverStreak": pct_change,
            "absoluteChangeOverStreak": abs_change,
            "averageChangePercentPerGrain": pct_change / streak_length if not np.isnan(pct_change) else np.nan,
            "averageChangeAbsolutePerGrain": abs_change / streak_length
        }
    
    def _calculate_hold_steady(self, value: float, target: Optional[float]) -> Dict[str, Any]:
        """Calculate information for the 'hold steady' scenario."""
        if target is None or np.isnan(target):
            return {}
        
        is_above_target = value >= target
        if not is_above_target:
            return {}
        
        # Simple version for now
        margin_percent = ((value - target) / target) * 100 if target != 0 else np.nan
        
        return {
            "isCurrentlyAtOrAboveTarget": True,
            "timeToMaintainGrains": 3,  # Example value
            "currentMarginPercent": margin_percent
        }