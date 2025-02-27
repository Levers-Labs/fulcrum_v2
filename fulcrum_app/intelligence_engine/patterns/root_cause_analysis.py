"""
Root Cause Analysis Pattern

This pattern analyzes the underlying factors contributing to a metric's change between two time periods.
It identifies top factors driving the change, which can be dimension segments, components, or influencing metrics.

The pattern produces a standardized output following this structure:
{
  "schemaVersion": "1.0.0",
  "patternName": "RootCauseAnalysis",
  "metricId": "conversion_rate",
  "grain": "week",

  "analysisDate": "2025-02-05",
  "evaluationTime": "2025-02-05 03:25:00",

  "analysisWindow": {
    "t0": {
      "startDate": "2025-01-22",
      "endDate": "2025-01-28",
      "metricValue": 5.2
    },
    "t1": {
      "startDate": "2025-01-29",
      "endDate": "2025-02-04",
      "metricValue": 4.6
    }
  },

  "metricDeltaAbsolute": -0.6,
  "metricDeltaPercent": -11.54,

  "topFactors": [
    {
      "rank": 1,
      "factorSubtype": "event_shock",
      "factorMetricName": null,
      "eventName": "Product Launch",
      "factorDimensionName": null,
      "factorSliceName": null,
      "currentValue": null,
      "priorValue": null,
      "factorChangeAbsolute": null,
      "factorChangePercent": null,
      "contributionAbsolute": -0.3,
      "contributionPercent": 50.0
    },
    {
      "rank": 2,
      "factorSubtype": "segment_performance",
      "factorMetricName": null,
      "factorDimensionName": "region",
      "factorSliceName": "North America",
      "currentValue": 6.0,
      "priorValue": 6.5,
      "factorChangeAbsolute": -0.5,
      "factorChangePercent": -7.69,
      "contributionAbsolute": -0.2,
      "contributionPercent": 33.3
    },
    {
      "rank": 3,
      "factorSubtype": "influence_value",
      "factorMetricName": "ad_spend",
      "factorDimensionName": null,
      "factorSliceName": null,
      "currentValue": 12000,
      "priorValue": 15000,
      "factorChangeAbsolute": -3000,
      "factorChangePercent": -20.0,
      "contributionAbsolute": -0.1,
      "contributionPercent": 16.7
    }
  ]
}
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from fulcrum_app.intelligence_engine.primitives.root_cause_analysis import (
    analyze_dimension_impact,
    decompose_metric_change,
    evaluate_seasonality_effect
)

class RootCauseAnalysisPattern:
    """
    Analyzes the root causes for changes in a metric between two time periods,
    identifying key factors that contributed to the difference.
    """
    
    PATTERN_NAME = "root_cause_analysis"
    PATTERN_VERSION = "1.0"
    
    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str],
        dimension_columns: Optional[List[str]] = None,
        event_data: Optional[pd.DataFrame] = None,
        influence_metrics: Optional[Dict[str, pd.DataFrame]] = None,
        formula_components: Optional[Dict[str, pd.DataFrame]] = None,
        max_factors: int = 5
    ) -> PatternOutput:
        """
        Analyze the root causes for a metric's change between T0 and T1 periods.
        
        Parameters
        ----------
        metric_id : str
            The identifier for the metric
        data : pd.DataFrame
            DataFrame with columns [date, value] for the main metric
        analysis_window : Dict[str, str]
            Dictionary with 't0_start_date', 't0_end_date', 't1_start_date', 't1_end_date' keys
        dimension_columns : Optional[List[str]], default None
            List of dimension column names in the data to analyze
        event_data : Optional[pd.DataFrame], default None
            DataFrame containing events with columns [date, event_name, event_type]
        influence_metrics : Optional[Dict[str, pd.DataFrame]], default None
            Dictionary mapping influence metric IDs to their DataFrames
        formula_components : Optional[Dict[str, pd.DataFrame]], default None
            Dictionary mapping component names to their DataFrames 
        max_factors : int, default 5
            Maximum number of top factors to return
        
        Returns
        -------
        PatternOutput
            Object containing the root cause analysis results
        """
        # Validate input data
        required_cols = ['date', 'value']
        for col in required_cols:
            if col not in data.columns:
                return self._create_empty_result(
                    metric_id, analysis_window, f"Missing required column '{col}'"
                )
        
        # Ensure data is in the right format
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # Extract time periods from analysis window
        t0_start = pd.to_datetime(analysis_window.get('t0_start_date', analysis_window.get('start_date')))
        t0_end = pd.to_datetime(analysis_window.get('t0_end_date', t0_start))
        t1_start = pd.to_datetime(analysis_window.get('t1_start_date', analysis_window.get('end_date')))
        t1_end = pd.to_datetime(analysis_window.get('t1_end_date', t1_start))
        
        grain = analysis_window.get('grain', 'day').lower()
        
        # Filter data for T0 and T1 periods
        t0_data = data[(data['date'] >= t0_start) & (data['date'] <= t0_end)]
        t1_data = data[(data['date'] >= t1_start) & (data['date'] <= t1_end)]
        
        if t0_data.empty or t1_data.empty:
            return self._create_empty_result(
                metric_id, analysis_window, "Insufficient data in one or both time periods"
            )
        
        # Calculate aggregate metric values for T0 and T1
        t0_value = t0_data['value'].mean()
        t1_value = t1_data['value'].mean()
        
        # Calculate the overall change
        absolute_delta = t1_value - t0_value
        percent_delta = (absolute_delta / abs(t0_value)) * 100 if t0_value != 0 else None
        
        # Initialize results with analysis window and overall delta
        analysis_window_result = {
            "t0": {
                "startDate": t0_start.strftime("%Y-%m-%d"),
                "endDate": t0_end.strftime("%Y-%m-%d"),
                "metricValue": t0_value
            },
            "t1": {
                "startDate": t1_start.strftime("%Y-%m-%d"),
                "endDate": t1_end.strftime("%Y-%m-%d"),
                "metricValue": t1_value
            }
        }
        
        results = {
            "analysisWindow": analysis_window_result,
            "metricDeltaAbsolute": absolute_delta,
            "metricDeltaPercent": percent_delta,
            "topFactors": []
        }
        
        # Collect all potential factors
        all_factors = []
        
        # 1. Analyze dimension impacts if dimension columns are provided
        if dimension_columns and len(dimension_columns) > 0:
            for dim_col in dimension_columns:
                if dim_col in data.columns:
                    try:
                        dim_impact = analyze_dimension_impact(
                            df_t0=t0_data,
                            df_t1=t1_data,
                            slice_col=dim_col,
                            value_col='value'
                        )
                        
                        # Process each dimension slice
                        for _, row in dim_impact.iterrows():
                            if abs(row['delta']) > 0:
                                factor = {
                                    "factorSubtype": "segment_performance",
                                    "factorDimensionName": dim_col,
                                    "factorSliceName": row[dim_col],
                                    "currentValue": row[f'value_t1'],
                                    "priorValue": row[f'value_t0'],
                                    "factorChangeAbsolute": row['delta'],
                                    "factorChangePercent": (row['delta'] / abs(row[f'value_t0'])) * 100 if row[f'value_t0'] != 0 else None,
                                    "contributionAbsolute": row['delta'] * (row['pct_of_total_delta'] / 100) if 'pct_of_total_delta' in row else row['delta'],
                                    "contributionPercent": row['pct_of_total_delta'] if 'pct_of_total_delta' in row else None
                                }
                                all_factors.append(factor)
                    except Exception as e:
                        # Log error but continue with other dimensions
                        print(f"Error analyzing dimension {dim_col}: {str(e)}")
        
        # 2. Analyze formula components if provided
        if formula_components:
            component_factors = []
            for comp_name, comp_data in formula_components.items():
                if 'date' in comp_data.columns and 'value' in comp_data.columns:
                    try:
                        # Filter component data for the same periods
                        comp_t0 = comp_data[(comp_data['date'] >= t0_start) & (comp_data['date'] <= t0_end)]
                        comp_t1 = comp_data[(comp_data['date'] >= t1_start) & (comp_data['date'] <= t1_end)]
                        
                        if not comp_t0.empty and not comp_t1.empty:
                            comp_t0_val = comp_t0['value'].mean()
                            comp_t1_val = comp_t1['value'].mean()
                            comp_delta = comp_t1_val - comp_t0_val
                            
                            component_factors.append({
                                "name": comp_name,
                                "delta": comp_delta
                            })
                    except Exception as e:
                        print(f"Error analyzing component {comp_name}: {str(e)}")
            
            if component_factors:
                try:
                    # Use decompose_metric_change to allocate impact across components
                    decomposition = decompose_metric_change(t0_value, t1_value, component_factors)
                    
                    for comp in decomposition.get('factors', []):
                        factor = {
                            "factorSubtype": "component_value",
                            "factorMetricName": comp['name'],
                            "factorChangeAbsolute": comp['delta'],
                            "factorChangePercent": None,  # Would need component values to calculate
                            "contributionAbsolute": comp.get('contribution_abs', comp['delta']),
                            "contributionPercent": comp.get('contribution_pct')
                        }
                        all_factors.append(factor)
                except Exception as e:
                    print(f"Error decomposing metric change: {str(e)}")
        
        # 3. Analyze influence metrics if provided
        if influence_metrics:
            for infl_name, infl_data in influence_metrics.items():
                if 'date' in infl_data.columns and 'value' in infl_data.columns:
                    try:
                        # Filter influence data for the same periods
                        infl_t0 = infl_data[(infl_data['date'] >= t0_start) & (infl_data['date'] <= t0_end)]
                        infl_t1 = infl_data[(infl_data['date'] >= t1_start) & (infl_data['date'] <= t1_end)]
                        
                        if not infl_t0.empty and not infl_t1.empty:
                            infl_t0_val = infl_t0['value'].mean()
                            infl_t1_val = infl_t1['value'].mean()
                            infl_delta = infl_t1_val - infl_t0_val
                            infl_pct_delta = (infl_delta / abs(infl_t0_val)) * 100 if infl_t0_val != 0 else None
                            
                            # Estimate contribution based on correlation or other heuristic
                            # Here we use a simple heuristic: 20% impact if direction matches
                            contribution_sign = 1 if (infl_delta * absolute_delta) > 0 else -1
                            contribution_pct = 20.0 * contribution_sign
                            contribution_abs = absolute_delta * (contribution_pct / 100)
                            
                            factor = {
                                "factorSubtype": "influence_value",
                                "factorMetricName": infl_name,
                                "currentValue": infl_t1_val,
                                "priorValue": infl_t0_val,
                                "factorChangeAbsolute": infl_delta,
                                "factorChangePercent": infl_pct_delta,
                                "contributionAbsolute": contribution_abs,
                                "contributionPercent": contribution_pct
                            }
                            all_factors.append(factor)
                    except Exception as e:
                        print(f"Error analyzing influence metric {infl_name}: {str(e)}")
        
        # 4. Analyze events if provided
        if event_data is not None and not event_data.empty:
            if 'date' in event_data.columns and 'event_name' in event_data.columns:
                # Look for events near the t0->t1 transition
                event_window_start = t0_end - pd.Timedelta(days=3)
                event_window_end = t1_start + pd.Timedelta(days=3)
                
                nearby_events = event_data[
                    (event_data['date'] >= event_window_start) & 
                    (event_data['date'] <= event_window_end)
                ]
                
                for _, event in nearby_events.iterrows():
                    # Estimate impact as 30% of total delta (simplified heuristic)
                    contribution_pct = 30.0
                    contribution_abs = absolute_delta * (contribution_pct / 100)
                    
                    factor = {
                        "factorSubtype": "event_shock",
                        "eventName": event['event_name'],
                        "factorMetricName": None,
                        "factorDimensionName": None,
                        "factorSliceName": None,
                        "currentValue": None,
                        "priorValue": None,
                        "factorChangeAbsolute": None,
                        "factorChangePercent": None,
                        "contributionAbsolute": contribution_abs,
                        "contributionPercent": contribution_pct
                    }
                    all_factors.append(factor)
        
        # 5. Consider seasonality impact if there's enough data
        if len(data) >= 30:  # Need enough data for seasonal analysis
            try:
                seasonality_effect = evaluate_seasonality_effect(
                    data, 
                    date_col='date', 
                    value_col='value',
                    period=7  # Weekly seasonality as default
                )
                
                seasonal_impact = seasonality_effect.get('seasonal_diff')
                if seasonal_impact is not None:
                    seasonal_fraction = seasonality_effect.get('fraction_of_total_diff', 0)
                    
                    factor = {
                        "factorSubtype": "seasonal_effect",
                        "factorMetricName": None,
                        "factorDimensionName": None,
                        "factorSliceName": None,
                        "currentValue": None,
                        "priorValue": None,
                        "factorChangeAbsolute": seasonal_impact,
                        "factorChangePercent": None,
                        "contributionAbsolute": seasonal_impact,
                        "contributionPercent": seasonal_fraction * 100
                    }
                    all_factors.append(factor)
            except Exception as e:
                print(f"Error analyzing seasonality: {str(e)}")
        
        # Sort factors by absolute contribution and limit to max_factors
        if all_factors:
            sorted_factors = sorted(
                all_factors,
                key=lambda x: abs(x.get('contributionAbsolute', 0) or 0),
                reverse=True
            )
            
            # Add rank and prepare top factors
            top_factors = []
            for idx, factor in enumerate(sorted_factors[:max_factors]):
                factor['rank'] = idx + 1
                top_factors.append(factor)
            
            results['topFactors'] = top_factors
        
        # Create and return pattern output
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results=results
        )
    
    def _create_empty_result(
        self, 
        metric_id: str, 
        analysis_window: Dict[str, str],
        error_message: str = "Insufficient data for analysis"
    ) -> PatternOutput:
        """Create an empty result with an error message."""
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results={
                "analysisWindow": {
                    "t0": {
                        "startDate": analysis_window.get('t0_start_date', ""),
                        "endDate": analysis_window.get('t0_end_date', ""),
                        "metricValue": None
                    },
                    "t1": {
                        "startDate": analysis_window.get('t1_start_date', ""),
                        "endDate": analysis_window.get('t1_end_date', ""),
                        "metricValue": None
                    }
                },
                "metricDeltaAbsolute": None,
                "metricDeltaPercent": None,
                "error": error_message,
                "topFactors": []
            }
        )