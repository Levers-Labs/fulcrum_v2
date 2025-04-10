"""
Root Cause Analysis Pattern

Analyzes the underlying factors contributing to a metric's change between two time periods.
Identifies top drivers for the difference, including dimension slices, component metrics, 
seasonality effects, and external influences or events.

Output Example:
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
    ...
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
    Analyzes root causes for changes in a metric between two time periods,
    identifying factors (dimensions, components, external influences, etc.).
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
        Analyze the root causes for a metric's change between T0 and T1.

        Parameters
        ----------
        metric_id : str
            The identifier for the metric.
        data : pd.DataFrame
            Must contain at least ['date', 'value'] columns for the main metric.
        analysis_window : Dict[str, str]
            Should include 't0_start_date', 't0_end_date', 't1_start_date', 't1_end_date'.
        dimension_columns : Optional[List[str]]
            Names of dimension columns to analyze.
        event_data : Optional[pd.DataFrame]
            DataFrame of events with at least ['date', 'event_name'] columns.
        influence_metrics : Optional[Dict[str, pd.DataFrame]]
            Mapping of influence metric IDs to DataFrames with ['date', 'value'].
        formula_components : Optional[Dict[str, pd.DataFrame]]
            Mapping of sub-component names to DataFrames with ['date', 'value'].
        max_factors : int, default 5
            Maximum number of top factors to return.

        Returns
        -------
        PatternOutput
            Contains the root-cause analysis results.
        """
        if not self._validate_main_data(data):
            return self._create_empty_result(
                metric_id,
                analysis_window,
                "DataFrame missing required columns ['date', 'value']"
            )

        data = data.copy()
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

        # Extract time periods
        t0_start, t0_end, t1_start, t1_end, grain = self._extract_time_periods(analysis_window)
        t0_data = data[(data['date'] >= t0_start) & (data['date'] <= t0_end)]
        t1_data = data[(data['date'] >= t1_start) & (data['date'] <= t1_end)]

        # Check data availability
        if t0_data.empty or t1_data.empty:
            return self._create_empty_result(
                metric_id,
                analysis_window,
                "Insufficient data in one or both time periods"
            )

        # Compute overall change
        t0_value = t0_data['value'].mean()
        t1_value = t1_data['value'].mean()
        absolute_delta = t1_value - t0_value
        percent_delta = (
            (absolute_delta / abs(t0_value)) * 100
            if t0_value != 0 else None
        )

        # Prepare top-level results
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

        all_factors = []

        # 1. Dimension-based factor analysis
        if dimension_columns:
            for dim_col in dimension_columns:
                if dim_col in data.columns:
                    try:
                        dim_impact_df = analyze_dimension_impact(
                            df_t0=t0_data,
                            df_t1=t1_data,
                            dimension_col=dim_col,
                            value_col='value'
                        )
                        # The returned df has columns:
                        # [dim_col, 'valT0', 'valT1', 'delta', 'share_of_total_delta']
                        for _, row in dim_impact_df.iterrows():
                            if abs(row['delta']) > 1e-12:
                                prior_val = row.get('valT0')
                                curr_val = row.get('valT1')
                                delta_abs = row['delta']
                                delta_pct = (
                                    (delta_abs / abs(prior_val)) * 100
                                    if prior_val and abs(prior_val) > 1e-12 else None
                                )
                                share_pct = row.get('share_of_total_delta')
                                factor = {
                                    "factorSubtype": "segment_performance",
                                    "factorDimensionName": dim_col,
                                    "factorSliceName": row[dim_col],
                                    "currentValue": curr_val,
                                    "priorValue": prior_val,
                                    "factorChangeAbsolute": delta_abs,
                                    "factorChangePercent": delta_pct,
                                    "contributionAbsolute": (
                                        delta_abs * (share_pct / 100.0)
                                        if share_pct is not None else delta_abs
                                    ),
                                    "contributionPercent": share_pct
                                }
                                all_factors.append(factor)
                    except Exception as e:
                        print(f"Dimension analysis error on '{dim_col}': {e}")

        # 2. Formula component analysis
        if formula_components:
            try:
                component_factors = []
                for comp_name, comp_df in formula_components.items():
                    if not self._validate_main_data(comp_df):
                        continue
                    comp_df = comp_df.copy()
                    comp_df['date'] = pd.to_datetime(comp_df['date'], errors='coerce')

                    comp_t0 = comp_df[(comp_df['date'] >= t0_start) & (comp_df['date'] <= t0_end)]
                    comp_t1 = comp_df[(comp_df['date'] >= t1_start) & (comp_df['date'] <= t1_end)]

                    if comp_t0.empty or comp_t1.empty:
                        continue

                    comp_t0_val = comp_t0['value'].mean()
                    comp_t1_val = comp_t1['value'].mean()
                    comp_delta = comp_t1_val - comp_t0_val

                    component_factors.append({
                        "name": comp_name,
                        "delta": comp_delta
                    })

                if component_factors:
                    # decompose_metric_change returns a DataFrame of columns:
                    # [factor, delta, contribution_absolute, contribution_percent]
                    decomp_df = decompose_metric_change(t0_value, t1_value, component_factors)
                    for _, comp_row in decomp_df.iterrows():
                        factor = {
                            "factorSubtype": "component_value",
                            "factorMetricName": comp_row['factor'],
                            "factorChangeAbsolute": comp_row['delta'],
                            "factorChangePercent": None,  # if needed, user can calculate
                            "contributionAbsolute": comp_row['contribution_absolute'],
                            "contributionPercent": comp_row['contribution_percent']
                        }
                        all_factors.append(factor)
            except Exception as e:
                print(f"Formula component analysis error: {e}")

        # 3. Influence metrics analysis
        if influence_metrics:
            for infl_name, infl_df in influence_metrics.items():
                if not self._validate_main_data(infl_df):
                    continue
                infl_df = infl_df.copy()
                infl_df['date'] = pd.to_datetime(infl_df['date'], errors='coerce')

                infl_t0 = infl_df[(infl_df['date'] >= t0_start) & (infl_df['date'] <= t0_end)]
                infl_t1 = infl_df[(infl_df['date'] >= t1_start) & (infl_df['date'] <= t1_end)]

                if infl_t0.empty or infl_t1.empty:
                    continue

                infl_t0_val = infl_t0['value'].mean()
                infl_t1_val = infl_t1['value'].mean()
                infl_delta = infl_t1_val - infl_t0_val
                if abs(infl_t0_val) > 1e-12:
                    infl_pct_delta = (infl_delta / abs(infl_t0_val)) * 100
                else:
                    infl_pct_delta = None

                # Simple heuristic for contribution:
                # Positive correlation => 20% of main delta, negative => -20%.
                # This can be replaced with more rigorous methods.
                sign_match = (infl_delta * absolute_delta) > 0
                contribution_sign = 1 if sign_match else -1
                contribution_pct = 20.0 * contribution_sign
                contribution_abs = absolute_delta * (contribution_pct / 100.0)

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

        # 4. Event-based factors
        if event_data is not None and not event_data.empty:
            if 'date' in event_data.columns and 'event_name' in event_data.columns:
                # Consider events near the boundary between T0 and T1
                event_window_start = t0_end - pd.Timedelta(days=3)
                event_window_end = t1_start + pd.Timedelta(days=3)

                events_sub = event_data[
                    (event_data['date'] >= event_window_start) &
                    (event_data['date'] <= event_window_end)
                ]

                for _, ev_row in events_sub.iterrows():
                    # Heuristic: 30% of total delta
                    contribution_pct = 30.0
                    contribution_abs = absolute_delta * (contribution_pct / 100.0)
                    factor = {
                        "factorSubtype": "event_shock",
                        "eventName": ev_row['event_name'],
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

        # 5. Seasonality effect
        if len(data) >= 30:
            try:
                seasonality_info = evaluate_seasonality_effect(
                    df=data,
                    date_col='date',
                    value_col='value',
                    period=7  # weekly cycle
                )
                # Suppose evaluate_seasonality_effect returns {"seasonal_fraction": 0.3}
                # meaning 30% of net change is from seasonality.
                seasonal_fraction = seasonality_info.get('seasonal_fraction', 0)
                if abs(seasonal_fraction) > 1e-12:
                    seasonal_contribution_abs = absolute_delta * seasonal_fraction
                    factor = {
                        "factorSubtype": "seasonal_effect",
                        "factorMetricName": None,
                        "factorDimensionName": None,
                        "factorSliceName": None,
                        "currentValue": None,
                        "priorValue": None,
                        "factorChangeAbsolute": seasonal_contribution_abs,
                        "factorChangePercent": None,
                        "contributionAbsolute": seasonal_contribution_abs,
                        "contributionPercent": seasonal_fraction * 100
                    }
                    all_factors.append(factor)
            except Exception as e:
                print(f"Seasonality analysis error: {e}")

        # Sort & select top factors
        if all_factors:
            sorted_factors = sorted(
                all_factors,
                key=lambda x: abs(x.get('contributionAbsolute', 0) or 0),
                reverse=True
            )
            top_factors = []
            for idx, factor_data in enumerate(sorted_factors[:max_factors]):
                factor_data['rank'] = idx + 1
                top_factors.append(factor_data)
            results['topFactors'] = top_factors

        # Finalize output
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results=results
        )

    def _validate_main_data(self, df: pd.DataFrame) -> bool:
        """Checks if df has at least ['date', 'value'] columns."""
        if not isinstance(df, pd.DataFrame):
            return False
        return all(col in df.columns for col in ['date', 'value'])

    def _extract_time_periods(self, analysis_window: Dict[str, str]) -> Tuple[pd.Timestamp, pd.Timestamp,
                                                                             pd.Timestamp, pd.Timestamp, str]:
        """Extracts T0/T1 time boundaries from the analysis_window."""
        t0_start = pd.to_datetime(analysis_window.get('t0_start_date', analysis_window.get('start_date', None)))
        t0_end = pd.to_datetime(analysis_window.get('t0_end_date', t0_start))
        t1_start = pd.to_datetime(analysis_window.get('t1_start_date', analysis_window.get('end_date', None)))
        t1_end = pd.to_datetime(analysis_window.get('t1_end_date', t1_start))
        grain = analysis_window.get('grain', 'day').lower()
        return t0_start, t0_end, t1_start, t1_end, grain

    def _create_empty_result(
        self,
        metric_id: str,
        analysis_window: Dict[str, str],
        error_message: str = "Insufficient data for analysis"
    ) -> PatternOutput:
        """Returns an empty result payload with an error."""
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
