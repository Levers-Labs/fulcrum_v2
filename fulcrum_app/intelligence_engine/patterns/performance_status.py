import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.performance import classify_metric_status

class PerformanceStatusPattern:
    """
    Checks if the metric is on_track/off_track vs. the latest known target row, or if no target => no_target.
    Typically we look at the final row in the DataFrame to determine current status.
    """

    PATTERN_NAME = "PerformanceStatus"
    PATTERN_VERSION = "1.0"

    def run(
        self, 
        metric_id: str, 
        data: pd.DataFrame, 
        analysis_window: Dict[str, str],
        threshold: float = 0.05
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        data : pd.DataFrame
            Expected columns: ['date', 'value', 'target'] (or at least 'value').
            We'll look at the final row's value & target.
        analysis_window : dict
            e.g. {"start_date":"2025-01-01","end_date":"2025-01-31"}
        threshold : float, default=0.05
            5% slack for on_track vs. off_track classification.

        Returns
        -------
        PatternOutput
            results={
              "status": "on_track"/"off_track"/"no_data"/"no_target",
              "final_value": float,
              "final_target": float or None,
              "threshold": float
            }
        """
        if data.empty:
            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results={"status": "no_data"}
            )

        # sort by date if needed
        data = data.sort_values("date")
        last_row = data.iloc[-1]
        final_val = float(last_row["value"])
        if "target" not in last_row or pd.isna(last_row["target"]):
            # no target => no_target
            return PatternOutput(
                self.PATTERN_NAME,
                self.PATTERN_VERSION,
                metric_id,
                analysis_window,
                results={
                    "status": "no_target",
                    "final_value": final_val,
                    "final_target": None,
                    "threshold": threshold
                }
            )

        final_tgt = float(last_row["target"])
        # call classify_metric_status
        status = classify_metric_status(
            row_val=final_val, 
            row_target=final_tgt, 
            threshold=threshold, 
            allow_negative_target=False
        )

        results = {
            "status": status,
            "final_value": final_val,
            "final_target": final_tgt,
            "threshold": threshold
        }
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results=results
        )
