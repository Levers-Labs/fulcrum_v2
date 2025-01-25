import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.performance import classify_metric_status

class PerformanceStatusPattern:
    PATTERN_NAME = "PerformanceStatus"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, data: pd.DataFrame, analysis_window: Dict[str, str], threshold=0.05) -> PatternOutput:
        # We'll assume data has columns: ["date", "value", "target"]
        if data.empty:
            return PatternOutput(
                pattern_name=self.PATTERN_NAME,
                pattern_version=self.PATTERN_VERSION,
                metric_id=metric_id,
                analysis_window=analysis_window,
                results={"status": "no_data"}
            )

        last_row = data.iloc[-1]
        status = classify_metric_status(last_row["value"], last_row["target"], threshold)
        results = {
            "status": status,
            "final_value": float(last_row["value"]),
            "final_target": float(last_row["target"]),
            "threshold": threshold
        }
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results=results
        )
