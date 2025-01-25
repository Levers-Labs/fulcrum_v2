import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.trend import detect_anomalies

class AnomalyAnalysisPattern:
    PATTERN_NAME = "AnomalyAnalysis"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, data: pd.DataFrame, analysis_window: Dict[str, str]) -> PatternOutput:
        df = detect_anomalies(data, value_col="value")
        anomalies = df[df["final_anomaly"]].to_dict("records")
        results = {
            "num_anomalies": len(anomalies),
            "anomalies": anomalies
        }
        return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, results)
