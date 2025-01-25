import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.trend import detect_anomalies

class AnomalyAnalysisPattern:
    """
    Combines multiple anomaly detection methods (variance, SPC, ML) 
    and returns flagged points.
    """

    PATTERN_NAME = "AnomalyAnalysis"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str]
    ) -> PatternOutput:
        """
        We call detect_anomalies and store the list of anomaly points.
        """
        if data.empty:
            return PatternOutput(self.PATTERN_NAME,self.PATTERN_VERSION,metric_id,analysis_window,{"message":"no_data"})

        df_anno = detect_anomalies(data, value_col="value")
        anomalies = df_anno[df_anno["final_anomaly"]].to_dict("records")
        results = {
            "num_anomalies": len(anomalies),
            "anomalies": anomalies
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
