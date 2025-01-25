import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.descriptive_stats import calculate_descriptive_stats
from ..primitives.trend import detect_anomaly_with_variance

class DataQualityPattern:
    """
    Checks for missing data, outlier counts, or suspicious volatility. 
    """

    PATTERN_NAME = "DataQuality"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str],
        window: int = 7,
        z_thresh: float = 3.0
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        data : pd.DataFrame
            Must have ['date','value'] presumably. 
        analysis_window : dict
        window : int, default=7
        z_thresh : float, default=3.0

        Returns
        -------
        PatternOutput
            results={
               "descriptive_stats": {...},
               "anomaly_count": int,
               "data_completeness": "XX rows"
            }
        """
        if data.empty:
            return PatternOutput(self.PATTERN_NAME,self.PATTERN_VERSION,metric_id,analysis_window,{"status":"no_data"})

        stats = calculate_descriptive_stats(data, value_col="value")
        anomaly_df = detect_anomaly_with_variance(data, value_col="value", window=window, z_thresh=z_thresh)
        anomalies = anomaly_df[anomaly_df["is_anomaly"]].shape[0]

        results = {
            "descriptive_stats": stats,
            "anomaly_count": anomalies,
            "data_completeness": f"{len(data)} rows"
        }
        return PatternOutput(self.PATTERN_NAME,self.PATTERN_VERSION,metric_id,analysis_window,results)
