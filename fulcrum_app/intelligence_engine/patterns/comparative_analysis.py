import pandas as pd
from typing import Dict, List
from ..data_structures import PatternOutput
from ..primitives.comparative_analysis import compare_metrics_correlation

class ComparativeAnalysisPattern:
    PATTERN_NAME = "ComparativeAnalysis"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, df: pd.DataFrame, metrics: List[str], analysis_window: Dict[str, str]) -> PatternOutput:
        corr = compare_metrics_correlation(df, metrics)
        results = {
            "correlation_matrix": corr.to_dict()
        }
        return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, results)
