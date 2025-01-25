import pandas as pd
from typing import Dict, List
from ..data_structures import PatternOutput
from ..primitives.comparative_analysis import compare_metrics_correlation

class ComparativeAnalysisPattern:
    """
    Compares multiple metrics for correlation or synergy.
    """

    PATTERN_NAME = "ComparativeAnalysis"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        df: pd.DataFrame,
        metrics: List[str],
        analysis_window: Dict[str, str],
        method: str = "pearson"
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        df : pd.DataFrame
            Should have columns in metrics list, e.g. [metricA, metricB, metricC].
        metrics : list of str
        analysis_window : dict
        method : str, default='pearson'

        Returns
        -------
        PatternOutput
            results={
               "correlation_matrix": <df.to_dict()> 
            }
        """
        corr = compare_metrics_correlation(df, metrics, method=method)
        results = {
            "correlation_matrix": corr.to_dict()
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
