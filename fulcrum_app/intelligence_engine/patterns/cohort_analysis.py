import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.cohort_analysis import perform_cohort_analysis

class CohortAnalysisPattern:
    """
    Analyzes retention or usage by cohort.
    Expects data with [cohort, period, value].
    """

    PATTERN_NAME = "CohortAnalysis"
    PATTERN_VERSION = "1.0"

    def run(
        self, 
        metric_id: str, 
        data: pd.DataFrame, 
        analysis_window: Dict[str, str]
    ) -> PatternOutput:
        if data.empty:
            return PatternOutput(
                self.PATTERN_NAME,
                self.PATTERN_VERSION,
                metric_id,
                analysis_window,
                {"message": "no_data"}
            )

        pivot_dict = perform_cohort_analysis(data, cohort_col="cohort", period_col="period", value_col="value")

        results = {
            "cohort_matrix": pivot_dict
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
