import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.cohort_analysis import perform_cohort_analysis

class CohortAnalysisPattern:
    """
    Analyzes retention or usage by cohort, generating a pivot table: 
    row=cohort_label, col=period_index, value= measure
    """

    PATTERN_NAME = "CohortAnalysis"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        df: pd.DataFrame,
        analysis_window: Dict[str, str],
        entity_id_col: str = "user_id",
        signup_date_col: str = "signup_date",
        activity_date_col: str = "activity_date",
        time_grain: str = "M",
        max_periods: int = 12,
        measure_col: str = None,
        measure_method: str = "count"
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        df : pd.DataFrame
          Must have [entity_id_col, signup_date_col, activity_date_col, (optionally measure_col)].
        analysis_window : dict
        entity_id_col : str, default='user_id'
        signup_date_col : str, default='signup_date'
        activity_date_col : str, default='activity_date'
        time_grain : str, default='M'
        max_periods : int, default=12
        measure_col : str or None
        measure_method : str

        Returns
        -------
        PatternOutput
            results={
              "cohort_matrix": pivot in dictionary form
            }
        """
        if df.empty:
            return PatternOutput(self.PATTERN_NAME,self.PATTERN_VERSION,metric_id,analysis_window,{"message":"no_data"})

        pivot_df = perform_cohort_analysis(
            df, 
            entity_id_col=entity_id_col,
            signup_date_col=signup_date_col,
            activity_date_col=activity_date_col,
            time_grain=time_grain,
            max_periods=max_periods,
            measure_col=measure_col,
            measure_method=measure_method
        )
        # Convert pivot table to e.g. a dictionary
        pivot_dict = pivot_df.to_dict(orient="split")  
        # or you can do pivot_df.to_dict() but "split" organizes columns/ data separately

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
