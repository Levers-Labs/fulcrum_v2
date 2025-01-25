import pandas as pd
from typing import Dict, List
from ..data_structures import PatternOutput
from ..primitives.funnel_analysis import perform_funnel_analysis

class FunnelAnalysisPattern:
    """
    Breaks data into funnel steps (Signup -> Activate -> Subscribe).
    Expects data with columns [funnel_step, value].
    """

    PATTERN_NAME = "FunnelAnalysis"
    PATTERN_VERSION = "1.0"

    def run(
        self, 
        metric_id: str, 
        data: pd.DataFrame, 
        analysis_window: Dict[str, str],
        steps: List[str]
    ) -> PatternOutput:
        # steps = ["Signup","Activated","Upgraded","Converted"] etc.
        if data.empty:
            return PatternOutput(
                self.PATTERN_NAME, 
                self.PATTERN_VERSION, 
                metric_id, 
                analysis_window, 
                {"message": "no_data"}
            )
        
        funnel_results = perform_funnel_analysis(data, steps, step_col="funnel_step", value_col="value")

        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            {"funnel_steps": funnel_results}
        )
