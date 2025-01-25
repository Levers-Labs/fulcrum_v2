import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.root_cause import quantify_event_impact

class EventAnalysisPattern:
    """
    Maps external events to metric changes. 
    Might do a pre/post analysis around each event date.
    """

    PATTERN_NAME = "EventAnalysis"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        events: pd.DataFrame,
        analysis_window: Dict[str, str],
        window_before: int = 7,
        window_after: int = 7
    ) -> PatternOutput:
        """
        For each event in events, do a naive pre/post approach using quantify_event_impact.
        returns a list of results.
        """
        if events.empty:
            results = {"event_impacts": []}
        else:
            impact_df = quantify_event_impact(df=data, event_df=events, date_col="date", value_col="value", 
                                              window_before=window_before, window_after=window_after)
            results = {
                "event_impacts": impact_df.to_dict("records")
            }

        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
