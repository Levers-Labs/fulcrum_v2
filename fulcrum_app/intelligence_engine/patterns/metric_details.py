import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput

class MetricDetailsPattern:
    """
    Returns metadata about a metric: definition, owner, disclaimers, etc. 
    Minimal numeric analysis—just pulling from config or DB.
    """

    PATTERN_NAME = "MetricDetails"
    PATTERN_VERSION = "1.0"

    def run(
        self, 
        metric_id: str,
        metadata: Dict[str, any],
        analysis_window: Dict[str, str]
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        metadata : dict
            e.g. {
              "definition":"some text",
              "owner_team":"Finance",
              "targets": [...],
              "disclaimers": "..."
            }
        analysis_window : dict

        Returns
        -------
        PatternOutput
        """
        results = {
            "definition": metadata.get("definition", ""),
            "owner_team": metadata.get("owner_team", ""),
            "targets": metadata.get("targets", []),
            "disclaimers": metadata.get("disclaimers", "")
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
