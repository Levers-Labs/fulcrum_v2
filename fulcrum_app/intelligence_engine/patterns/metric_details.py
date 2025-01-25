import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput

class MetricDetailsPattern:
    """
    Surfaces metadata about a metric: definition, owner, disclaimers, etc.
    Typically no heavy data needed, just DB/config references.
    """

    PATTERN_NAME = "MetricDetails"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, metadata: Dict[str, str], analysis_window: Dict[str, str]) -> PatternOutput:
        # We assume 'metadata' is a dict with 'definition', 'owner_team', 'targets', 'disclaimers' etc.
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
