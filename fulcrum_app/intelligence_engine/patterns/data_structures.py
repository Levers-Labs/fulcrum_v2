from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PatternOutput:
    """
    A generic container for the result of a Pattern.
    """
    pattern_name: str
    pattern_version: str
    metric_id: str
    analysis_window: Dict[str, str]  # e.g. {"start_date": "...", "end_date": "..."}
    results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this object to a Python dict (e.g., for JSON serialization).
        """
        return {
            "pattern_name": self.pattern_name,
            "pattern_version": self.pattern_version,
            "metric_id": self.metric_id,
            "analysis_window": self.analysis_window,
            "results": self.results
        }
