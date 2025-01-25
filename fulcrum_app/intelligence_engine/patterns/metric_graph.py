from typing import Dict
from ..data_structures import PatternOutput

class MetricGraphPattern:
    """
    Explores the graph of metric relationships. Possibly calls the GraphService.
    """
    PATTERN_NAME = "MetricGraph"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, graph_info: Dict, analysis_window: Dict[str, str]) -> PatternOutput:
        # 'graph_info' might contain upstream/downstream relationships. 
        # This is a stub.
        results = {
            "upstream_metrics": graph_info.get("upstream", []),
            "downstream_metrics": graph_info.get("downstream", [])
        }
        return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, results)
