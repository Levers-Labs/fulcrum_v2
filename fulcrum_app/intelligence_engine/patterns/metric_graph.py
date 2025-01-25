from typing import Dict
from ..data_structures import PatternOutput

class MetricGraphPattern:
    """
    Explores the graph of metric relationships. 
    Typically calls a GraphService to find upstream/downstream for this metric.
    """

    PATTERN_NAME = "MetricGraph"
    PATTERN_VERSION = "1.0"

    def run(
        self, 
        metric_id: str,
        graph_service,
        analysis_window: Dict[str, str]
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        graph_service : an object that can do e.g. get_upstream_metrics(metric_id), etc.
        analysis_window : dict

        Returns
        -------
        PatternOutput
            results={
              "upstream_metrics": [...],
              "downstream_metrics": [...]
            }
        """
        # We'll assume graph_service has get_upstream_metrics, get_downstream_metrics
        # or we do it ourselves.
        ups = graph_service.get_upstream_metrics(metric_id)  # returns dict
        # If we had a get_downstream, do that or skip
        # For demonstration, let's do ups only
        results = {
            "upstream_metrics": list(ups.keys()),  # or a more complex object
            "downstream_metrics": []
        }

        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
