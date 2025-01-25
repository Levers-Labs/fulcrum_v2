from typing import Dict
from .graph_models import MetricGraph, MetricDefinition


class GraphService:
    """
    Manages the in-memory store of the MetricGraph.
    In a production scenario, you'd persist these definitions in a DB.
    """

    def __init__(self):
        self._metric_graph: MetricGraph = MetricGraph()
        # For fast lookups, we also keep a dict: metric_id -> MetricDefinition
        self._metrics_by_id: Dict[str, MetricDefinition] = {}

    def load_metric_graph(self, graph: MetricGraph):
        """
        Overwrite current store with a new MetricGraph.
        """
        self._metric_graph = graph
        self._metrics_by_id.clear()
        for m in graph.metrics:
            self._metrics_by_id[m.id] = m

    def validate_metric_graph(self) -> bool:
        """
        Run validations:
         - Check for duplicate IDs (already handled by dictionary).
         - Check influences refer to valid metrics.
         - Potentially parse formulas to confirm references, if needed.

        Returns True if valid, otherwise raises ValueError.
        """
        self._check_duplicate_ids()
        self._check_influences_exist()
        # Optionally: self._check_formula_references()

        return True

    def _check_duplicate_ids(self):
        ids = [m.id for m in self._metric_graph.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate metric IDs found in the graph.")

    def _check_influences_exist(self):
        valid_ids = set(self._metrics_by_id.keys())
        for m in self._metric_graph.metrics:
            for inf in m.influences:
                if inf.source not in valid_ids:
                    raise ValueError(
                        f"Metric '{m.id}' references influence source "
                        f"'{inf.source}', which is not defined."
                    )

    def get_metric_by_id(self, metric_id: str) -> MetricDefinition:
        """
        Retrieve a metric definition by ID; KeyError if not found.
        """
        return self._metrics_by_id[metric_id]

    def all_metrics(self):
        """
        Returns a list of all metric definitions.
        """
        return list(self._metrics_by_id.values())

    def get_graph(self) -> MetricGraph:
        return self._metric_graph
