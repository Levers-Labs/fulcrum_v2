from typing import Dict
from .graph_models import MetricGraph, MetricDefinition
from fulcrum_app.core.exceptions import InvalidMetricReference

class GraphService:
    def __init__(self):
        self._metric_graph: MetricGraph = MetricGraph()
        self._metrics_by_id: Dict[str, MetricDefinition] = {}

    def load_metric_graph(self, graph: MetricGraph):
        self._metric_graph = graph
        self._metrics_by_id.clear()
        for m in graph.metrics:
            self._metrics_by_id[m.id] = m

    def validate_metric_graph(self) -> bool:
        self._check_duplicate_ids()
        self._check_influences_exist()
        self._check_formula_references_exist()
        return True

    def _check_duplicate_ids(self):
        ids = [m.id for m in self._metric_graph.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate metric IDs found.")

    def _check_influences_exist(self):
        valid_ids = set(self._metrics_by_id.keys())
        for m in self._metric_graph.metrics:
            for inf in m.influences:
                if inf.source not in valid_ids:
                    raise InvalidMetricReference(
                        f"{m.id} references influence '{inf.source}' not in graph."
                    )

    def _check_formula_references_exist(self):
        valid_ids = set(self._metrics_by_id.keys())
        for m in self._metric_graph.metrics:
            for ref in m.formula_references:
                if ref not in valid_ids:
                    # If your formulas might reference *dimensions* or 
                    # other special placeholders, you'd handle that logic here.
                    raise InvalidMetricReference(
                        f"Metric '{m.id}' formula references '{ref}' not in graph."
                    )

    # Example convenience methods:
    def get_metric_by_id(self, metric_id: str) -> MetricDefinition:
        return self._metrics_by_id[metric_id]

    def all_metrics(self):
        return list(self._metrics_by_id.values())

    def get_graph(self) -> MetricGraph:
        return self._metric_graph

    # Optionally, a simple BFS for "what metrics does this metric depend on?"
    def get_upstream_metrics(self, metric_id: str) -> Dict[str, bool]:
        """
        Returns a dict of {upstream_metric_id: True} for all upstream references 
        in formulas or influences.
        """
        visited = set()
        stack = [metric_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            mdef = self._metrics_by_id[current]
            # combine influences + formula references for adjacency
            adjacent_ids = [inf.source for inf in mdef.influences] + mdef.formula_references
            for adj in adjacent_ids:
                if adj not in visited:
                    stack.append(adj)
        # remove the root (because the question is "which metrics feed into me?")
        visited.remove(metric_id)
        return {u: True for u in visited}