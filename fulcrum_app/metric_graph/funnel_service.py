from typing import Dict
from .funnel_models import FunnelCollection, FunnelDefinition

class FunnelService:
    """
    Manages in-memory store of funnels.
    Provides validation that all funnel steps refer to valid metrics in the graph.
    """

    def __init__(self):
        self._funnel_collection: FunnelCollection = FunnelCollection()
        self._funnels_by_id: Dict[str, FunnelDefinition] = {}

    def load_funnel_collection(self, collection: FunnelCollection):
        self._funnel_collection = collection
        self._funnels_by_id.clear()
        for f in collection.funnels:
            self._funnels_by_id[f.funnel_id] = f

    def validate_funnels(self, valid_metric_ids: set) -> bool:
        """
        Ensure each funnel step references a metric in valid_metric_ids.
        Raise ValueError if not.
        """
        for f in self._funnel_collection.funnels:
            for step in f.steps:
                if step.metric_id not in valid_metric_ids:
                    raise ValueError(
                        f"Funnel '{f.funnel_id}' step '{step.name}' references "
                        f"unknown metric_id '{step.metric_id}'."
                    )
        return True

    def get_funnel_by_id(self, funnel_id: str) -> FunnelDefinition:
        return self._funnels_by_id[funnel_id]

    def all_funnels(self):
        return list(self._funnels_by_id.values())
