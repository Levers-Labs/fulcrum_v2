from .graph_service import GraphService
from .mock_semantic_connector import MockSemanticConnector

def sync_check_with_semantic_layer(graph_svc: GraphService, connector: MockSemanticConnector) -> bool:
    """
    Verifies that all metrics in the graph have a `member` recognized 
    by the semantic connector. Raises ValueError if not valid.
    Returns True if successful.
    """
    for metric in graph_svc.all_metrics():
        if metric.member:
            if not connector.member_exists(metric.member):
                raise ValueError(
                    f"Metric '{metric.id}' references member '{metric.member}', "
                    "which is not found in the semantic layer."
                )
        else:
            # Possibly accept or reject metrics without a 'member'
            pass
    return True
