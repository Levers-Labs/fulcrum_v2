import pytest

from fulcrum_app.metric_graph.graph_parser import parse_metric_graph_toml
from fulcrum_app.metric_graph.graph_service import GraphService

from fulcrum_app.metric_graph.funnel_parser import parse_funnel_defs_toml
from fulcrum_app.metric_graph.funnel_service import FunnelService

from fulcrum_app.metric_graph.mock_semantic_connector import MockSemanticConnector
from fulcrum_app.metric_graph.semantic_sync_service import sync_check_with_semantic_layer


def test_metric_graph_and_funnels_integration():
    metric_graph_toml = """
    [[metrics]]
    id = "metric_1"
    label = "Metric One"
    definition = "Example metric"
    unit = "n"
    owner_team = "Sales"

    [metrics.formula]
    expression_str = "{AcceptOpps} * {SQOToWinRate}"

    [[metrics.influences]]
    source = "some_other_metric"
    strength = 0.8
    confidence = 0.7

    [[metrics.dimensions]]
    id = "region"
    label = "Region"
    reference = "account_region"
    cube = "dim_opportunity"
    member_type = "dimension"

    [metrics.metadata]
    cube = "some_cube"
    member = "metric_1_member"
    member_type = "measure"
    time_dimension = "created_at"

    [[metrics]]
    id = "some_other_metric"
    label = "Some Other Metric"
    definition = "A second metric for demonstration"
    unit = "n"
    owner_team = "Marketing"

    [metrics.metadata]
    cube = "some_cube"
    member = "some_other_metric_member"
    member_type = "measure"
    time_dimension = "created_at"
    """

    funnel_toml = """
    [[funnels]]
    funnel_id = "user_onboarding"
    label = "User Onboarding Funnel"

    [[funnels.steps]]
    name = "Signup"
    metric_id = "metric_1"

    [[funnels.steps]]
    name = "Activation"
    metric_id = "some_other_metric"
    """

    # 1. Parse the metric graph
    metric_graph = parse_metric_graph_toml(metric_graph_toml)
    graph_svc = GraphService()
    graph_svc.load_metric_graph(metric_graph)

    # 2. Validate the metric graph
    assert graph_svc.validate_metric_graph() is True

    # 3. Parse funnels
    funnel_defs = parse_funnel_defs_toml(funnel_toml)
    funnel_svc = FunnelService()
    funnel_svc.load_funnel_collection(funnel_defs)

    # 4. Validate funnels (check all references exist in the metric graph)
    valid_metric_ids = set(graph_svc.get_graph().get_metric_ids())
    assert funnel_svc.validate_funnels(valid_metric_ids) is True

    # 5. Sync check with a mock semantic layer
    mock_connector = MockSemanticConnector(
        known_members=["metric_1_member", "some_other_metric_member"]
    )
    # This should pass
    assert sync_check_with_semantic_layer(graph_svc, mock_connector) is True

    # Check that we have 2 metrics
    assert len(graph_svc.all_metrics()) == 2

    # Check that we have 1 funnel with 2 steps
    all_funnels = funnel_svc.all_funnels()
    assert len(all_funnels) == 1
    onboarding = all_funnels[0]
    assert len(onboarding.steps) == 2