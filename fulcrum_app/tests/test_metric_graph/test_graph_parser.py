import pytest
from fulcrum_app.metric_graph.graph_parser import parse_metric_graph_toml
from fulcrum_app.metric_graph.graph_service import GraphService


def test_parse_simple_metric_graph():
    # A minimal TOML snippet with two metrics
    toml_str = """
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
    id = "metric_2"
    label = "Metric Two"
    definition = "Another example metric"
    unit = "n"
    owner_team = "Marketing"
    """

    # Parse into a MetricGraph
    graph = parse_metric_graph_toml(toml_str)

    # Create a GraphService instance and load it
    service = GraphService()
    service.load_metric_graph(graph)

    # Validate
    assert service.validate_metric_graph() is True

    # We should now have 2 metrics
    all_m = service.all_metrics()
    assert len(all_m) == 2

    # Check that the first metric has a formula
    metric_1 = service.get_metric_by_id("metric_1")
    assert metric_1.label == "Metric One"
    assert metric_1.formula is not None
    assert metric_1.formula.expression_str == "{AcceptOpps} * {SQOToWinRate}"

    # Check influences
    assert len(metric_1.influences) == 1
    inf = metric_1.influences[0]
    assert inf.source == "some_other_metric"
    assert inf.strength == 0.8
    assert inf.confidence == 0.7

    # Check second metric
    metric_2 = service.get_metric_by_id("metric_2")
    assert metric_2.label == "Metric Two"