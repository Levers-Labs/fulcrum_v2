import toml
from typing import Any, Dict
from .graph_models import (
    MetricGraph,
    MetricDefinition,
    InfluenceDefinition,
    DimensionDefinition,
    FormulaDefinition
)
from .formula_parser import parse_formula_references

def parse_metric_graph_toml(toml_str: str) -> MetricGraph:
    data = toml.loads(toml_str)
    raw_metrics = data.get("metrics", [])

    metric_defs = []
    for raw_m in raw_metrics:
        metric_defs.append(_parse_single_metric(raw_m))

    return MetricGraph(metrics=metric_defs)

def _parse_single_metric(raw_m: Dict[str, Any]) -> MetricDefinition:
    metric_id = raw_m["id"]
    label = raw_m.get("label", metric_id)
    definition = raw_m.get("definition", "")
    unit = raw_m.get("unit", "n")
    owner_team = raw_m.get("owner_team", "Unknown")

    # Formula
    formula_data = raw_m.get("formula")
    formula = None
    formula_references = []
    if formula_data and "expression_str" in formula_data:
        expr = formula_data["expression_str"]
        formula = FormulaDefinition(expression_str=expr)
        formula_references = parse_formula_references(expr)

    # Influences
    influences_list = []
    for inf in raw_m.get("influences", []):
        influences_list.append(
            InfluenceDefinition(
                source=inf["source"],
                strength=inf["strength"],
                confidence=inf["confidence"]
            )
        )

    # Dimensions
    dims_list = []
    for dim in raw_m.get("dimensions", []):
        dims_list.append(
            DimensionDefinition(
                id=dim["id"],
                label=dim["label"],
                reference=dim["reference"],
                cube=dim["cube"],
                member_type=dim["member_type"]
            )
        )

    # Metadata
    metadata = raw_m.get("metadata", {})
    cube = metadata.get("cube")
    member = metadata.get("member")
    member_type = metadata.get("member_type")
    time_dimension = metadata.get("time_dimension")

    return MetricDefinition(
        id=metric_id,
        label=label,
        definition=definition,
        unit=unit,
        owner_team=owner_team,
        formula=formula,
        formula_references=formula_references,  # <-- new field
        influences=influences_list,
        dimensions=dims_list,
        cube=cube,
        member=member,
        member_type=member_type,
        time_dimension=time_dimension
    )