# file: component_drift_svc.py

import copy
import logging
from typing import Dict, Any, Optional, List

# We assume you have the new/merged component_analysis in your codebase:
# from .component_analysis import advanced_component_drift

logger = logging.getLogger(__name__)


def calculate_recursive_component_drift(
    metric_expression: Dict[str, Any],
    parent_drift: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Recursively compute component drift for a metric expression and all its children.

    This function:
      1) Calls `advanced_component_drift` (from your merged component_analysis.py)
         on the current metric_expression to get top-level drift info.
      2) Recursively processes each sub-component if they themselves have a nested expression.
      3) Returns a combined dictionary: 
         {
            "metric_id": str,
            "evaluation_value": float,
            "comparison_value": float,
            "drift": {...fields from advanced_component_drift...},
            "components": [ ... sub-components ... ]
         }

    Assumptions:
      - `metric_expression` has the shape:
         {
           "metric_id": "CAC",
           "type": "expression",  # or "metric"
           "operator": "/",
           "operands": [
              {
                "type": "metric",
                "metric_id": "SalesSpend",
                "evaluation_value": 100.0,
                "comparison_value": 90.0
              },
              ...
           ]
         }
      - Each leaf node of type "metric" already has `evaluation_value` and `comparison_value`.
      - If a node also has "expression", we handle sub-operands recursively.
      - `parent_drift` is an optional dictionary describing the parent's drift if you want
        to chain old code logic (like relative_impact_root). 
        Typically, we can leave it None for the top-most expression.

    Returns
    -------
    Dict[str, Any]
        A structure like:
          {
            "metric_id": "CAC",
            "evaluation_value": 120.0,
            "comparison_value": 100.0,
            "drift": {
              "absolute_drift": 20.0,
              "percentage_drift": 20.0,
              "relative_impact": 1.0,
              "marginal_contribution": 1.0,
              "relative_impact_root": 1.0,
              "marginal_contribution_root": 1.0
            },
            "components": [
               {
                 "metric_id": "SalesSpend",
                 "evaluation_value": 80.0,
                 "comparison_value": 70.0,
                 "drift": {...},
                 "components": [ ... ],
               },
               ...
            ]
          }
    """
    from .component_analysis import advanced_component_drift  # lazy import

    # 1) Make a deep copy so we don't mutate the input.
    expr_copy = copy.deepcopy(metric_expression)

    # 2) If the user wants to chain parent's drift into this node, they can do so:
    #    advanced_component_drift(...) itself doesn't need a "parent drift" param,
    #    because it computes from the top of that expression. 
    #    If you'd like to incorporate parent's drift into the root node's relative_impact,
    #    you'd have to do that manually. Typically you can let advanced_component_drift
    #    start at relative_impact=100%. We'll skip it unless you have a reason to chain them.

    # 3) Call advanced_component_drift on this expression node => returns the parse tree annotated with "drift"
    annotated = advanced_component_drift(expr_copy)

    # 4) Build a top-level result dictionary
    result = {
        "metric_id": annotated.get("metric_id", None),
        "evaluation_value": annotated.get("evaluation_value"),
        "comparison_value": annotated.get("comparison_value"),
        "drift": annotated.get("drift", {}),
        "components": []
    }

    # 5) If "operands" in annotated, we check each child. If that child is "type"="metric", we have a leaf.
    #    If "type"="expression", we recursively call ourselves.
    #    We'll build out the result["components"] array.

    def build_components(node) -> List[Dict[str, Any]]:
        comps = []
        if node.get("type") == "expression":
            for child in node.get("operands", []):
                comp_data = {
                    "metric_id": child.get("metric_id"),
                    "evaluation_value": child.get("evaluation_value"),
                    "comparison_value": child.get("comparison_value"),
                    "drift": child.get("drift", {}),
                    "components": []
                }
                if child.get("type") == "expression":
                    # Recurse deeper
                    deeper_annotated = calculate_recursive_component_drift(child, parent_drift=child.get("drift"))
                    comp_data["metric_id"] = deeper_annotated.get("metric_id")
                    comp_data["evaluation_value"] = deeper_annotated.get("evaluation_value")
                    comp_data["comparison_value"] = deeper_annotated.get("comparison_value")
                    comp_data["drift"] = deeper_annotated.get("drift", {})
                    comp_data["components"] = deeper_annotated.get("components", [])
                # else if child is "metric", we just keep the leaf info
                comps.append(comp_data)
        elif node.get("type") == "metric":
            # leaf
            pass
        return comps

    result["components"] = build_components(annotated)

    return result


#
# Example usage:
#
# metric_expr = {
#   "metric_id": "CAC",
#   "type": "expression",
#   "operator": "/",
#   "operands": [
#       {
#         "type": "expression",
#         "operator": "+",
#         "operands": [
#             {
#               "type": "metric",
#               "metric_id": "SalesSpend",
#               "evaluation_value": 80.0,
#               "comparison_value": 70.0
#             },
#             {
#               "type": "metric",
#               "metric_id": "MarketingSpend",
#               "evaluation_value": 40.0,
#               "comparison_value": 30.0
#             }
#         ]
#       },
#       {
#         "type": "expression",
#         "operator": "-",
#         "operands": [
#             {
#               "type": "metric",
#               "metric_id": "NewCust",
#               "evaluation_value": 60.0,
#               "comparison_value": 55.0
#             },
#             {
#               "type": "metric",
#               "metric_id": "LostCust",
#               "evaluation_value": 5.0,
#               "comparison_value": 7.0
#             }
#         ]
#       }
#    ]
# }
#
# final_drift = calculate_recursive_component_drift(metric_expr)
# logger.info("Final drift result: %s", final_drift)
#
