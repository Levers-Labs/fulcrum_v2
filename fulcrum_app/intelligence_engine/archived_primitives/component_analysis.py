# file: component_analysis.py

import pandas as pd
import numpy as np
from typing import Optional, Any, Dict, List
import math
import logging

logger = logging.getLogger(__name__)

def decompose_metric_change(
    val_t0: float, 
    val_t1: float, 
    factors: dict,
    relationship: str = "additive"
) -> dict:
    """
    Decompose the change in a metric (val_t0 -> val_t1) among multiple factors.

    relationship='additive' => M = sum of factors
    relationship='multiplicative' => M ~ product of factors, partial effect in log space.

    This simpler function stands alone if all factor values are known
    and we just want to see how each factor contributed to the top-level
    change from val_t0 to val_t1.
    """
    total_change = val_t1 - val_t0

    if relationship == "additive":
        factor_changes = {}
        sum_of_changes = 0.0

        for f_name, (f0, f1) in factors.items():
            delta = f1 - f0
            factor_changes[f_name] = {"delta": delta}
            sum_of_changes += delta

        residual = total_change - sum_of_changes

        for f_name, fdict in factor_changes.items():
            delta = fdict["delta"]
            if total_change != 0:
                abs_contrib = delta
                pct_contrib = (delta / total_change) * 100.0
            else:
                abs_contrib = 0.0
                pct_contrib = 0.0
            fdict["contribution_abs"] = abs_contrib
            fdict["contribution_pct"] = pct_contrib

        return {
            "total_change": total_change,
            "factors": factor_changes,
            "residual": residual
        }

    elif relationship == "multiplicative":
        if val_t0 <= 0 or val_t1 <= 0:
            return {"error": "multiplicative assumed positive metric, got zero or negative."}

        ratio_M = val_t1 / val_t0
        if ratio_M <= 0:
            return {"error": "cannot do multiplicative with non-positive ratio."}

        log_M = math.log(ratio_M)
        factor_changes = {}
        sum_of_factor_logs = 0.0

        for f_name, (f0, f1) in factors.items():
            if f0 <= 0 or f1 <= 0:
                factor_changes[f_name] = {
                    "delta": None, 
                    "contribution_abs": None, 
                    "contribution_pct": None
                }
                continue
            ratio_f = f1 / f0
            log_f = math.log(ratio_f)
            sum_of_factor_logs += log_f
            factor_changes[f_name] = {"delta": ratio_f - 1.0, "log_change": log_f}

        residual_log = log_M - sum_of_factor_logs

        for f_name, fdict in factor_changes.items():
            if "log_change" not in fdict or fdict["log_change"] is None:
                continue
            log_f = fdict["log_change"]
            if abs(log_M) > 1e-9:
                ratio_share = log_f / log_M
            else:
                ratio_share = 0.0
            partial_abs = ratio_share * (val_t1 - val_t0)
            fdict["contribution_abs"] = partial_abs
            if abs(total_change) > 1e-9:
                fdict["contribution_pct"] = (partial_abs / total_change) * 100.0
            else:
                fdict["contribution_pct"] = 0.0

        return {
            "total_change": total_change,
            "factors": factor_changes,
            "residual_log": residual_log
        }

    else:
        return {"error": f"Unknown relationship: '{relationship}'"}


def calculate_component_drift(
    df: pd.DataFrame,
    formula: str,
    id_col: str = "component",
    value_col_t0: str = "value_t0",
    value_col_t1: str = "value_t1"
) -> pd.DataFrame:
    """
    Evaluate how each operand in a formula changed from T0->T1 (naive additive approach).
    E.g. formula="A = B + C". If B changed from 60->80, C from 40->50 => we show their deltas.

    This remains a simpler approach if we have a direct formula like "X = Y + Z".
    We do not do log or invert logic here; see 'compute_drift' for advanced usage.
    """
    if "+" in formula and "=" in formula:
        right_side = formula.split("=")[1].strip()
        operands = [s.strip(" ()") for s in right_side.split("+")]
        sub = df[df[id_col].isin(operands)].copy()
        sub["delta"] = sub[value_col_t1] - sub[value_col_t0]
        sub["partial_effect"] = sub["delta"]
        return sub
    else:
        return df.assign(delta=None, partial_effect=None)


def calculate_delta(
    eval_value: float,
    comp_value: float,
    operator: str,
    invert: bool=False
) -> float:
    """
    For operator in [*, /], use log-based approach if "multiplicative" (like old code).
    If invert=True and operator='/', handle denominator logic.
    For operator in [+, -], do a simple difference.

    This is used inside 'compute_drift' for advanced expression-based decomposition.
    """
    if operator in ["*", "/"]:
        # old code approach: log(evaluation_value) - log(comparison_value)
        # or if invert, log(1/eval) - log(1/comp) => -log(eval) - [-log(comp)] => same as log(comp) - log(eval).
        if eval_value <= 0 or comp_value <= 0:
            # skip or 0? We'll just return 0 if invalid
            return 0.0

        if invert:
            # means denominator => log(1/eval_value) - log(1/comp_value)
            # => -log(eval_value) - [-log(comp_value)] => (log(comp_value) - log(eval_value))
            return math.log(comp_value) - math.log(eval_value)
        else:
            return math.log(eval_value) - math.log(comp_value)
    else:
        # additive
        return eval_value - comp_value


def compute_drift(
    node: Dict[str, Any],
    parent_operator: str = "+",
    parent_node: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Recursively compute 'drift' for a node in an expression parse-tree.
    This merges old code's advanced approach (with invert_values for division) and
    new code's existing structure.

    For each node, we compute:
      - absolute_drift
      - percentage_drift
      - relative_impact
      - marginal_contribution
      - relative_impact_root
      - marginal_contribution_root

    The parent's drift info is used to chain up relative impacts.
    """
    ev = node["evaluation_value"]
    cv = node["comparison_value"]

    # If no parent => top-level
    if not parent_node:
        abs_drift = ev - cv
        pct_drift = (abs_drift / cv * 100) if cv != 0 else 0
        # set them all
        node["drift"] = {
            "absolute_drift": abs_drift,
            "percentage_drift": pct_drift,
            "relative_impact": 1.0,  # top-level => 100% of itself
            "marginal_contribution": 1.0,
            "relative_impact_root": 1.0,
            "marginal_contribution_root": 1.0
        }
    else:
        # compute "delta" for this node using old code approach
        invert = False
        if parent_operator == "/" and node.get("_operand_index", 0) > 0:
            invert = True

        parent_ev = parent_node["evaluation_value"]
        parent_cv = parent_node["comparison_value"]

        # parent's total "delta" in either additive or multiplicative sense
        parent_delta = calculate_delta(parent_ev, parent_cv, parent_operator, invert=False)

        my_delta = calculate_delta(ev, cv, parent_operator, invert=invert)

        abs_drift = ev - cv
        pct_drift = (abs_drift / cv * 100) if cv != 0 else 0

        # relative_impact w.r.t parent's delta
        rel_impact = 0.0
        if abs(parent_delta) > 1e-9:
            rel_impact = (my_delta / parent_delta)

        # parent's drift
        pdrift = parent_node.get("drift", {})
        parent_rel_impact = pdrift.get("relative_impact", 1.0)
        parent_marg_contrib = pdrift.get("marginal_contribution", 1.0)
        parent_rel_impact_root = pdrift.get("relative_impact_root", 1.0)
        parent_marg_contrib_root = pdrift.get("marginal_contribution_root", 1.0)

        # combine
        # "relative_impact" = fraction w.r.t parent's drift
        # old code multiplied parent's relative_impact also
        # so that if parent had 50% share, child might have 0.1 => child is 5% overall
        # but let's keep it consistent
        actual_rel_impact = rel_impact * parent_rel_impact
        # "marginal_contribution" * parent's "marginal_contribution"
        # old code used something like that
        # let's define "marginal_contribution" = rel_impact * parent's percentage_drift?
        # The old code: marginal_contribution = relative_impact * parent's drift["percentage_drift"]...
        # We'll keep it simpler:
        # We'll define "marginal_contribution" = rel_impact * parent_marg_contrib
        # so if parent's is 1 => child is 0.1 => etc.
        actual_marg_contrib = rel_impact * parent_marg_contrib

        # "relative_impact_root": chain up
        actual_rel_impact_root = rel_impact * parent_rel_impact_root
        actual_marg_contrib_root = rel_impact * parent_marg_contrib_root

        node["drift"] = {
            "absolute_drift": abs_drift,
            "percentage_drift": pct_drift,
            "relative_impact": actual_rel_impact,
            "marginal_contribution": actual_marg_contrib,
            "relative_impact_root": actual_rel_impact_root,
            "marginal_contribution_root": actual_marg_contrib_root
        }

    # If node has sub-expressions => keep recursing
    if node.get("type") == "expression":
        op = node["operator"]
        for i, child in enumerate(node["operands"]):
            child["_operand_index"] = i
            compute_drift(child, parent_operator=op, parent_node=node)
    return node


def advanced_component_drift(root_expression: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate sub-expressions, then compute drift recursively.
    Returns the fully annotated expression tree with node["drift"] at each node.

    1) Evaluate final numeric for each sub-node (the new code's "evaluate_expression").
    2) Then call compute_drift(...) recursively (the merged logic).
    """

    def evaluate_expression(expr: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate expression's 'evaluation_value' and 'comparison_value' from children.

        The user presumably sets expr["type"], e.g. "value" or "expression".
        The new code's old approach used "type"="value", or "metric".
        We'll unify them: if expr["type"] == "metric", we use existing "evaluation_value" in the expr,
        otherwise if it's "expression", we sum/multiply child results, etc.
        """
        if expr.get("type") == "value":
            return {
                "evaluation_value": expr["evaluation_value"],
                "comparison_value": expr["comparison_value"]
            }
        elif expr.get("type") == "metric":
            # If the node itself has "evaluation_value", use it
            return {
                "evaluation_value": expr["evaluation_value"],
                "comparison_value": expr["comparison_value"]
            }
        elif expr.get("type") == "expression":
            op = expr["operator"]
            operands = expr["operands"]
            # Evaluate children
            child_vals = [evaluate_expression(o) for o in operands]
            if op == "+":
                ev = sum(v["evaluation_value"] for v in child_vals)
                cv = sum(v["comparison_value"] for v in child_vals)
            elif op == "-":
                # assume first minus the rest
                if len(child_vals) == 0:
                    ev, cv = 0, 0
                else:
                    ev = child_vals[0]["evaluation_value"]
                    cv = child_vals[0]["comparison_value"]
                    for subv in child_vals[1:]:
                        ev -= subv["evaluation_value"]
                        cv -= subv["comparison_value"]
            elif op == "*":
                ev, cv = 1.0, 1.0
                for v in child_vals:
                    ev *= v["evaluation_value"]
                    cv *= v["comparison_value"]
            elif op == "/":
                # assume first / second if length=2
                if len(child_vals) < 2:
                    ev, cv = 0, 0
                else:
                    ev = child_vals[0]["evaluation_value"]
                    cv = child_vals[0]["comparison_value"]
                    for subv in child_vals[1:]:
                        sub_ev = subv["evaluation_value"]
                        sub_cv = subv["comparison_value"]
                        if abs(sub_ev) < 1e-12 or abs(sub_cv) < 1e-12:
                            # skip or handle
                            if abs(sub_ev) < 1e-12:
                                ev = float('inf')
                            if abs(sub_cv) < 1e-12:
                                cv = float('inf')
                        else:
                            ev = ev / sub_ev
                            cv = cv / sub_cv
            else:
                # fallback => sum
                ev = sum(v["evaluation_value"] for v in child_vals)
                cv = sum(v["comparison_value"] for v in child_vals)

            return {
                "evaluation_value": ev,
                "comparison_value": cv
            }
        else:
            # fallback
            return {"evaluation_value": 0.0, "comparison_value": 0.0}

    # 1) Evaluate final numeric
    top_vals = evaluate_expression(root_expression)
    root_expression["evaluation_value"] = top_vals["evaluation_value"]
    root_expression["comparison_value"] = top_vals["comparison_value"]

    # 2) compute drift
    compute_drift(root_expression, parent_operator="+", parent_node=None)
    return root_expression
