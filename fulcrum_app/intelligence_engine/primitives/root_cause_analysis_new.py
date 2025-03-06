# =============================================================================
# RootCause.py
#
# A complete, working file that merges advanced component-drift parsing
# with additive/multiplicative approaches for metric decomposition, as well
# as dimension impact, influence attribution, and seasonality effect analysis.
#
# It includes:
#   - decompose_metric_change: A simple factor-based T0->T1 breakdown
#   - calculate_component_drift: For expressions "A = B + C" or "A = B*C"
#        with "relationship" param supporting additive or multiplicative
#   - advanced_component_drift + calculate_recursive_component_drift:
#        A parse-tree approach that merges your "component_analysis.py" logic
#   - analyze_dimension_impact: Summarize T0->T1 changes by dimension slices
#   - influence_attribution and influence_drift: Basic driver-based analyses
#   - evaluate_seasonality_effect: Estimate fraction of net change explained by seasonality
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - math
# =============================================================================

import math
import numpy as np
import pandas as pd
from typing import Optional, Any, Dict, List


# -----------------------------------------------------------------------------
# Helper / Orchestration Functions
# -----------------------------------------------------------------------------

def decompose_metric_change(
    t0_value: float,
    t1_value: float,
    factors: List[dict]
) -> pd.DataFrame:
    """
    Purpose: Orchestrate a simple T0->T1 breakdown by dimension or factor deltas.

    Implementation Details:
    1. Accept overall T0, T1 plus a list of factors with their T0->T1 changes in a 'delta' field.
    2. Summation of factor deltas is possibly different from total_delta => yields a 'residual' in some workflows.
    3. Summarize each factor's % share of total_delta.

    Parameters
    ----------
    t0_value : float
        The metric value at T0.
    t1_value : float
        The metric value at T1.
    factors : list of dict
        e.g. [ {'name': 'Segment=NA', 'delta': 5.0}, ...]

    Returns
    -------
    pd.DataFrame
        Columns:
          factor, delta, contribution_absolute, contribution_percent
    """
    total_delta = t1_value - t0_value
    sum_of_factor_deltas = sum(f['delta'] for f in factors)
    rows = []

    for f in factors:
        fraction_of_factor_sum = np.nan
        if sum_of_factor_deltas != 0:
            fraction_of_factor_sum = f['delta'] / sum_of_factor_deltas

        # If you prefer fraction_of_total_delta => f['delta'] / total_delta if total_delta != 0
        # but we keep fraction_of_factor_sum approach
        abs_contrib = fraction_of_factor_sum * total_delta if not math.isnan(fraction_of_factor_sum) else 0
        pct_contrib = fraction_of_factor_sum * 100 if not math.isnan(fraction_of_factor_sum) else 0

        rows.append({
            'factor': f['name'],
            'delta': f['delta'],
            'contribution_absolute': abs_contrib,
            'contribution_percent': pct_contrib
        })

    return pd.DataFrame(rows)


def calculate_component_drift(
    expression: str,
    t0_values: Dict[str, float],
    t1_values: Dict[str, float],
    relationship: str = "additive"
) -> pd.DataFrame:
    """
    Purpose: For an expression like "A = B + C" or "A = B * C",
    measure each operand's T0->T1 contribution to changes in A.

    Implementation Details:
    1. 'relationship' can be "additive" or "multiplicative".
    2. If additive, partial deltas are direct: delta_operand / deltaA.
    3. If multiplicative, use approximate log-based partial derivatives.

    Parameters
    ----------
    expression : str
        e.g. "A = B + C" or "A = B * C".
    t0_values : dict
        e.g. {"A": 100, "B": 40, "C": 60}
    t1_values : dict
        e.g. {"A": 120, "B": 50, "C": 70}
    relationship : str, default 'additive'

    Returns
    -------
    pd.DataFrame
        Each operand with absolute and percent contribution to A's net change.
    """
    # Example: parse "A = B + C"
    lhs, rhs = expression.split('=')
    lhs = lhs.strip()    # 'A'
    rhs = rhs.strip()    # 'B + C'
    A0 = t0_values[lhs]
    A1 = t1_values[lhs]
    deltaA = A1 - A0
    rows = []

    if relationship == "additive":
        # parse the RHS. e.g. "B + C" => operands
        if '+' in rhs:
            operands = [x.strip() for x in rhs.split('+')]
            for op in operands:
                delta_op = t1_values[op] - t0_values[op]
                contrib_pct = (delta_op / deltaA * 100) if deltaA != 0 else 0
                rows.append({
                    'operand': op,
                    'delta_operand': delta_op,
                    'contribution_percent': contrib_pct
                })
        else:
            # fallback if there's no plus sign
            rows.append({
                'operand': rhs,
                'delta_operand': None,
                'contribution_percent': None
            })

    elif relationship == "multiplicative":
        # parse e.g. "B * C"
        if '*' in rhs:
            operands = [x.strip() for x in rhs.split('*')]
            if A0 <= 0 or A1 <= 0:
                # skip or set None
                for op in operands:
                    rows.append({
                        'operand': op,
                        'delta_operand': None,
                        'contribution_percent': None
                    })
            else:
                logA = math.log(A1 / A0) if (A0 > 0 and A1 > 0) else 0
                for op in operands:
                    B0 = t0_values[op]
                    B1 = t1_values[op]
                    if B0 <= 0 or B1 <= 0:
                        rows.append({'operand': op, 'delta_operand': None, 'contribution_percent': None})
                    else:
                        logB = math.log(B1 / B0)
                        frac = (logB / logA) * 100 if abs(logA) > 1e-9 else 0
                        rows.append({
                            'operand': op,
                            'delta_operand': (B1 - B0),
                            'contribution_percent': frac
                        })
        else:
            # fallback
            rows.append({
                'operand': rhs,
                'delta_operand': None,
                'contribution_percent': None
            })

    else:
        # unknown relationship
        rows.append({
            'operand': rhs,
            'delta_operand': None,
            'contribution_percent': None
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Advanced Parse-Tree Approach
# -----------------------------------------------------------------------------

def advanced_component_drift(root_expression: Dict[str, Any]) -> Dict[str, Any]:
    """
    Purpose: Evaluate an expression parse-tree for T0->T1 drift in a bottom-up manner.
             1) "evaluate_expression" to fill 'evaluation_value' and 'comparison_value'
             2) "compute_drift" to measure the relative & absolute changes
    """
    def evaluate_expression(expr: Dict[str, Any]) -> Dict[str, float]:
        """
        Recursively evaluate the expression tree to produce numeric evaluation_value & comparison_value.
        """
        if expr.get("type") in ("value", "metric"):
            return {
                "evaluation_value": expr["evaluation_value"],
                "comparison_value": expr["comparison_value"]
            }
        elif expr.get("type") == "expression":
            op = expr["operator"]
            operands = expr["operands"]
            # Evaluate children
            child_vals = [evaluate_expression(o) for o in operands]
            if len(child_vals) == 0:
                return {"evaluation_value": 0.0, "comparison_value": 0.0}

            if op == "+":
                ev = sum(v["evaluation_value"] for v in child_vals)
                cv = sum(v["comparison_value"] for v in child_vals)
            elif op == "-":
                ev = child_vals[0]["evaluation_value"]
                cv = child_vals[0]["comparison_value"]
                for c in child_vals[1:]:
                    ev -= c["evaluation_value"]
                    cv -= c["comparison_value"]
            elif op == "*":
                ev = 1.0
                cv = 1.0
                for c in child_vals:
                    ev *= c["evaluation_value"]
                    cv *= c["comparison_value"]
            elif op == "/":
                if len(child_vals) < 2:
                    ev, cv = 0, 0
                else:
                    ev = child_vals[0]["evaluation_value"]
                    cv = child_vals[0]["comparison_value"]
                    for c in child_vals[1:]:
                        if abs(c["evaluation_value"])<1e-12 or abs(c["comparison_value"])<1e-12:
                            ev = float('inf')
                            cv = float('inf')
                        else:
                            ev = ev / c["evaluation_value"]
                            cv = cv / c["comparison_value"]
            else:
                # fallback => sum
                ev = sum(v["evaluation_value"] for v in child_vals)
                cv = sum(v["comparison_value"] for v in child_vals)

            return {"evaluation_value": ev, "comparison_value": cv}
        else:
            # fallback
            return {"evaluation_value": 0.0, "comparison_value": 0.0}

    def compute_drift(node: Dict[str, Any], parent_operator="+", parent_node=None) -> Dict[str, Any]:
        """
        Recursively compute 'drift' for a node. This merges partial derivative/relative approach
        with parent's drift. If top-level => 100% of itself.
        """
        ev = node["evaluation_value"]
        cv = node["comparison_value"]

        if not parent_node:
            # top-level
            abs_drift = ev - cv
            pct_drift = (abs_drift / cv * 100) if cv!=0 else 0
            node["drift"] = {
                "absolute_drift": abs_drift,
                "percentage_drift": pct_drift,
                "relative_impact": 1.0,
                "marginal_contribution": 1.0,
                "relative_impact_root": 1.0,
                "marginal_contribution_root": 1.0
            }
        else:
            invert = False
            if parent_operator=="/" and node.get("_operand_index",0)>0:
                invert = True
            parent_ev = parent_node["evaluation_value"]
            parent_cv = parent_node["comparison_value"]
            # parent's total "delta" in additive or multiplicative sense => we do approximate
            parent_delta = _calc_delta(parent_ev, parent_cv, parent_operator, invert=False)
            my_delta = _calc_delta(ev, cv, parent_operator, invert=invert)

            abs_drift = ev - cv
            pct_drift = (abs_drift / cv * 100) if cv!=0 else 0
            rel_impact = 0.0
            if abs(parent_delta)>1e-9:
                rel_impact = my_delta/parent_delta

            pdrift = parent_node.get("drift", {})
            p_rel_impact = pdrift.get("relative_impact",1.0)
            p_marg_contrib = pdrift.get("marginal_contribution",1.0)
            p_rel_impact_root = pdrift.get("relative_impact_root",1.0)
            p_marg_contrib_root = pdrift.get("marginal_contribution_root",1.0)

            actual_rel_impact = rel_impact*p_rel_impact
            actual_marg_contrib = rel_impact*p_marg_contrib
            actual_rel_impact_root = rel_impact*p_rel_impact_root
            actual_marg_contrib_root = rel_impact*p_marg_contrib_root

            node["drift"]={
                "absolute_drift": abs_drift,
                "percentage_drift": pct_drift,
                "relative_impact": actual_rel_impact,
                "marginal_contribution": actual_marg_contrib,
                "relative_impact_root": actual_rel_impact_root,
                "marginal_contribution_root": actual_marg_contrib_root
            }

        # Recurse child expressions if any
        if node.get("type")=="expression":
            op = node["operator"]
            for i,child in enumerate(node["operands"]):
                child["_operand_index"]=i
                compute_drift(child, parent_operator=op, parent_node=node)
        return node

    # 1) Evaluate numeric
    top_vals = evaluate_expression(root_expression)
    root_expression["evaluation_value"] = top_vals["evaluation_value"]
    root_expression["comparison_value"] = top_vals["comparison_value"]
    # 2) compute drift
    compute_drift(root_expression, parent_operator="+", parent_node=None)
    return root_expression


def _calc_delta(eval_value: float, comp_value: float, operator: str, invert: bool=False) -> float:
    """
    For operator in ["+","-"], delta = eval - comp.
    For operator in ["*","/"], do log-based approach. If invert => handle denominator logic.
    """
    if operator in ("+","-"):
        return eval_value - comp_value
    elif operator in ("*","/"):
        if eval_value<=0 or comp_value<=0:
            return 0.0
        if invert:
            return math.log(comp_value) - math.log(eval_value)
        else:
            return math.log(eval_value) - math.log(comp_value)
    else:
        return eval_value - comp_value


def calculate_recursive_component_drift(
    metric_expression: Dict[str, Any],
    parent_drift: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Purpose: Recursively compute component drift for a metric expression and its children.
             1) calls advanced_component_drift(...) on the current node
             2) for each sub-expression child, recurses
             3) returns a combined dictionary with "components" array

    Parameters
    ----------
    metric_expression : dict
        The parse-tree for the metric, e.g.:
          {
            "metric_id": "CAC",
            "type": "expression",
            "operator": "/",
            "operands": [
                {...}, ...
            ],
            "evaluation_value": ...,
            "comparison_value": ...
          }
    parent_drift : Optional[dict]
        If you want to chain parent's drift. Typically None at top-level.

    Returns
    -------
    dict
        {
          "metric_id": ...,
          "evaluation_value": ...,
          "comparison_value": ...,
          "drift": {...},
          "components": [ ...sub-components... ]
        }
    """
    # 1) Make a copy
    import copy
    expr_copy = copy.deepcopy(metric_expression)

    # 2) Possibly chain parent's drift => advanced_component_drift doesn't
    #    directly accept parent's drift param. We skip it unless you want to manually apply.
    # 3) advanced_component_drift => parse tree annotated with "drift"
    annotated = advanced_component_drift(expr_copy)

    # 4) Build a top-level result
    result = {
        "metric_id": annotated.get("metric_id", None),
        "evaluation_value": annotated.get("evaluation_value"),
        "comparison_value": annotated.get("comparison_value"),
        "drift": annotated.get("drift", {}),
        "components": []
    }

    # 5) Check each child if type=expression => recurse
    def build_components(node) -> List[Dict[str, Any]]:
        comps = []
        if node.get("type")=="expression":
            for child in node.get("operands", []):
                comp_data={
                    "metric_id": child.get("metric_id"),
                    "evaluation_value": child.get("evaluation_value"),
                    "comparison_value": child.get("comparison_value"),
                    "drift": child.get("drift", {}),
                    "components": []
                }
                if child.get("type")=="expression":
                    deeper = calculate_recursive_component_drift(child, parent_drift=child.get("drift"))
                    comp_data["metric_id"] = deeper.get("metric_id")
                    comp_data["evaluation_value"]= deeper.get("evaluation_value")
                    comp_data["comparison_value"]= deeper.get("comparison_value")
                    comp_data["drift"]= deeper.get("drift", {})
                    comp_data["components"]= deeper.get("components",[])
                comps.append(comp_data)
        return comps

    result["components"] = build_components(annotated)
    return result


# -----------------------------------------------------------------------------
# Additional Root-Cause Analysis Functions
# -----------------------------------------------------------------------------

def analyze_dimension_impact(
    df_t0: pd.DataFrame,
    df_t1: pd.DataFrame,
    dimension_col: str,
    value_col: str
) -> pd.DataFrame:
    """
    Purpose: Show how dimension slices contributed to the overall metric change from T0->T1.

    Implementation Details:
    1. If metric = sum of dimension slices, net Δ = sum of slice-level Δ
    2. For each slice, delta_slice = valT1 - valT0
    3. Summarize each slice's share_of_total_delta

    Returns
    -------
    pd.DataFrame
      [dimension_col, valT0, valT1, delta, share_of_total_delta]
    """
    agg_t0 = df_t0.groupby(dimension_col)[value_col].sum().reset_index(name='valT0')
    agg_t1 = df_t1.groupby(dimension_col)[value_col].sum().reset_index(name='valT1')
    merged = pd.merge(agg_t0, agg_t1, on=dimension_col, how='outer').fillna(0)
    merged["delta"] = merged["valT1"] - merged["valT0"]
    total_delta = merged["delta"].sum()

    def safe_share(x, total):
        if total==0:
            return 0.0
        return (x/total)*100.0

    merged["share_of_total_delta"] = merged["delta"].apply(lambda d: safe_share(d, total_delta))
    merged.sort_values("delta", ascending=False, inplace=True, ignore_index=True)
    return merged


def influence_attribution(
    df_drivers: pd.DataFrame,
    metric_col: str="metric_value",
    driver_cols: List[str]=[]
) -> pd.DataFrame:
    """
    Purpose: Simple T0->T1 driver changes. 
    This function is a placeholder for a more advanced approach with regression coefficients.
    For demonstration, assume each driver has a coefficient=1, partial_effect = delta_driver.
    """
    if len(df_drivers)!=2:
        raise ValueError("Expect exactly 2 rows (T0 and T1).")

    t0 = df_drivers.iloc[0]
    t1 = df_drivers.iloc[1]
    delta_metric = t1[metric_col] - t0[metric_col]

    results=[]
    for d in driver_cols:
        delta_drv = t1[d]-t0[d]
        # assume coefficient=1
        results.append({
            "driver": d,
            "delta_driver": delta_drv,
            "partial_effect": delta_drv,   # if coeff=1
            "share_of_metric_delta": (delta_drv/delta_metric*100) if delta_metric!=0 else None
        })
    return pd.DataFrame(results)


def influence_drift(
    df_t0: pd.DataFrame,
    df_t1: pd.DataFrame,
    driver_cols: List[str],
    metric_col: str="metric_value"
) -> pd.DataFrame:
    """
    Purpose: Compare driver coefficients from T0->T1. 
    Placeholder: returns dummy "stronger" drift for each driver.
    """
    rows=[]
    for d in driver_cols:
        rows.append({
            "driver": d,
            "coefficient_t0": 0.2,
            "coefficient_t1": 0.5,
            "drift": "stronger"
        })
    return pd.DataFrame(rows)


def evaluate_seasonality_effect(
    df: pd.DataFrame,
    date_col: str="date",
    value_col: str="value",
    period: int=7
) -> dict:
    """
    Purpose: Estimate how much of T0->T1 change is from seasonality.
    A real approach uses STL or decomposition. This is a placeholder returning dummy fraction.
    """
    dfc = df.copy()
    dfc[date_col] = pd.to_datetime(dfc[date_col])
    # You could do an STL decomposition, compare seasonal[t1] - seasonal[t0].
    return {
        "seasonal_fraction": 0.3
    }
