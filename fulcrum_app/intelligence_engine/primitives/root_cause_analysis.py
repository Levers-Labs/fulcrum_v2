# =============================================================================
# RootCauseAnalysis
#
# This file provides primitives for determining the causes of metric changes:
# - Decomposing metric changes by factors/components
# - Analyzing dimension impacts on metrics
# - Attribution of changes to drivers
# - Seasonality impact analysis
# - Quantifying event impacts
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - math
# =============================================================================

import math
import numpy as np
import pandas as pd
from typing import Optional, Any, Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _safe_divide(numerator, denominator):
    """Safely divide, handling zeros."""
    if denominator == 0 or pd.isna(denominator):
        return 0 if numerator == 0 or pd.isna(numerator) else float('inf')
    return numerator / denominator

def _calc_delta(eval_value: float, comp_value: float, operator: str, invert: bool=False) -> float:
    """
    Calculate delta between values based on operator type.
    
    For operator in ["+","-"], delta = eval - comp.
    For operator in ["*","/"], use log-based approach. If invert => handle denominator logic.
    """
    if operator in ("+", "-"):
        return eval_value - comp_value
    elif operator in ("*", "/"):
        if eval_value <= 0 or comp_value <= 0:
            return 0.0
        if invert:
            return math.log(comp_value) - math.log(eval_value)
        else:
            return math.log(eval_value) - math.log(comp_value)
    else:
        return eval_value - comp_value

# -----------------------------------------------------------------------------
# Basic Factor-based Decomposition
# -----------------------------------------------------------------------------

def decompose_metric_change(
    val_t0: float, 
    val_t1: float, 
    factors: Dict[str, Tuple[float, float]],
    relationship: str = "additive"
) -> Dict[str, Any]:
    """
    Decompose the change in a metric (val_t0 -> val_t1) among multiple factors.

    Parameters
    ----------
    val_t0 : float
        The metric's value at time T0.
    val_t1 : float
        The metric's value at time T1.
    factors : dict
        A dictionary mapping factor names to tuples of (value_at_t0, value_at_t1).
        E.g., {"segmentA": (100, 120), "segmentB": (80, 70)}.
    relationship : str, default "additive"
        The relationship between factors:
        - "additive": metric = sum(factors)
        - "multiplicative": metric = product(factors)

    Returns
    -------
    dict
        Contains:
        - "total_change": The overall metric change val_t1 - val_t0
        - "factors": Dict mapping each factor to its contribution
        - "residual": Any unexplained change
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
            return {"error": "multiplicative requires positive values, got zero or negative."}

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
            partial_abs = ratio_share * total_change
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

# -----------------------------------------------------------------------------
# Component Expression Analysis
# -----------------------------------------------------------------------------

def calculate_component_drift(
    df: pd.DataFrame,
    formula: str,
    id_col: str = "component",
    value_col_t0: str = "value_t0",
    value_col_t1: str = "value_t1"
) -> pd.DataFrame:
    """
    Evaluate how each operand in a formula changed from T0->T1.
    
    For expressions like "A = B + C", if B changed from 60->80, C from 40->50, 
    this shows their deltas and partial effect on A's change.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with components and their values at T0 and T1.
    formula : str
        Expression like "A = B + C" or "A = B * C".
    id_col : str, default "component"
        Column containing component identifiers.
    value_col_t0 : str, default "value_t0"
        Column containing values at T0.
    value_col_t1 : str, default "value_t1"
        Column containing values at T1.

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with 'delta' and 'partial_effect' columns.
    """
    if "+" in formula and "=" in formula:
        right_side = formula.split("=")[1].strip()
        operands = [s.strip(" ()") for s in right_side.split("+")]
        sub = df[df[id_col].isin(operands)].copy()
        sub["delta"] = sub[value_col_t1] - sub[value_col_t0]
        sub["partial_effect"] = sub["delta"]
        return sub
    elif "*" in formula and "=" in formula:
        right_side = formula.split("=")[1].strip()
        operands = [s.strip(" ()") for s in right_side.split("*")]
        sub = df[df[id_col].isin(operands)].copy()
        sub["delta"] = sub[value_col_t1] - sub[value_col_t0]
        
        # For multiplicative, we use approximate partial derivatives
        # If A = B * C, the partial effect of B's change is approximately C*ΔB
        lhs = formula.split("=")[0].strip()
        a0 = df.loc[df[id_col] == lhs, value_col_t0].iloc[0]
        a1 = df.loc[df[id_col] == lhs, value_col_t1].iloc[0]
        total_delta = a1 - a0
        
        results = []
        for _, row in sub.iterrows():
            component = row[id_col]
            other_operands = [op for op in operands if op != component]
            
            # Get average of other operands
            other_vals_product = 1.0
            for other in other_operands:
                other_t0 = df.loc[df[id_col] == other, value_col_t0].iloc[0]
                other_t1 = df.loc[df[id_col] == other, value_col_t1].iloc[0]
                other_vals_product *= (other_t0 + other_t1) / 2  # Use average
            
            # Partial effect: (change in component) * (product of other components)
            partial_effect = row["delta"] * other_vals_product
            
            new_row = row.copy()
            new_row["partial_effect"] = partial_effect
            results.append(new_row)
        
        return pd.DataFrame(results)
    else:
        return df.assign(delta=None, partial_effect=None)

# -----------------------------------------------------------------------------
# Advanced Expression Tree Analysis
# -----------------------------------------------------------------------------

def advanced_component_drift(root_expression: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate an expression parse-tree for T0->T1 drift in a bottom-up manner.
    
    First evaluates numeric values through the tree, then computes drift.
    Supports complex expressions with nested operations.

    Parameters
    ----------
    root_expression : dict
        A parse tree representing an expression with following structure:
        {
            "type": "expression", 
            "operator": "+", 
            "operands": [
                {"type": "metric", "evaluation_value": 100, "comparison_value": 90},
                {"type": "metric", "evaluation_value": 200, "comparison_value": 180}
            ]
        }

    Returns
    -------
    dict
        Expression tree annotated with "drift" information for each node.
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

def calculate_recursive_component_drift(
    metric_expression: Dict[str, Any],
    parent_drift: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Recursively compute component drift for a metric expression and all its children.
    
    This is a higher-level wrapper around advanced_component_drift that:
    1. Creates a clean result structure with "components" array
    2. Recursively handles subexpressions

    Parameters
    ----------
    metric_expression : dict
        A parse tree representation of a metric expression
    parent_drift : Optional[dict]
        For internal use in recursive calls

    Returns
    -------
    dict
        Structure with drift and components information
    """
    # 1) Make a copy
    import copy
    expr_copy = copy.deepcopy(metric_expression)

    # 2) Call advanced_component_drift => parse tree annotated with "drift"
    annotated = advanced_component_drift(expr_copy)

    # 3) Build a top-level result
    result = {
        "metric_id": annotated.get("metric_id", None),
        "evaluation_value": annotated.get("evaluation_value"),
        "comparison_value": annotated.get("comparison_value"),
        "drift": annotated.get("drift", {}),
        "components": []
    }

    # 4) Process each child if type=expression => recurse
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
                    deeper = calculate_recursive_component_drift(
                        child, parent_drift=child.get("drift"))
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
# Dimension Impact Analysis
# -----------------------------------------------------------------------------

def analyze_dimension_impact(
    df_t0: pd.DataFrame,
    df_t1: pd.DataFrame,
    dimension_col: str = "segment",
    value_col: str = "value"
) -> pd.DataFrame:
    """
    Analyze how dimension slices contributed to overall metric change from T0 to T1.
    
    For example, how much of revenue change came from each region.

    Parameters
    ----------
    df_t0 : pd.DataFrame
        DataFrame for time T0.
    df_t1 : pd.DataFrame
        DataFrame for time T1.
    dimension_col : str, default "segment"
        Column representing the dimension.
    value_col : str, default "value"
        Column with metric values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - dimension_col: The dimension value
        - value_t0: Value at T0
        - value_t1: Value at T1
        - delta: Change in value
        - pct_of_total_delta: What percentage of overall change is from this slice
    """
    # Group by dimension, calculate sum for each time
    t0_agg = df_t0.groupby(dimension_col)[value_col].sum().reset_index(name='value_t0')
    t1_agg = df_t1.groupby(dimension_col)[value_col].sum().reset_index(name='value_t1')
    
    # Combine results
    merged = pd.merge(t0_agg, t1_agg, on=dimension_col, how='outer').fillna(0)
    
    # Calculate deltas
    merged["delta"] = merged["value_t1"] - merged["value_t0"]
    total_delta = merged["delta"].sum()
    
    # Calculate percentage contribution
    if total_delta != 0:
        merged["pct_of_total_delta"] = (merged["delta"] / total_delta) * 100
    else:
        merged["pct_of_total_delta"] = 0.0
    
    # Sort by absolute contribution
    merged.sort_values("delta", ascending=False, inplace=True, ignore_index=True)
    
    return merged

# -----------------------------------------------------------------------------
# Driver Attribution Analysis
# -----------------------------------------------------------------------------

def influence_attribution(
    model, 
    X_t0: Dict[str, float],
    X_t1: Dict[str, float],
    y_t0: float,
    y_t1: float,
    driver_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Attribute a metric change to its drivers using a regression model.
    
    For a model y = f(X), this calculates how much of Δy is due to each ΔX.

    Parameters
    ----------
    model : object
        A trained regression model with .predict() method
    X_t0 : dict
        Driver values at T0: {driver_name: value}
    X_t1 : dict
        Driver values at T1: {driver_name: value}
    y_t0 : float
        Output metric value at T0
    y_t1 : float
        Output metric value at T1
    driver_names : Optional[List[str]]
        If provided, use these driver names. Otherwise, extract from model if possible.

    Returns
    -------
    dict
        Contains:
        - estimated_delta: Model-estimated change 
        - residual: Actual - estimated change
        - drivers: Dict of driver contributions
    """
    # If driver_names not provided, try to extract from the model
    if driver_names is None:
        try:
            driver_names = model.feature_names_in_
        except AttributeError:
            driver_names = list(X_t0.keys())
    
    # Calculate actual delta
    actual_delta = y_t1 - y_t0
    
    # Ensure X is in the right format for prediction
    X_t0_array = np.array([X_t0[d] for d in driver_names]).reshape(1, -1)
    X_t1_array = np.array([X_t1[d] for d in driver_names]).reshape(1, -1)
    
    # Predicted values
    y_pred_t0 = model.predict(X_t0_array)[0]
    y_pred_t1 = model.predict(X_t1_array)[0]
    estimated_delta = y_pred_t1 - y_pred_t0
    
    # Calculate residual
    residual = actual_delta - estimated_delta
    
    # Calculate driver-level contributions using partial derivatives
    driver_impacts = {}
    for i, driver in enumerate(driver_names):
        # Create a perturbed X_t0 that has the driver value from X_t1
        X_perturbed = X_t0.copy()
        X_perturbed[driver] = X_t1[driver]
        X_perturbed_array = np.array([X_perturbed[d] for d in driver_names]).reshape(1, -1)
        
        # Calculate model output with just this one driver changed
        y_perturbed = model.predict(X_perturbed_array)[0]
        
        # The partial effect is the difference from baseline
        partial_effect = y_perturbed - y_pred_t0
        
        # Calculate percent of estimated delta
        pct_of_est = 0.0
        if abs(estimated_delta) > 1e-9:
            pct_of_est = (partial_effect / estimated_delta) * 100
        
        # Calculate percent of actual delta
        pct_of_act = 0.0
        if abs(actual_delta) > 1e-9:
            pct_of_act = (partial_effect / actual_delta) * 100
        
        driver_impacts[driver] = {
            "delta_x": X_t1[driver] - X_t0[driver],
            "partial_effect": partial_effect,
            "pct_of_estimated": pct_of_est,
            "pct_of_actual": pct_of_act
        }
    
    # Return complete result
    return {
        "estimated_delta": estimated_delta,
        "residual": residual,
        "drivers": driver_impacts
    }

def influence_drift(
    model_t0, 
    model_t1, 
    driver_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare driver coefficients from two time periods to see how influence has changed.

    Parameters
    ----------
    model_t0 : object
        Model trained on T0 data
    model_t1 : object
        Model trained on T1 data
    driver_names : List[str]
        List of driver/feature names

    Returns
    -------
    dict
        Maps each driver to a dict with:
        - coef_t0: Coefficient at T0
        - coef_t1: Coefficient at T1
        - delta_coef: Change in coefficient
    """
    result = {}
    
    # Extract coefficients from models
    coefs_t0 = getattr(model_t0, 'coef_', None)
    coefs_t1 = getattr(model_t1, 'coef_', None)
    
    # Handle intercepts separately
    intercept_t0 = getattr(model_t0, 'intercept_', 0.0)
    intercept_t1 = getattr(model_t1, 'intercept_', 0.0)
    
    # If coefs not accessible, return empty result
    if coefs_t0 is None or coefs_t1 is None:
        return {}
    
    # Process each driver
    for i, driver in enumerate(driver_names):
        coef_t0 = coefs_t0[i] if i < len(coefs_t0) else 0.0
        coef_t1 = coefs_t1[i] if i < len(coefs_t1) else 0.0
        delta = coef_t1 - coef_t0
        
        result[driver] = {
            "coef_t0": float(coef_t0),
            "coef_t1": float(coef_t1),
            "delta_coef": float(delta)
        }
    
    # Add intercept
    result["intercept"] = {
        "coef_t0": float(intercept_t0),
        "coef_t1": float(intercept_t1),
        "delta_coef": float(intercept_t1 - intercept_t0)
    }
    
    return result

# -----------------------------------------------------------------------------
# Event and Seasonality Impact
# -----------------------------------------------------------------------------

def evaluate_seasonality_effect(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    period: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate how much of the metric change is due to seasonality.
    
    Uses seasonal decomposition to separate seasonal component from trend and residual.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data
    date_col : str, default "date"
        Column with dates
    value_col : str, default "value"
        Column with metric values
    period : Optional[int]
        Seasonality period (e.g., 7 for weekly, 12 for monthly)
        If None, tries to detect automatically

    Returns
    -------
    dict
        Contains:
        - seasonal_diff: Change attributed to seasonality
        - total_diff: Total observed change
        - fraction_of_total_diff: Portion of change due to seasonality
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        return {
            "error": "statsmodels not installed. Cannot perform seasonal decomposition."
        }
    
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    
    # Detect period if not specified
    if period is None:
        # Simple heuristic for period detection
        # Weekly: 7, Monthly: ~30, Quarterly: ~90
        date_diffs = df[date_col].diff().dt.days.dropna()
        if date_diffs.empty:
            return {"error": "Insufficient data for seasonal analysis"}
        
        median_diff = date_diffs.median()
        if median_diff <= 1:  # Daily data
            period = 7  # Assume weekly seasonality for daily data
        elif median_diff <= 7:  # Weekly data
            period = 4  # Assume monthly seasonality for weekly data
        else:  # Monthly data
            period = 12  # Assume yearly seasonality for monthly data
    
    # Check if we have enough data
    if len(df) < period * 2:
        return {"error": "Not enough data for the specified period"}
    
    # Perform decomposition
    try:
        values = df[value_col].values
        decomposition = seasonal_decompose(values, period=period, model='additive')
        seasonal = decomposition.seasonal
        
        # Calculate differences
        val_t0 = df[value_col].iloc[0]
        val_t1 = df[value_col].iloc[-1]
        seasonal_t0 = seasonal[0]
        seasonal_t1 = seasonal[-1]
        
        total_diff = val_t1 - val_t0
        seasonal_diff = seasonal_t1 - seasonal_t0
        
        # Calculate fraction of change due to seasonality
        if abs(total_diff) > 1e-9:
            fraction = seasonal_diff / total_diff
        else:
            fraction = 0.0
        
        return {
            "seasonal_diff": float(seasonal_diff),
            "total_diff": float(total_diff),
            "fraction_of_total_diff": float(fraction)
        }
    except Exception as e:
        return {"error": f"Error in seasonal decomposition: {str(e)}"}

def quantify_event_impact(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    window_before: int = 7,
    window_after: int = 7
) -> pd.DataFrame:
    """
    Estimate the impact of events on a metric by comparing before/after windows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data
    event_df : pd.DataFrame
        DataFrame with events, must have date_col
    date_col : str, default "date"
        Column with dates
    value_col : str, default "value"
        Column with metric values
    window_before : int, default 7
        Number of days to include in the before-event window
    window_after : int, default 7
        Number of days to include in the after-event window

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per event, including:
        - date: Event date
        - before_avg: Average value before event
        - after_avg: Average value after event
        - absolute_impact: after_avg - before_avg
        - percent_impact: Percent change from before to after
    """
    # Ensure date columns are datetime
    df = df.copy()
    event_df = event_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    event_df[date_col] = pd.to_datetime(event_df[date_col])
    
    # Sort by date
    df.sort_values(date_col, inplace=True)
    
    # Create a result DataFrame
    results = []
    
    # Process each event
    for _, event_row in event_df.iterrows():
        event_date = event_row[date_col]
        
        # Calculate windows
        before_start = event_date - pd.Timedelta(days=window_before)
        after_end = event_date + pd.Timedelta(days=window_after)
        
        # Get data from before/after windows
        before_data = df[(df[date_col] >= before_start) & (df[date_col] < event_date)]
        after_data = df[(df[date_col] > event_date) & (df[date_col] <= after_end)]
        
        # Calculate averages
        before_avg = before_data[value_col].mean() if not before_data.empty else np.nan
        after_avg = after_data[value_col].mean() if not after_data.empty else np.nan
        
        # Calculate impact
        abs_impact = after_avg - before_avg if not (pd.isna(before_avg) or pd.isna(after_avg)) else np.nan
        pct_impact = (abs_impact / before_avg * 100) if not pd.isna(abs_impact) and before_avg != 0 else np.nan
        
        # Store results
        event_dict = {
            "date": event_date,
            "before_avg": before_avg,
            "after_avg": after_avg,
            "absolute_impact": abs_impact,
            "percent_impact": pct_impact
        }
        
        # Add any additional columns from the event row
        for col in event_df.columns:
            if col != date_col and col not in event_dict:
                event_dict[col] = event_row[col]
        
        results.append(event_dict)
    
    # Return as DataFrame
    return pd.DataFrame(results)