# file: leverage_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_driver_sensitivity(model, current_point: Dict[str, float]) -> Dict[str, float]:
    """
    For a linear model y = b0 + sum_j(b_j * x_j), the sensitivity wrt x_j = b_j (the coefficient).
    This function returns {driver_name: coefficient}, using model.feature_names_in_ if present.
    """
    driver_names = list(getattr(model, "feature_names_in_", []))
    if not driver_names:
        # fallback if model doesn't store it
        driver_names = [f"x_{i}" for i in range(len(model.coef_))]
    betas = model.coef_
    return {drv: float(betas[i]) for i, drv in enumerate(driver_names)}


def simulate_driver_scenarios(
    model,
    current_point: Dict[str, float],
    scenario: Dict[str, float],
    relative: bool = True
) -> float:
    """
    Evaluate model prediction under scenario driver shifts.
    If relative=True, scenario[drv] is a fraction => new_x = curr_x*(1+ fraction).
    If relative=False, scenario[drv] is an absolute shift => new_x = curr_x + shift.
    """
    driver_names = list(getattr(model, "feature_names_in_", []))
    if not driver_names:
        driver_names = list(current_point.keys())  # fallback

    new_x = []
    for drv in driver_names:
        cur_val = current_point.get(drv, 0.0)
        shift_val = scenario.get(drv, 0.0)
        if relative:
            new_val = cur_val * (1.0 + shift_val)
        else:
            new_val = cur_val + shift_val
        new_x.append(new_val)

    arr = np.array(new_x).reshape(1, -1)
    return float(model.predict(arr)[0])


def backcalculate_driver_targets(
    model,
    current_point: Dict[str, float],
    target_y: float,
    driver_name: str,
    relative: bool = False
) -> Optional[float]:
    """
    Solve for how much 'driver_name' must change so the model hits 'target_y' if all other drivers are fixed.
    For linear model y = b0 + sum_j(b_j * x_j):
      needed_delta = (target_y - b0 - sum_{i!=j} b_i*x_i)/ b_j  - x_j   (if relative=False)
    Or a fraction if relative=True.
    Returns None if b_j=0 or x_j=0 in relative mode.
    """
    betas = model.coef_
    driver_names = list(getattr(model, "feature_names_in_", []))
    intercept = getattr(model, "intercept_", 0.0)

    if not driver_names:
        # fallback
        driver_names = [f"x_{i}" for i in range(len(betas))]

    if driver_name not in driver_names:
        raise ValueError(f"Driver '{driver_name}' not found in model features: {driver_names}")

    sum_others = 0.0
    j_index = driver_names.index(driver_name)
    b_j = betas[j_index]
    x_j = current_point.get(driver_name, 0.0)

    # sum of other drivers
    for i, dname in enumerate(driver_names):
        if dname != driver_name:
            sum_others += betas[i] * current_point.get(dname, 0.0)

    if abs(b_j) < 1e-12:
        return None

    numerator = target_y - intercept - sum_others - b_j*x_j
    delta = numerator / b_j

    if not relative:
        return delta
    else:
        if abs(x_j) < 1e-12:
            return None
        # fraction = (x_j + delta)/ x_j - 1
        return (x_j + delta)/x_j - 1.0


def evaluate_driver_adjustment_costs(
    scenario: Dict[str, float],
    cost_map: Dict[str, float],
    relative: bool = True
) -> Dict[str, float]:
    """
    If scenario[driver]= fraction or delta, cost_map[driver]= cost per fractional or absolute unit.
    Returns the cost of each driver in the scenario.
    """
    costs = {}
    for drv, shift in scenario.items():
        cost_per_unit = cost_map.get(drv, 0.0)
        costs[drv] = shift * cost_per_unit
    return costs


def rank_drivers_by_leverage(
    model, 
    current_point: Dict[str, float],
    cost_map: Dict[str, float]
) -> pd.DataFrame:
    """
    Sort drivers by 'ROI' ~ coefficient / cost_per_unit for a linear model.
    If cost_per_unit=0 => ROI=∞ if coefficient>0 else -∞ if coefficient<0.
    """
    import pandas as pd
    driver_names = list(getattr(model, "feature_names_in_", []))
    if not driver_names:
        driver_names = list(current_point.keys())

    betas = model.coef_
    rows = []
    for i, drv in enumerate(driver_names):
        b_j = betas[i]
        cost_j = cost_map.get(drv, np.inf)
        if abs(cost_j) < 1e-12:
            roi = np.inf if b_j > 0 else -np.inf
        else:
            roi = b_j / cost_j
        rows.append({"driver": drv, "coefficient": float(b_j), "cost_per_unit": float(cost_j), "roi": float(roi)})

    df = pd.DataFrame(rows)
    df.sort_values("roi", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def analyze_cross_driver_effects(model, drivers: list) -> dict:
    """
    If the model has interaction terms named e.g. 'X1:X2', parse the model params to find synergy.
    For scikit-learn standard LinearRegression, we typically won't have param names for interactions
    unless a pipeline or patsy used. So might need a statsmodels approach.
    """
    if not hasattr(model, "params"):
        return {"error": "Model does not have .params (likely scikit-learn). No synergy detection."}

    results = {}
    for param_name in model.params.index:
        if ":" in param_name:
            coef_val = model.params[param_name]
            results[param_name.replace(":","_")] = float(coef_val)
    return results


def identify_improvement_headroom(
    current_point: Dict[str, float],
    max_feasible: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare current driver level to a max feasible level => return the difference.
    If driver not in max_feasible => set None for that driver.
    """
    result = {}
    for drv, cur_val in current_point.items():
        if drv in max_feasible:
            result[drv] = max_feasible[drv] - cur_val
        else:
            result[drv] = None
    return result


def evaluate_implementation_constraints(
    scenario: Dict[str, float],
    constraints: Dict[str, float],
    relative: bool = True
) -> Dict[str, float]:
    """
    Check if each scenario shift is within constraints. If shift>max => clamp to max.
    If shift< -max => clamp to -max. 
    """
    feasible = {}
    for drv, shift in scenario.items():
        max_shift = constraints.get(drv, None)
        if max_shift is None:
            feasible[drv] = shift
        else:
            feasible[drv] = min(max_shift, shift)
            if feasible[drv] < -max_shift:
                feasible[drv] = -max_shift
    return feasible


def rank_improvement_opportunities(
    model,
    current_point: Dict[str, float],
    cost_map: Dict[str, float],
    feasible_shifts: Dict[str, float],
    default_shift: float = 0.1
) -> pd.DataFrame:
    """
    For each driver, assume a shift up to feasible_shifts[driver] (or default_shift),
    compute delta_y, compute cost => rank by ROI = delta_y / cost.
    """
    import pandas as pd
    driver_names = list(getattr(model, "feature_names_in_", []))
    if not driver_names:
        driver_names = list(current_point.keys())

    # baseline
    baseline_arr = [current_point.get(drv, 0.0) for drv in driver_names]
    baseline_y = float(model.predict([baseline_arr])[0])

    rows = []
    for drv in driver_names:
        max_shift = feasible_shifts.get(drv, default_shift)
        scenario = {drv: max_shift}
        # simulate
        new_x = []
        for d2 in driver_names:
            cur_val = current_point.get(d2, 0.0)
            if d2 == drv:
                new_val = cur_val * (1.0 + max_shift)  # always relative in this logic
            else:
                new_val = cur_val
            new_x.append(new_val)
        new_y = float(model.predict([new_x])[0])

        delta_y = new_y - baseline_y
        cost_per_unit = cost_map.get(drv, np.inf)
        cost_val = max_shift * cost_per_unit
        if abs(cost_val) < 1e-12:
            roi = np.inf if delta_y > 0 else -np.inf
        else:
            roi = delta_y / cost_val

        rows.append({
            "driver": drv,
            "shift": max_shift,
            "new_y": new_y,
            "delta_y": delta_y,
            "cost": cost_val,
            "roi": roi
        })

    df = pd.DataFrame(rows)
    df.sort_values("roi", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df