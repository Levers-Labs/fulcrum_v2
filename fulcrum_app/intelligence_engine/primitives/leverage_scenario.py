import pandas as pd
import numpy as np
from typing import Dict, List, Optional
# from .performance or from your other modules

def calculate_driver_sensitivity(
    model, 
    current_point: Dict[str, float]
) -> Dict[str, float]:
    """
    For a linear model y = b0 + b1*x1 + ... + bn*xn, the "sensitivity"
    of y wrt xj is b_j in the simplest sense.

    If the model is non-linear, we might evaluate partial derivatives at current_point.

    Parameters
    ----------
    model : a fitted regression model (e.g. scikit-learn) with .coef_ and .feature_names_in_ 
            or you supply them separately.
    current_point : dict
        { 'driverA': valueA, 'driverB': valueB, ... } 
        The current levels of each driver, used only if we needed partial derivatives for a non-linear model.

    Returns
    -------
    Dict[str, float]
        driver_name -> sensitivity (coefficient or partial derivative at current_point).

    Notes
    -----
    - For a purely linear model, sensitivity ~ model.coef_[driver_index].
    - If model is something else, we'd do partial derivatives at current_point. This code only does linear or 
      returns the model’s direct coefficients as "sensitivity."
    """
    # We'll assume the model is linear from scikit-learn or similar 
    # and has .coef_ plus an attribute that maps coefficients to driver names.
    # For scikit-learn <=1.0, we might have to store feature names separately.
    # If the model doesn't store them, we can accept them as an argument.
    
    # For demonstration, let's assume model.feature_names_in_ is the driver list in order:
    driver_names = list(model.feature_names_in_)
    betas = model.coef_
    # if the model is multi-output or something, we assume single-output => betas is 1D
    # We'll build a result dict
    result = {}
    for i, drv in enumerate(driver_names):
        result[drv] = float(betas[i])  # sensitivity
    return result

def simulate_driver_scenarios(
    model,
    current_point: Dict[str, float],
    scenario: Dict[str, float],
    relative: bool = True
) -> float:
    """
    Re-run the model with certain driver shifts. If relative=True,
    scenario[driver] is a fraction => x_j_new = x_j_current * (1 + scenario[driver]).
    If relative=False, scenario[driver] is an absolute shift => x_j_new = x_j_current + shift.

    Parameters
    ----------
    model : a fitted linear model with .predict
    current_point : dict
        driver -> current_value
    scenario : dict
        driver -> shift (relative or absolute)
    relative : bool, default=True
        If True, we interpret scenario values as percentages (0.05 => +5%).
        If False, interpret them as absolute offsets.

    Returns
    -------
    float
        The new predicted y under this scenario.

    Example
    -------
    scenario={'driverA': +0.10, 'driverB': -0.05} => driverA up 10%, driverB down 5% from current.
    """
    # build new driver vector
    driver_names = list(model.feature_names_in_)
    new_x = []
    for drv in driver_names:
        cur_val = current_point.get(drv, 0.0)
        shift_val = scenario.get(drv, 0.0)
        if relative:
            new_val = cur_val * (1.0 + shift_val)
        else:
            new_val = cur_val + shift_val
        new_x.append(new_val)
    new_x = np.array(new_x).reshape(1, -1)
    # predict
    y_new = model.predict(new_x)[0]
    return float(y_new)

def backcalculate_driver_targets(
    model,
    current_point: Dict[str, float],
    target_y: float,
    driver_name: str,
    relative: bool = False
) -> float:
    """
    Solve for how much driver_name must change (absolute or relative) 
    to hit target_y if all other drivers are fixed.

    For linear model: target_y = b0 + b_j*(x_j + delta) + sum_{i!=j} b_i*x_i
    => delta = (target_y - b0 - sum_{i!=j} b_i*x_i)/b_j - x_j.

    If relative=True, we solve fraction => x_j_new = x_j*(1 + fraction).
    Then fraction = x_j_new/x_j - 1 => (some algebra).

    Parameters
    ----------
    model : fitted linear model
    current_point : dict
    target_y : float
    driver_name : str
        The driver we want to solve for.
    relative : bool, default=False
        If True, return the fraction (x_j_new - x_j)/ x_j. If False, return absolute delta.

    Returns
    -------
    float
        The required shift or fraction for driver_name. 
        If it’s impossible (b_j=0), we might return None or raise an error.
    """
    driver_names = list(model.feature_names_in_)
    betas = model.coef_
    intercept = model.intercept_

    if driver_name not in driver_names:
        raise ValueError(f"Driver {driver_name} not found in model features {driver_names}.")

    # compute the sum_of_other drivers
    sum_others = 0.0
    for i, drv in enumerate(driver_names):
        if drv != driver_name:
            sum_others += betas[i] * current_point.get(drv, 0.0)

    j_index = driver_names.index(driver_name)
    b_j = betas[j_index]
    if abs(b_j) < 1e-12:
        return None  # can't solve if coefficient is zero => or means no influence

    x_j = current_point.get(driver_name, 0.0)
    # linear eqn: target_y = intercept + b_j*(x_j + delta) + sum_others
    # => b_j*delta = target_y - intercept - sum_others - b_j*x_j
    # => delta = [target_y - intercept - sum_others - b_j*x_j]/ b_j
    numerator = target_y - intercept - sum_others - b_j*x_j
    delta = numerator / b_j

    if relative:
        # fraction => (x_j + delta)/ x_j - 1
        # if x_j=0 => we do a special approach
        if abs(x_j) < 1e-12:
            # can't do relative shift from 0 => define something or return None
            return None
        fraction = (x_j + delta)/ x_j - 1.0
        return fraction
    else:
        return delta

def evaluate_driver_adjustment_costs(
    scenario: Dict[str, float],
    cost_map: Dict[str, float],
    relative: bool = True
) -> Dict[str, float]:
    """
    Combine cost/difficulty data with required changes => ROI or absolute cost. 
    For each driver j in scenario, we have scenario[j] = fraction or delta. 
    cost_map[driver]= cost_per_unit (if relative? cost per 1%?).

    Parameters
    ----------
    scenario : dict
        driver -> shift. If relative=True, shift is fraction like +0.05 => +5%.
    cost_map : dict
        driver -> cost per unit. 
        E.g. if driver='budget', cost per +1% is $10k. Then scenario['budget']=0.1 => cost= 0.1 * $10k = $1k.
    relative : bool, default=True
        If True, scenario shift is fraction => cost= shift * cost_map. 
        If False, scenario shift is absolute => cost= shift * cost_map (unit).

    Returns
    -------
    Dict[str, float]
        driver -> cost_of_that_shift

    Example
    -------
    scenario={'driverA': +0.1}, cost_map={'driverA': 10000} => cost= +0.1*10000= 1000 => means $1k for a 10% driver increase.
    """
    costs = {}
    for drv, shift in scenario.items():
        # If not in cost_map => cost=0 or we skip
        if drv not in cost_map:
            costs[drv] = 0.0
        else:
            cost_per_unit = cost_map[drv]
            # If relative, shift is fraction => cost= fraction * cost_per_unit
            # If absolute => cost= shift * cost_per_unit
            # either way the formula is the same, just different meaning of shift
            cost_val = shift * cost_per_unit
            costs[drv] = cost_val
    return costs

def rank_drivers_by_leverage(
    model, 
    current_point: Dict[str, float],
    cost_map: Dict[str, float]
) -> pd.DataFrame:
    """
    Sort drivers by net impact potential or ROI. For each driver j, ROI ~ coefficient / cost_per_unit.

    Parameters
    ----------
    model : fitted linear model
    current_point : dict
        for reference if we needed partial derivatives in a non-linear model.
    cost_map : dict
        driver -> cost_per_unit

    Returns
    -------
    pd.DataFrame
        columns: [driver, coefficient, cost_per_unit, roi], sorted descending by roi.

    Example
    -------
    If driverA has b_j=2.0, cost_map= {driverA: 10.0}, => roi=2/10=0.2 => effect on y per cost unit.
    """
    driver_names = list(model.feature_names_in_)
    betas = model.coef_

    rows = []
    for i, drv in enumerate(driver_names):
        b_j = betas[i]
        cost_j = cost_map.get(drv, np.inf)
        if abs(cost_j) < 1e-12:
            roi = np.inf
        else:
            roi = b_j/cost_j
        rows.append({
            "driver": drv,
            "coefficient": float(b_j),
            "cost_per_unit": float(cost_j),
            "roi": float(roi)
        })
    df = pd.DataFrame(rows)
    # sort descending by roi
    df.sort_values("roi", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def analyze_cross_driver_effects(
    model, 
    drivers: list
) -> dict:
    """
    If the model has interaction terms, see synergy. We look for terms like X1:X2 in the coefficient set.

    Parameters
    ----------
    model : a statsmodels or patsy-based model that has param names indicating interactions, 
            e.g. 'X1:X2'.
    drivers : list
        base driver names, e.g. ['X1','X2'].

    Returns
    -------
    dict
        { 'X1_X2': coefficient_of_interaction, ...}

    Notes
    -----
    - scikit-learn doesn't automatically name interaction terms. We might rely on 
      a formula-based approach or custom pipeline to store them. 
      This function is domain-specific. 
    """
    # We'll assume we stored the model param names in model.params if it's a statsmodels object.
    # If it's scikit-learn, you might not have these names unless you used PolynomialFeatures or patsy.
    results = {}
    if not hasattr(model, "params"):
        return {"error": "Model does not have .params (statsmodels?). Cannot detect synergy."}

    for param_name in model.params.index:
        if ":" in param_name:
            # that indicates an interaction in a patsy formula
            coef_val = model.params[param_name]
            results[param_name.replace(":","_")] = float(coef_val)
    return results

def identify_improvement_headroom(
    current_point: Dict[str, float],
    max_feasible: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare current driver level to its max feasible level. 
    headroom = max_feasible[drv] - current_point[drv].

    Parameters
    ----------
    current_point : dict
        { driver: current_value }
    max_feasible : dict
        { driver: maximum_plausible_value }

    Returns
    -------
    Dict[str, float]
        { driver: headroom_value }. If driver not in max_feasible => None or skip.
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
    Check if a scenario exceeds budgets/capacity for each driver. 
    If scenario[driver] > constraints[driver], we clamp to constraints[driver] 
    or mark invalid.

    Parameters
    ----------
    scenario : dict
        driver-> shift (fraction or absolute).
    constraints : dict
        driver-> max allowable shift (fraction or absolute).
    relative : bool, default=True
        If True, scenario shifts are fractional. If scenario[driver] > constraints[driver], 
        we clamp or skip.
    
    Returns
    -------
    dict
        The final feasible scenario after applying constraints, possibly clamping. 
        If a shift is negative and there's no negative constraint, or it’s below -constraints[driver], 
        we handle that too. 
        You might also add a "valid" flag if something is infeasible.

    Example
    -------
    scenario={'driverA': +0.2}, constraints={'driverA':0.15} => clamp to +0.15 if scenario is bigger.
    """
    feasible_scenario = {}
    for drv, shift in scenario.items():
        max_shift = constraints.get(drv, None)
        if max_shift is None:
            # no constraint => we allow it
            feasible_scenario[drv] = shift
        else:
            # clamp in [-max_shift, +max_shift], or define some approach
            # if you want unidirectional constraint => define domain specifically
            # We'll assume max_shift is the absolute or fractional limit in the positive direction,
            # and -max_shift is limit in negative direction
            if shift > max_shift:
                feasible_scenario[drv] = max_shift
            elif shift < -max_shift:
                feasible_scenario[drv] = -max_shift
            else:
                feasible_scenario[drv] = shift
    return feasible_scenario

def rank_improvement_opportunities(
    model,
    current_point: Dict[str, float],
    cost_map: Dict[str, float],
    feasible_shifts: Dict[str, float],
    default_shift: float = 0.1
) -> pd.DataFrame:
    """
    Produce a final priority list of single-driver scenario changes 
    (like +10% to each driver), evaluate cost vs. benefit => rank.

    Parameters
    ----------
    model : fitted linear model
    current_point : dict
    cost_map : dict
        driver-> cost_per 1.0 fractional shift
    feasible_shifts : dict
        driver-> max fractional shift possible
    default_shift : float, default=0.1
        If feasible_shifts not defined for a driver, we assume we can do up to +10%.

    Returns
    -------
    pd.DataFrame
        columns = [driver, shift, new_y, delta_y, cost, roi], sorted descending by roi.

    Example
    -------
    - We do a scenario for each driver => + shift. 
      new_y = simulate_driver_scenarios(...). 
      cost= shift * cost_map. 
      delta_y= new_y - baseline_y. 
      roi= delta_y/cost.
    """
    # baseline
    import numpy as np
    driver_names = list(model.feature_names_in_)
    x_array = [current_point.get(drv,0.0) for drv in driver_names]
    baseline_y = model.predict([x_array])[0]

    rows = []
    for drv in driver_names:
        max_shift = feasible_shifts.get(drv, default_shift)
        scenario = {drv: max_shift}
        # new y
        new_y = simulate_driver_scenarios(model, current_point, scenario, relative=True)
        delta_y = new_y - baseline_y
        # cost
        cost_per_unit = cost_map.get(drv, np.inf)
        cost_val = max_shift * cost_per_unit
        if abs(cost_val) < 1e-12:
            roi = np.inf if delta_y>0 else -np.inf
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
