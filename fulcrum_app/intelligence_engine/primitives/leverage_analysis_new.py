# =============================================================================
# LeverageScenario
#
# This file includes primitives for scenario modeling and driver "leverage":
# sensitivity, simulation, driver targets, ROI, synergy among drivers, etc.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _apply_linear_model(drivers, coefficients):
    """
    For a linear model: metric = sum(coeff_i * driver_i).
    """
    total = 0
    for k, v in drivers.items():
        coef = coefficients.get(k, 0)
        total += coef * v
    return total

# -----------------------------------------------------------------------------
# Main Analysis Functions
# -----------------------------------------------------------------------------

def calculate_driver_sensitivity(coefficient, current_value):
    """
    Purpose: Measures how a 1-unit change in a driver affects the target metric, or 1% change if needed.

    Implementation Details:
    1. For linear model, sensitivity = coefficient.
    2. For a multiplicative model, we might approximate partial derivative.
    3. Return numeric sensitivity.

    Parameters
    ----------
    coefficient : float
    current_value : float  # optional usage

    Returns
    -------
    float
    """
    return coefficient


def simulate_driver_scenarios(base_drivers, coefficients, scenarios):
    """
    Purpose: Re-run forecast/regression given hypothetical driver changes (+5%, etc.).

    Implementation Details:
    1. Accept a dict of base_drivers: {driver_name: current_value}.
    2. For each scenario, apply driver changes to base_drivers.
    3. Compute new metric = sum(coeff_i * new_driver_value).
    4. Return DataFrame: scenario, new_metric.

    Parameters
    ----------
    base_drivers : dict
    coefficients : dict
    scenarios : list of dict
        e.g. [{'name': 'Scenario A', 'changes': {'driverX': 1.05}}, ...]

    Returns
    -------
    pd.DataFrame
        [scenario, new_metric]
    """
    rows = []
    for sc in scenarios:
        name = sc['name']
        changes = sc['changes']
        new_drivers = {}
        for d, val in base_drivers.items():
            factor = changes.get(d, 1.0)
            new_drivers[d] = val * factor
        new_metric = _apply_linear_model(new_drivers, coefficients)
        rows.append({'scenario': name, 'new_metric': new_metric})
    return pd.DataFrame(rows)


def backcalculate_driver_targets(target_metric_value, coefficients, current_drivers):
    """
    Purpose: Solve for the driver changes needed to hit a desired metric target (linear).

    Implementation Details:
    1. If metric = sum(coeff_i * driver_i), and we want metric = target_metric_value,
       we can solve for at most one unknown if all others are fixed. For multiple unknowns,
       we'd do numeric approach or ask user to specify one driver to adjust.

    Parameters
    ----------
    target_metric_value : float
    coefficients : dict
    current_drivers : dict

    Returns
    -------
    dict
        {driver_name: required_value}
    """
    # For demonstration, we'll assume there's exactly one driver with a non-zero coefficient
    # that we want to solve for. All others remain fixed.
    # If there's more than one, we'll do a naive approach and pick the largest coefficient driver.
    max_coef_driver = max(coefficients, key=coefficients.get)
    total_fixed_contribution = 0
    for d, c in coefficients.items():
        if d != max_coef_driver:
            total_fixed_contribution += c * current_drivers[d]

    needed = (target_metric_value - total_fixed_contribution) / coefficients[max_coef_driver]
    new_values = dict(current_drivers)
    new_values[max_coef_driver] = needed
    return new_values


def evaluate_driver_adjustment_costs(driver_changes, cost_per_unit):
    """
    Purpose: Combine cost/difficulty data to produce an ROI or feasibility measure.

    Implementation Details:
    1. For each driver, define cost per unit change.
    2. Multiply cost * required change => total cost.
    3. ROI = ?

    For demonstration, we'll just return total cost.
    """
    results = []
    for d, change in driver_changes.items():
        cost = cost_per_unit.get(d, 0) * change
        results.append({'driver': d, 'change': change, 'cost': cost})
    return pd.DataFrame(results)


def rank_drivers_by_leverage(drivers_sensitivity, cost_dict):
    """
    Purpose: Rank potential driver adjustments by net impact or cost-effectiveness.

    Implementation Details:
    1. For each driver, sensitivity = coefficient, cost = cost_dict.get(driver).
    2. net_effectiveness = sensitivity / cost (naive).
    3. Sort descending.

    Parameters
    ----------
    drivers_sensitivity : dict
        {driver: sensitivity}
    cost_dict : dict
        {driver: cost_per_unit}

    Returns
    -------
    pd.DataFrame
        [driver, sensitivity, cost, net_effectiveness]
    """
    rows = []
    for d, sens in drivers_sensitivity.items():
        c = cost_dict.get(d, 1)
        net_eff = sens / c if c != 0 else np.inf
        rows.append({
            'driver': d,
            'sensitivity': sens,
            'cost': c,
            'net_effectiveness': net_eff
        })
    df = pd.DataFrame(rows)
    df.sort_values(by='net_effectiveness', ascending=False, inplace=True)
    return df


def analyze_cross_driver_effects(df, drivers, outcome_col='metric'):
    """
    Purpose: Check synergy or conflicts if multiple drivers move together.

    Implementation Details:
    1. If the model includes interaction terms, we can measure synergy.
    2. For demonstration, just return placeholder synergy factor.

    Parameters
    ----------
    df : pd.DataFrame
    drivers : list
    outcome_col : str

    Returns
    -------
    dict
        e.g. {'synergy_factor': 1.2}
    """
    return {'synergy_factor': 1.2}


def identify_improvement_headroom(current_values, max_feasible):
    """
    Purpose: Compare each driver's current level to a max feasible level.

    Implementation Details:
    1. headroom = max_feasible_value - current_value.
    """
    results = []
    for d, val in current_values.items():
        feasible = max_feasible.get(d, val)
        headroom = feasible - val
        results.append({
            'driver': d,
            'current_value': val,
            'max_feasible': feasible,
            'headroom': headroom
        })
    return pd.DataFrame(results)


def evaluate_implementation_constraints(scenario_changes, constraints):
    """
    Purpose: Check if scenario exceeds budgets/capacity.

    Implementation Details:
    1. Compare scenario changes vs. constraints (budget, capacity).
    2. If exceed, mark invalid or clip.
    """
    # Placeholder
    feasible = True
    for k, v in constraints.items():
        if scenario_changes.get(k, 0) > v:
            feasible = False
    return {'feasible': feasible}


def rank_improvement_opportunities(scenarios):
    """
    Purpose: Output a final priority list of scenario changes (ROI, synergy, feasibility).

    Implementation Details:
    1. Summarize each scenario's scores, sort descending.

    NOTE: We just do a placeholder approach. Real approach would combine multiple metrics.

    Parameters
    ----------
    scenarios : list of dict
        e.g. [{'name': 'Scenario A', 'roi': 2.0, 'feasibility': True, 'synergy': 1.1}, ...]

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(scenarios)
    df['opportunity_score'] = df['roi'] * df['synergy'] * df['feasibility'].apply(lambda x: 1 if x else 0)
    df.sort_values(by='opportunity_score', ascending=False, inplace=True)
    return df
