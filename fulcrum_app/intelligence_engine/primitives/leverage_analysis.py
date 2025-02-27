# =============================================================================
# LeverageScenario
#
# This file includes primitives for scenario modeling and driver "leverage" analysis:
# - Sensitivity calculation for driver impacts on a metric
# - Simulation of different driver scenarios
# - Backcalculation of required driver values to reach targets
# - Cost-benefit analysis of driver adjustments
# - Ranking drivers by ROI or impact potential
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _apply_linear_model(
    drivers: Dict[str, float], 
    coefficients: Dict[str, float],
    intercept: float = 0.0
) -> float:
    """
    Apply a linear model to calculate metric value from driver values.
    
    Parameters
    ----------
    drivers : Dict[str, float]
        Dictionary of driver names and values
    coefficients : Dict[str, float]
        Dictionary of driver names and coefficients
    intercept : float, default=0.0
        Intercept term in the model
        
    Returns
    -------
    float
        Calculated metric value: intercept + sum(coefficient_i * driver_i)
    """
    total = intercept
    for driver, value in drivers.items():
        coef = coefficients.get(driver, 0.0)
        total += coef * value
    return total

def _apply_nonlinear_model(
    drivers: Dict[str, float], 
    model_function: callable
) -> float:
    """
    Apply a nonlinear model to calculate metric value from driver values.
    
    Parameters
    ----------
    drivers : Dict[str, float]
        Dictionary of driver names and values
    model_function : callable
        Function that takes drivers dictionary and returns metric value
        
    Returns
    -------
    float
        Calculated metric value from the model function
    """
    return model_function(drivers)

# -----------------------------------------------------------------------------
# Main Analysis Functions
# -----------------------------------------------------------------------------

def calculate_driver_sensitivity(
    coefficients: Dict[str, float],
    current_values: Dict[str, float],
    model_type: str = "linear",
    model_function: Optional[callable] = None,
    delta_pct: float = 1.0
) -> Dict[str, float]:
    """
    Calculate the sensitivity of a metric to changes in each driver.
    
    Parameters
    ----------
    coefficients : Dict[str, float]
        Dictionary of driver coefficients for linear models
    current_values : Dict[str, float]
        Dictionary of current driver values
    model_type : str, default="linear"
        Type of model: 'linear' or 'nonlinear'
    model_function : Optional[callable], default=None
        Function for nonlinear models that takes drivers dictionary and returns metric value
    delta_pct : float, default=1.0
        Percentage change to apply for sensitivity calculation (1.0 = 1%)
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping each driver to its sensitivity value
        
    Notes
    -----
    For linear models, sensitivity is simply the coefficient.
    For nonlinear models, sensitivity is calculated as the partial derivative
    approximated by a small change in the driver value.
    """
    sensitivities = {}
    
    if model_type == "linear":
        # For linear models, sensitivity = coefficient
        for driver, coef in coefficients.items():
            sensitivities[driver] = coef
    
    elif model_type == "nonlinear":
        if model_function is None:
            raise ValueError("model_function must be provided for nonlinear models")
        
        # Calculate base metric value
        base_value = model_function(current_values)
        
        # Calculate sensitivity for each driver
        for driver, value in current_values.items():
            delta = value * (delta_pct / 100.0)
            if delta == 0:
                # Use a small absolute delta if current value is zero
                delta = 0.001
            
            # Create adjusted driver values
            adjusted_values = current_values.copy()
            adjusted_values[driver] = value + delta
            
            # Calculate new metric value
            new_value = model_function(adjusted_values)
            
            # Sensitivity = (change in metric) / (change in driver)
            sensitivity = (new_value - base_value) / delta
            sensitivities[driver] = sensitivity
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'linear' or 'nonlinear'.")
    
    return sensitivities


def simulate_driver_scenarios(
    base_drivers: Dict[str, float],
    coefficients: Dict[str, float],
    scenarios: List[Dict[str, Any]],
    model_type: str = "linear",
    model_function: Optional[callable] = None,
    intercept: float = 0.0
) -> pd.DataFrame:
    """
    Simulate different driver scenarios and predict the resulting metric values.
    
    Parameters
    ----------
    base_drivers : Dict[str, float]
        Dictionary of base driver values
    coefficients : Dict[str, float]
        Dictionary of driver coefficients for linear models
    scenarios : List[Dict[str, Any]]
        List of scenario dictionaries. Each scenario should have:
        - 'name': Scenario name
        - 'changes': Dict mapping driver names to multipliers
    model_type : str, default="linear"
        Type of model: 'linear' or 'nonlinear'
    model_function : Optional[callable], default=None
        Function for nonlinear models that takes drivers dictionary and returns metric value
    intercept : float, default=0.0
        Intercept term for linear models
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'scenario': Scenario name
        - 'new_metric': Predicted metric value
        - '{driver}_value': New value for each driver
        - '{driver}_change_pct': Percentage change for each driver
        
    Example
    -------
    >>> base_drivers = {'ad_spend': 100, 'price': 50}
    >>> coefficients = {'ad_spend': 0.5, 'price': -0.2}
    >>> scenarios = [
    ...     {'name': 'Increase Ad Spend', 'changes': {'ad_spend': 1.2}},
    ...     {'name': 'Decrease Price', 'changes': {'price': 0.9}}
    ... ]
    >>> simulate_driver_scenarios(base_drivers, coefficients, scenarios)
    """
    rows = []
    
    for scenario in scenarios:
        scenario_name = scenario.get('name', f"Scenario {len(rows) + 1}")
        changes = scenario.get('changes', {})
        
        # Create new driver values
        new_drivers = {}
        driver_change_pcts = {}
        
        for driver, base_val in base_drivers.items():
            factor = changes.get(driver, 1.0)
            new_val = base_val * factor
            new_drivers[driver] = new_val
            
            # Calculate percentage change
            if base_val != 0:
                change_pct = ((new_val / base_val) - 1) * 100
            else:
                change_pct = np.inf if new_val > 0 else -np.inf if new_val < 0 else 0
            
            driver_change_pcts[driver] = change_pct
        
        # Calculate new metric value
        if model_type == "linear":
            new_metric = _apply_linear_model(new_drivers, coefficients, intercept)
        elif model_type == "nonlinear":
            if model_function is None:
                raise ValueError("model_function must be provided for nonlinear models")
            new_metric = model_function(new_drivers)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'linear' or 'nonlinear'.")
        
        # Create row
        row = {
            'scenario': scenario_name,
            'new_metric': new_metric
        }
        
        # Add driver values and changes
        for driver, value in new_drivers.items():
            row[f"{driver}_value"] = value
            row[f"{driver}_change_pct"] = driver_change_pcts[driver]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def backcalculate_driver_targets(
    target_metric_value: float,
    coefficients: Dict[str, float],
    current_drivers: Dict[str, float],
    intercept: float = 0.0,
    drivers_to_solve: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Solve for the driver values needed to reach a target metric value.
    
    Parameters
    ----------
    target_metric_value : float
        Desired metric value
    coefficients : Dict[str, float]
        Dictionary of driver coefficients
    current_drivers : Dict[str, float]
        Dictionary of current driver values
    intercept : float, default=0.0
        Intercept term in the model
    drivers_to_solve : Optional[List[str]], default=None
        Specific drivers to solve for. If None, solves for driver with largest coefficient.
    constraints : Optional[Dict[str, Dict[str, float]]], default=None
        Constraints on driver values, e.g. {'driver1': {'min': 0, 'max': 100}}
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping driver names to solutions:
        {
            'driver1': {
                'current': current_value,
                'target': target_value,
                'change': absolute_change,
                'pct_change': percentage_change
            }
        }
    """
    # If no specific drivers to solve for, use driver with largest coefficient
    if drivers_to_solve is None:
        # Find driver with largest coefficient magnitude
        drivers_to_solve = [max(coefficients.items(), key=lambda x: abs(x[1]))[0]]
    
    # Validate inputs
    for driver in drivers_to_solve:
        if driver not in coefficients:
            raise ValueError(f"Driver '{driver}' not found in coefficients")
        if driver not in current_drivers:
            raise ValueError(f"Driver '{driver}' not found in current_drivers")
    
    # Calculate the fixed contribution from drivers we're not solving for
    fixed_contribution = intercept
    for driver, value in current_drivers.items():
        if driver not in drivers_to_solve:
            coef = coefficients.get(driver, 0.0)
            fixed_contribution += coef * value
    
    solutions = {}
    
    # Solve for each driver
    for driver in drivers_to_solve:
        coef = coefficients[driver]
        current_value = current_drivers[driver]
        
        # Skip if coefficient is zero (no impact)
        if coef == 0:
            solutions[driver] = {
                'current': current_value,
                'target': current_value,
                'change': 0.0,
                'pct_change': 0.0,
                'note': "Coefficient is zero, cannot solve"
            }
            continue
        
        # Calculate target driver value
        target_value = (target_metric_value - fixed_contribution) / coef
        
        # Apply constraints if provided
        if constraints and driver in constraints:
            driver_constraints = constraints[driver]
            if 'min' in driver_constraints and target_value < driver_constraints['min']:
                target_value = driver_constraints['min']
            if 'max' in driver_constraints and target_value > driver_constraints['max']:
                target_value = driver_constraints['max']
        
        # Calculate changes
        change = target_value - current_value
        if current_value != 0:
            pct_change = (change / abs(current_value)) * 100
        else:
            pct_change = None
        
        solutions[driver] = {
            'current': current_value,
            'target': target_value,
            'change': change,
            'pct_change': pct_change
        }
    
    return solutions


def evaluate_driver_adjustment_costs(
    driver_changes: Dict[str, float],
    cost_per_unit: Dict[str, float],
    relative: bool = False
) -> pd.DataFrame:
    """
    Calculate the cost of making specified driver adjustments.
    
    Parameters
    ----------
    driver_changes : Dict[str, float]
        Dictionary mapping driver names to change amounts
    cost_per_unit : Dict[str, float]
        Dictionary mapping driver names to per-unit costs
    relative : bool, default=False
        If True, driver_changes are percentage changes
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'driver': Driver name
        - 'change': Absolute change amount
        - 'cost_per_unit': Cost per unit
        - 'total_cost': Total cost of change
    """
    rows = []
    
    for driver, change in driver_changes.items():
        # Get cost per unit for this driver
        unit_cost = cost_per_unit.get(driver, 0)
        
        # Calculate total cost
        total_cost = unit_cost * change
        
        rows.append({
            'driver': driver,
            'change': change,
            'cost_per_unit': unit_cost,
            'total_cost': total_cost
        })
    
    return pd.DataFrame(rows)


def rank_drivers_by_leverage(
    sensitivities: Dict[str, float],
    current_values: Dict[str, float],
    cost_dict: Dict[str, float],
    constraints: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Rank drivers by their effectiveness or ROI.
    
    Parameters
    ----------
    sensitivities : Dict[str, float]
        Dictionary mapping driver names to sensitivity values
    current_values : Dict[str, float]
        Dictionary of current driver values
    cost_dict : Dict[str, float]
        Dictionary mapping driver names to per-unit costs
    constraints : Optional[Dict[str, Dict[str, float]]], default=None
        Constraints on driver values, e.g. {'driver1': {'min': 0, 'max': 100}}
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'driver': Driver name
        - 'sensitivity': Sensitivity value
        - 'cost_per_unit': Cost per unit
        - 'roi': Sensitivity / cost (higher is better)
        - 'headroom': Room for improvement based on constraints
        sorted by ROI (descending)
    """
    rows = []
    
    for driver, sensitivity in sensitivities.items():
        # Get cost for this driver
        cost = cost_dict.get(driver, 1.0)  # Default to 1.0 if not specified
        
        # Calculate ROI = sensitivity / cost
        roi = sensitivity / cost if cost != 0 else float('inf')
        
        # Calculate headroom based on constraints
        current_value = current_values.get(driver, 0)
        headroom = None
        
        if constraints and driver in constraints:
            driver_constraints = constraints[driver]
            if sensitivity > 0 and 'max' in driver_constraints:
                # Positive sensitivity: increasing driver is good
                headroom = driver_constraints['max'] - current_value
            elif sensitivity < 0 and 'min' in driver_constraints:
                # Negative sensitivity: decreasing driver is good
                headroom = current_value - driver_constraints['min']
        
        rows.append({
            'driver': driver,
            'sensitivity': sensitivity,
            'cost_per_unit': cost,
            'roi': roi,
            'headroom': headroom
        })
    
    # Sort by ROI (descending)
    df = pd.DataFrame(rows)
    return df.sort_values('roi', ascending=False).reset_index(drop=True)


def analyze_cross_driver_effects(
    interaction_coefficients: Dict[str, float],
    driver_names: List[str]
) -> pd.DataFrame:
    """
    Analyze interaction effects between drivers.
    
    Parameters
    ----------
    interaction_coefficients : Dict[str, float]
        Dictionary mapping interaction terms to coefficients
        (e.g., {'driver1:driver2': 0.5})
    driver_names : List[str]
        List of driver names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'driver1': First driver name
        - 'driver2': Second driver name
        - 'interaction_coefficient': Coefficient for the interaction
        - 'synergy': True if the interaction is positive, False otherwise
    """
    rows = []
    
    # Process each interaction term
    for term, coef in interaction_coefficients.items():
        # Parse drivers from interaction term
        parts = term.split(':')
        if len(parts) != 2:
            continue
        
        driver1, driver2 = parts
        
        # Skip if drivers not in the provided list
        if driver1 not in driver_names or driver2 not in driver_names:
            continue
        
        # Determine if the interaction is synergistic (positive) or antagonistic (negative)
        synergy = coef > 0
        
        rows.append({
            'driver1': driver1,
            'driver2': driver2,
            'interaction_coefficient': coef,
            'synergy': synergy
        })
    
    return pd.DataFrame(rows)


def identify_improvement_headroom(
    current_values: Dict[str, float],
    max_feasible: Dict[str, float],
    sensitivities: Dict[str, float]
) -> pd.DataFrame:
    """
    Calculate how much each driver can be improved and the potential impact.
    
    Parameters
    ----------
    current_values : Dict[str, float]
        Dictionary of current driver values
    max_feasible : Dict[str, float]
        Dictionary of maximum feasible values for each driver
    sensitivities : Dict[str, float]
        Dictionary of driver sensitivities
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'driver': Driver name
        - 'current_value': Current driver value
        - 'max_feasible': Maximum feasible value
        - 'headroom': max_feasible - current_value
        - 'sensitivity': Sensitivity value
        - 'potential_impact': headroom * sensitivity
    """
    rows = []
    
    for driver, current in current_values.items():
        # Get maximum feasible value
        max_val = max_feasible.get(driver, current)
        
        # Calculate headroom
        headroom = max_val - current
        
        # Get sensitivity
        sensitivity = sensitivities.get(driver, 0.0)
        
        # Calculate potential impact
        potential_impact = headroom * sensitivity
        
        rows.append({
            'driver': driver,
            'current_value': current,
            'max_feasible': max_val,
            'headroom': headroom,
            'sensitivity': sensitivity,
            'potential_impact': potential_impact
        })
    
    # Sort by potential impact (descending)
    df = pd.DataFrame(rows)
    return df.sort_values('potential_impact', ascending=False).reset_index(drop=True)


def evaluate_implementation_constraints(
    scenario_changes: Dict[str, float],
    constraints: Dict[str, float],
    relative: bool = True
) -> Dict[str, Any]:
    """
    Check if a scenario exceeds implementation constraints.
    
    Parameters
    ----------
    scenario_changes : Dict[str, float]
        Dictionary mapping driver names to change amounts
    constraints : Dict[str, float]
        Dictionary mapping driver names to maximum allowable changes
    relative : bool, default=True
        If True, changes and constraints are in percentage terms
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'feasible': True if all changes are within constraints
        - 'violations': List of drivers that exceed their constraints
        - 'clipped_changes': Dictionary of changes clipped to constraints
    """
    feasible = True
    violations = []
    clipped_changes = {}
    
    for driver, change in scenario_changes.items():
        # Get constraint for this driver
        constraint = constraints.get(driver, float('inf'))
        
        # Check if change exceeds constraint
        if abs(change) > constraint:
            feasible = False
            violations.append(driver)
            
            # Clip change to constraint
            clipped_changes[driver] = np.sign(change) * constraint
        else:
            clipped_changes[driver] = change
    
    return {
        'feasible': feasible,
        'violations': violations,
        'clipped_changes': clipped_changes
    }


def rank_improvement_opportunities(
    sensitivities: Dict[str, float],
    current_values: Dict[str, float],
    cost_map: Dict[str, float],
    feasible_shifts: Dict[str, Dict[str, float]],
    default_shift: float = 0.1
) -> pd.DataFrame:
    """
    Rank improvement opportunities by ROI, considering constraints.
    
    Parameters
    ----------
    sensitivities : Dict[str, float]
        Dictionary mapping driver names to sensitivity values
    current_values : Dict[str, float]
        Dictionary of current driver values
    cost_map : Dict[str, float]
        Dictionary mapping driver names to per-unit costs
    feasible_shifts : Dict[str, Dict[str, float]]
        Dictionary mapping driver names to feasible shift ranges
        (e.g., {'driver1': {'min': -0.2, 'max': 0.3}})
    default_shift : float, default=0.1
        Default shift amount (10%) for drivers without specific constraints
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'driver': Driver name
        - 'shift': Proposed shift amount
        - 'new_value': New driver value after shift
        - 'delta_value': Change in driver value
        - 'impact': Expected impact on the metric
        - 'cost': Cost of implementing the shift
        - 'roi': Impact / cost ratio
        sorted by ROI (descending)
    """
    rows = []
    
    for driver, sensitivity in sensitivities.items():
        current_value = current_values.get(driver, 0)
        cost_per_unit = cost_map.get(driver, 1.0)
        
        # Determine shift direction based on sensitivity
        shift_direction = 1 if sensitivity > 0 else -1
        
        # Determine shift amount
        if driver in feasible_shifts:
            # Use feasible shifts if provided
            constraints = feasible_shifts[driver]
            if shift_direction > 0:
                # Use maximum positive shift
                shift = constraints.get('max', default_shift)
            else:
                # Use maximum negative shift
                shift = constraints.get('min', -default_shift)
        else:
            # Use default shift
            shift = shift_direction * default_shift
        
        # Calculate new value
        new_value = current_value * (1 + shift)
        delta_value = new_value - current_value
        
        # Calculate impact and cost
        impact = sensitivity * delta_value
        cost = cost_per_unit * abs(delta_value)
        
        # Calculate ROI
        roi = impact / cost if cost > 0 else float('inf')
        
        rows.append({
            'driver': driver,
            'shift': shift,
            'new_value': new_value,
            'delta_value': delta_value,
            'impact': impact,
            'cost': cost,
            'roi': roi
        })
    
    # Sort by ROI (descending)
    df = pd.DataFrame(rows)
    return df.sort_values('roi', ascending=False).reset_index(drop=True)