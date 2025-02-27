# =============================================================================
# CohortAnalysis
#
# This file includes advanced cohort analysis functions for tracking user or entity
# retention over time, with support for different time grains and measurement methods.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union

def perform_cohort_analysis(
    df: pd.DataFrame,
    entity_id_col: str = "entity_id",
    cohort_date_col: str = "cohort_date",
    activity_date_col: str = "activity_date",
    time_grain: str = "M",
    max_periods: int = 12,
    measure_col: Optional[str] = None,
    measure_method: str = "count"
) -> pd.DataFrame:
    """
    Perform a full cohort analysis that returns both absolute cohort values and retention rates.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for entity_id_col, cohort_date_col, activity_date_col.
        If measure_method != 'count', must also contain measure_col.
    entity_id_col : str, default="entity_id"
        Column containing entity identifiers (e.g., user_id)
    cohort_date_col : str, default="cohort_date"
        Column containing the date when an entity joined a cohort (e.g., signup_date)
    activity_date_col : str, default="activity_date"
        Column containing dates of activities
    time_grain : str, default="M"
        Pandas frequency alias ('D','W','M','Q','Y')
    max_periods : int, default=12
        Maximum number of periods after the cohort start date to include
    measure_col : Optional[str], default=None
        Column name for the metric to aggregate if measure_method != 'count'
    measure_method : str, default="count"
        Aggregation method: 'count', 'sum', 'mean', 'min', or 'max'

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with multi-level columns: (measure_type, period_index)
        where measure_type in ['absolute', 'cumulative', 'delta']
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'entity_id': [1, 1, 1, 2, 2, 3],
    ...     'cohort_date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-02-01'],
    ...     'activity_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-01-01', '2023-02-01', '2023-02-01'],
    ...     'revenue': [10, 20, 15, 8, 12, 25]
    ... })
    >>> result = perform_cohort_analysis(df, entity_id_col='entity_id', cohort_date_col='cohort_date', 
    ...                                  activity_date_col='activity_date', measure_col='revenue', 
    ...                                  measure_method='sum')
    >>> # Result will contain absolute sums, cumulative values, and period-over-period retention
    """
    # Validate required columns
    required_cols = {entity_id_col, cohort_date_col, activity_date_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_grains = {'D', 'W', 'M', 'Q', 'Y'}
    time_grain = time_grain.upper()
    if time_grain not in valid_grains:
        raise ValueError(f"Unsupported time_grain '{time_grain}'. Must be one of {valid_grains}.")

    # Validate measure_method and measure_col
    valid_methods = {"count", "sum", "mean", "min", "max"}
    if measure_method not in valid_methods:
        raise ValueError(f"Unsupported measure_method '{measure_method}'. Must be one of {valid_methods}.")
        
    if measure_method != "count" and measure_col is None:
        raise ValueError(f"measure_col must be provided for measure_method '{measure_method}'.")

    if measure_method != "count" and measure_col not in df.columns:
        raise ValueError(f"Column '{measure_col}' not found in DataFrame.")

    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Convert date columns to datetime
    for col in [cohort_date_col, activity_date_col]:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    
    # Drop rows with null dates
    df_copy = df_copy.dropna(subset=[cohort_date_col, activity_date_col])
    
    if df_copy.empty:
        return pd.DataFrame()  # Return empty DataFrame if no valid data

    # Create period columns for time_grain
    df_copy["cohort_period"] = df_copy[cohort_date_col].dt.to_period(time_grain)
    df_copy["activity_period"] = df_copy[activity_date_col].dt.to_period(time_grain)

    # Compute period_index as difference in periods
    df_copy["period_index"] = (df_copy["activity_period"] - df_copy["cohort_period"]).astype(int)
    df_copy = df_copy[(df_copy["period_index"] >= 0) & (df_copy["period_index"] <= max_periods)]

    if df_copy.empty:
        return pd.DataFrame()  # Return empty DataFrame if no data after period filtering

    group_cols = ["cohort_period", "period_index"]

    if measure_method == "count":
        # Drop duplicate entity occurrences
        agg_df = (
            df_copy.drop_duplicates(subset=[entity_id_col, "cohort_period", "period_index"])
            .groupby(group_cols)[entity_id_col]
            .count()
            .reset_index(name="measure")
        )
    else:
        # Ensure measure_col is numeric
        if not pd.api.types.is_numeric_dtype(df_copy[measure_col]):
            try:
                df_copy[measure_col] = pd.to_numeric(df_copy[measure_col], errors='coerce')
                df_copy = df_copy.dropna(subset=[measure_col])
            except:
                raise ValueError(f"Column '{measure_col}' must be numeric for aggregation.")
        
        agg_df = (
            df_copy.groupby(group_cols)[measure_col]
            .agg(measure_method)
            .reset_index(name="measure")
        )

    # Pivot
    pivot_table = agg_df.pivot_table(
        index="cohort_period",
        columns="period_index",
        values="measure",
        fill_value=0
    )
    pivot_table.sort_index(inplace=True)

    # Ensure base period 0 exists for each cohort
    if 0 not in pivot_table.columns:
        raise ValueError("Base period (0) is missing from the pivot table. Ensure each cohort has activity in period 0.")

    # Calculate cumulative retention (cumulative pivot_table / base). Replace 0 in base to avoid /0
    base = pivot_table[0].replace(0, np.nan)
    cumulative = pivot_table.div(base, axis=0)

    # Delta retention: ratio of current cumulative to previous cumulative
    delta = cumulative.div(cumulative.shift(axis=1))
    delta[0] = np.nan  # base period has no delta from previous

    # Combine absolute, cumulative, delta
    combined = pd.concat(
        {"absolute": pivot_table, "cumulative": cumulative, "delta": delta},
        axis=1
    )
    combined = combined.sort_index(axis=1, level=1)
    return combined