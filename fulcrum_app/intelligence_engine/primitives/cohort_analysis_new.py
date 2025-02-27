# =============================================================================
# CohortAnalysis
#
#   This file merges advanced "perform_cohort_analysis" logic
#   from your sample code, including measure_method, max_periods,
#   multi-level pivot with absolute/cumulative/delta columns, etc.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional

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
    Purpose: Perform a full cohort analysis that returns both absolute cohort values and retention rates.

    Implementation Details:
    1. Group entities into cohorts based on their start date (cohort_date_col).
    2. track a specified measure (or entity counts) over subsequent time periods (based on time_grain).
    3. Output a combined pivot table with a MultiIndex on the columns: ('absolute', 'cumulative', 'delta').

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for entity_id_col, cohort_date_col, activity_date_col.
        If measure_method != 'count', must also contain measure_col.
    entity_id_col : str, default "entity_id"
    cohort_date_col : str, default "cohort_date"
    activity_date_col : str, default "activity_date"
    time_grain : str, default "M"
        Pandas frequency alias ('D','W','M','Q','Y').
    max_periods : int, default 12
        Maximum number of periods after the cohort start date to include.
    measure_col : Optional[str], default None
        Numeric column for aggregation if measure_method != 'count'.
    measure_method : str, default "count"
        Aggregation method: 'count', 'sum', 'mean', 'min', or 'max'.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with multi-level columns: (measure_type, period_index)
        where measure_type in ['absolute', 'cumulative', 'delta'].
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

    df_copy = df.copy()

    # Convert date columns to datetime
    for col in [cohort_date_col, activity_date_col]:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='raise')

    # If measure_method != 'count', ensure measure_col is provided and numeric
    if measure_method in {"sum", "mean", "min", "max"}:
        if measure_col is None:
            raise ValueError(f"measure_col must be provided for measure_method '{measure_method}'.")
        if measure_col not in df_copy.columns:
            raise ValueError(f"Column '{measure_col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df_copy[measure_col]):
            raise ValueError(f"Column '{measure_col}' must be numeric for aggregation.")

    # Create period columns for time_grain
    df_copy["cohort_period"] = df_copy[cohort_date_col].dt.to_period(time_grain)
    df_copy["activity_period"] = df_copy[activity_date_col].dt.to_period(time_grain)

    # Compute period_index as difference in periods
    df_copy["period_index"] = (df_copy["activity_period"] - df_copy["cohort_period"]).astype(int)
    df_copy = df_copy[(df_copy["period_index"] >= 0) & (df_copy["period_index"] <= max_periods)]

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