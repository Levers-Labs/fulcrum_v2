# fulcrum_app/intelligence_engine/primitives/cohort_analysis.py

import pandas as pd
import numpy as np
from typing import Optional

def perform_cohort_analysis(
    df: pd.DataFrame,
    entity_id_col: str = "entity_id",
    signup_date_col: str = "signup_date",
    activity_date_col: str = "activity_date",
    time_grain: str = "M",
    max_periods: int = 12,
    measure_col: Optional[str] = None,
    measure_method: str = "count"
) -> pd.DataFrame:
    """
    Break data into cohorts by signup date, then track retention or usage over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [entity_id_col, signup_date_col, activity_date_col]. 
        Optionally measure_col if measure_method != 'count'.
    entity_id_col : str
        The column identifying the entity (e.g. user_id).
    signup_date_col : str
        The column indicating when the entity joined or started (cohort definition).
    activity_date_col : str
        The column indicating a date on which the entity was active or had usage.
    time_grain : str, default='M'
        Pandas frequency for grouping (e.g. 'M' => monthly cohorts, 'W' => weekly).
    max_periods : int, default=12
        The maximum number of periods (months/weeks) after signup to track.
    measure_col : str or None, default=None
        If None and measure_method='count', we do a pure user count. 
        Otherwise, if e.g. measure_col='revenue' and measure_method='sum', we sum revenue.
    measure_method : str, default='count'
        'count' => count distinct entity IDs,
        'sum' => sum measure_col,
        'mean' => average measure_col, etc. 
        We can handle other aggregator logic as well.

    Returns
    -------
    pd.DataFrame
        A pivot table where rows = cohort start period, columns = period index (0..max_periods),
        values = the measure (count, sum, etc.). 
        Typically you might then compute retention = pivot / pivot.iloc[:,0] or something.

    Notes
    -----
    - We define a 'cohort_label' for each entity based on signup_date_col truncated to time_grain 
      (e.g. 2025-01 if monthly).
    - Then for each activity_date, we compute the 'period_index' = difference in time periods from the cohort_label. 
      e.g. if user signed up in 2025-01, and time_grain='M', then an activity in 2025-02 => period_index=1.
    - We limit up to max_periods to keep the table manageable.
    - The pivot table columns are the period_index, the rows are the cohort_label.
    """
    data = df.copy()
    # Ensure date columns are datetime
    data[signup_date_col] = pd.to_datetime(data[signup_date_col])
    data[activity_date_col] = pd.to_datetime(data[activity_date_col])

    # Step 1: define the cohort_label by truncating signup_date to the desired time_grain
    data["cohort_label"] = data[signup_date_col].dt.to_period(time_grain).dt.to_timestamp()

    # Step 2: define the activity_label similarly
    data["activity_label"] = data[activity_date_col].dt.to_period(time_grain).dt.to_timestamp()

    # Step 3: compute period_index = difference in # of time_grain units between activity_label and cohort_label
    # e.g. if user signed up 2025-01, activity 2025-03 => period_index=2 (month difference)
    data["period_index"] = ((data["activity_label"].dt.year - data["cohort_label"].dt.year)*12 
                            + (data["activity_label"].dt.month - data["cohort_label"].dt.month))
    # for weekly, you'd do a different difference approach, or convert to ordinal day / 7, etc. 
    # We'll do a universal approach for months. If time_grain='W', might need logic to compute # of weeks difference.

    # Step 4: limit period_index up to max_periods
    data = data[data["period_index"] >= 0]
    data = data[data["period_index"] <= max_periods]

    # Step 5: aggregator
    # if measure_method='count', we typically want distinct entity_id_col
    # so group by [cohort_label, period_index], then aggregator
    group_cols = ["cohort_label", "period_index"]
    if measure_method == "count":
        pivot_df = (data.drop_duplicates(subset=[entity_id_col, "cohort_label", "period_index"])
                        .groupby(group_cols)[entity_id_col]
                        .count()
                        .reset_index(name="measure"))
    elif measure_method in ["sum","mean","min","max"]:
        if not measure_col:
            raise ValueError(f"measure_col is required for measure_method={measure_method}")
        pivot_df = data.groupby(group_cols)[measure_col].agg(measure_method).reset_index(name="measure")
    else:
        raise ValueError(f"Unsupported measure_method: {measure_method}")

    # Step 6: pivot => rows=cohort_label, columns=period_index
    pivot_table = pivot_df.pivot_table(
        index="cohort_label",
        columns="period_index",
        values="measure",
        fill_value=0
    )
    pivot_table.sort_index(inplace=True)
    pivot_table.columns.name = None  # remove the columns name "period_index"
    return pivot_table

def compute_retention_rates(
    pivot_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert an absolute pivot table (e.g. user counts) into a retention rate table 
    by dividing each row by the row's column 0.

    Parameters
    ----------
    pivot_table : pd.DataFrame
        Typically the output of perform_cohort_analysis, 
        where columns are period_index and values are user counts or similar.

    Returns
    -------
    pd.DataFrame
        The same shape pivot_table, but each row is normalized by column=0, 
        expressed in fraction or percent.

    Notes
    -----
    - If pivot_table has a column 0 of zeros, we skip or produce NaN in that row.
    - You can multiply by 100 if you want percentages.
    """
    df = pivot_table.copy()
    if 0 not in df.columns:
        # means there's no period_index=0 => can't do standard retention
        return df

    base = df[0]  # the first column
    # If base=0, the ratio is undefined => produce NaN or 0
    retention_df = df.div(base, axis=0)
    return retention_df