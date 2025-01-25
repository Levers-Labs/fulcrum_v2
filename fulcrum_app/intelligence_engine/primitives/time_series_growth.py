import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import linregress

def calculate_pop_growth(
    df: pd.DataFrame, 
    value_col: str = "value",
    sort_by_date: Optional[str] = None,
    fill_method: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate period-over-period (PoP) growth for each consecutive row.

    For each row i, pop_growth_i = ((value_i - value_{i-1}) / abs(value_{i-1})) * 100,
    expressed as a percentage.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing at least one numeric column (value_col).
        Typically each row represents a consecutive time period, e.g. daily data.
    value_col : str, default='value'
        Name of the numeric column in df whose period-over-period growth we want.
    sort_by_date : str, optional
        If provided, the function will first sort df by this column in ascending order
        before computing PoP growth. Useful if df is not already chronologically sorted.
    fill_method : str or None, optional
        If provided, can be e.g. 'ffill' or 'bfill' for forward/backward filling of missing
        values. If None, missing values remain as is.

    Returns
    -------
    pd.DataFrame
        A new copy of df with one additional column: "pop_growth" in percent.
        If the previous row's value is zero or NaN, pop_growth is set to NaN.

    Notes
    -----
    - If there's only one row or no previous row for the first row, pop_growth is typically NaN.
    - If the previous row's value is 0, the ratio is undefined => result is NaN.
    - If you want a sign-aware approach (i.e. dividing by previous row's value which can be negative),
      you might replace abs(value_{i-1}) with value_{i-1}. That’s domain-specific.

    Example
    -------
    Suppose df has daily revenue in 'value_col'. If row i has $110 and row i-1 has $100,
    pop_growth_i = (110 - 100)/100 * 100 = 10%.
    """
    df = df.copy()
    if sort_by_date:
        df.sort_values(sort_by_date, inplace=True)

    if fill_method:
        df[value_col] = df[value_col].fillna(method=fill_method)

    # Shift the series by 1 row to get the previous value
    prev_vals = df[value_col].shift(1)

    # Compute PoP growth
    # We'll handle zero or missing previous values by yielding NaN
    growth = (df[value_col] - prev_vals) / prev_vals.abs() * 100.0

    # If prev_vals=0 => growth => inf => we can mask that to NaN
    # If you prefer inf => you can keep it
    growth[prev_vals == 0] = np.nan

    df["pop_growth"] = growth
    return df

def calculate_to_date_growth_rates(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    freq: str = "M",  # 'M' for monthly, 'W' for weekly, etc.
    agg_method: str = "sum"
) -> pd.DataFrame:
    """
    Compare partial-to-date vs. prior partial (MTD, WTD, etc.) for a time series.

    For example, if freq='M', on 2025-02-10 we sum/aggregate data from 2025-02-01 through
    2025-02-10 and compare it to the sum/aggregate from 2025-01-01 through 2025-01-10.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [date_col, value_col].
    date_col : str, default='date'
        The datetime column. Ensure it's a datetime type for resampling or grouping.
    value_col : str, default='value'
        Numeric column to aggregate.
    freq : str, default='M'
        Frequency code for grouping partial. Examples: 'M' => monthly, 'W' => weekly,
        'Q' => quarterly, etc. 
        (This approach uses resampling or grouping on date_col if it’s set as datetime.)
    agg_method : str, default='sum'
        How to aggregate within each partial period. e.g. 'sum', 'mean', etc.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        [date_col, aggregated_value, prior_aggregated_value, partial_growth_pct]
        where partial_growth_pct = (aggregated_value - prior_aggregated_value) / abs(prior_aggregated_value) * 100

    Notes
    -----
    - We assume df is sorted by date. If not, sort first. 
    - This is an example approach using resampling. Another approach is manual grouping.
    - The first partial period won't have a prior partial => partial_growth_pct=NaN.

    Example
    -------
    If freq='M', on 2025-02-10 we have sum of 2025-02-01..2025-02-10. 
    We compare to sum of 2025-01-01..2025-01-10 => partial growth.

    Implementation Approach
    -----------------------
    1. Resample df by freq, computing daily cumsums within each freq sub-window
       or directly by partial date alignment. This can get tricky. 
    2. Alternatively, we can group each date by e.g. (year, month, day_of_month).
    """
    df = df.copy()
    # Ensure date_col is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)

    # We'll do a daily cumsum and then pick the last day in partial period
    # This approach is somewhat simplified. Another approach is to store partial sums for each day_of_period.
    
    # Step 1: set index to date_col for resampling
    df.set_index(date_col, inplace=True)

    # Step 2: Possibly aggregate data daily first
    daily_df = df.resample("D").agg({value_col: agg_method})

    # Step 3: Create a cumsum over entire timeline
    daily_df["cumulative"] = daily_df[value_col].cumsum()

    # Step 4: Group by freq to find the partial cumsum at each day.
    # We'll keep each day in the group, so we can compare day i of this period vs. day i of last period.

    # Let's create a "period_label" so we know which freq period a day belongs to:
    if freq.upper() == "W":
        # weekly
        daily_df["period_label"] = daily_df.index.to_period("W")
        daily_df["day_in_period"] = daily_df.index.weekday  # or day of week
    elif freq.upper() == "M":
        daily_df["period_label"] = daily_df.index.to_period("M")
        daily_df["day_in_period"] = daily_df.index.day
    elif freq.upper() == "Q":
        daily_df["period_label"] = daily_df.index.to_period("Q")
        daily_df["day_in_period"] = daily_df.index.dayofyear % 90  # naive approach
    else:
        # fallback or custom approach
        daily_df["period_label"] = daily_df.index.to_period(freq)
        daily_df["day_in_period"] = daily_df.index.day

    # Step 5: For each (period_label, day_in_period), find the cumulative value
    # We can do a groupby then pick the last row or do a transform.

    daily_df["cumulative_in_period"] = daily_df.groupby(["period_label"])[value_col].cumsum()

    # Actually, a simpler approach might be:
    # cumulative_in_period = overall_cumulative - cumulative of first day of period (like a block offset)
    # but let's keep it simpler for demonstration.

    # Now let's pivot day_in_period as rows, period_label as columns, to compare partial day i vs. prior period's day i
    pivoted = daily_df.reset_index().pivot_table(
        index="day_in_period",
        columns="period_label",
        values="cumulative_in_period",
        aggfunc="last"  # or you might want something else
    )
    # pivoted: row => day_in_period, columns => each freq period, value => cumsum within that period

    # Now we can compare each period's partial day_in_period to the previous period's partial
    periods = list(pivoted.columns)
    partial_summaries = []
    for i, period_lbl in enumerate(periods):
        if i == 0:
            continue  # can't compare first period to previous
        prev_period_lbl = periods[i-1]
        # For each day_in_period row:
        # current_value = pivoted[period_lbl]
        # prior_value = pivoted[prev_period_lbl]
        # partial_growth = (current_value - prior_value)/abs(prior_value)*100
        # We'll flatten to a long table

        compare_df = pd.DataFrame({
            "day_in_period": pivoted.index,
            "period_label": period_lbl,
            "aggregated_value": pivoted[period_lbl],
            "prior_aggregated_value": pivoted[prev_period_lbl]
        })
        compare_df["partial_growth_pct"] = (
            (compare_df["aggregated_value"] - compare_df["prior_aggregated_value"]).div(
                compare_df["prior_aggregated_value"].abs()
            ) * 100
        )
        partial_summaries.append(compare_df)

    if not partial_summaries:
        return pd.DataFrame(columns=["day_in_period", "period_label", "aggregated_value", "prior_aggregated_value", "partial_growth_pct"])

    result = pd.concat(partial_summaries, ignore_index=True)
    return result

def calculate_average_growth(
    df: pd.DataFrame, 
    value_col: str = "value",
    method: str = "arithmetic"
) -> float:
    """
    Compute an 'average growth rate' over the entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a numeric column value_col.
    value_col : str, default='value'
        The column representing the time-series values.
    method : str, default='arithmetic'
        The type of average:
          - 'arithmetic': compute row-by-row % growth, then take the arithmetic mean
          - 'geometric': interpret the first vs last value => (last / first)^(1/(N-1)) - 1
            if N>1 and first>0, last>0
          - 'regression': use a linear regression slope approach => optional

    Returns
    -------
    float
        A single number representing the average growth rate (e.g. 0.05 => 5%).
        If not enough data or invalid scenario, returns 0.0 or np.nan based on logic.

    Notes
    -----
    - For 'arithmetic', we do the same logic as calculate_pop_growth for each consecutive row,
      then average them ignoring NaNs.
    - For 'geometric', we only compare the first and last non-NaN values. If either is <=0, or length<2,
      we return 0 or np.nan.
    """
    df = df.copy().dropna(subset=[value_col])
    if len(df) < 2:
        return 0.0

    if method == "arithmetic":
        # Reuse pop_growth from above
        df_with_growth = calculate_pop_growth(df, value_col=value_col)
        # take mean of pop_growth ignoring NaN
        return df_with_growth["pop_growth"].mean() / 100.0  # convert from % to decimal

    elif method == "geometric":
        # Compare first vs last
        first_val = df[value_col].iloc[0]
        last_val = df[value_col].iloc[-1]
        n = len(df)
        if first_val <= 0 or last_val <= 0:
            return 0.0  # or np.nan => domain-specific
        ratio = last_val / first_val
        if n <= 1 or ratio <= 0:
            return 0.0
        # compound rate
        # note: if these data points represent consecutive periods, the # of intervals is n-1
        # or the user might have a different approach
        rate = ratio**(1.0 / (n-1)) - 1.0
        return rate

    elif method == "regression":
        # or we can do a linear regression on log(value) vs. time index => exponentiate slope
        # for a pure growth rate. We'll do a simplified version:
        from scipy.stats import linregress
        df.reset_index(drop=True, inplace=True)  # ensure a 0..N-1 index
        # If any non-positive values, skip or handle
        if (df[value_col] <= 0).any():
            return 0.0
        log_vals = np.log(df[value_col])
        x = np.arange(len(log_vals))
        slope, intercept, r_val, p_val, std_err = linregress(x, log_vals)
        # slope => average growth in log domain => e^slope -1 => growth rate
        return np.exp(slope) - 1.0

    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_rolling_averages(
    df: pd.DataFrame, 
    value_col: str = "value", 
    window: int = 7,
    min_periods: Optional[int] = None,
    center: bool = False,
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a rolling average column over the time-series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the numeric column value_col. Should be sorted by date if time-based.
    value_col : str, default='value'
        Column to average.
    window : int, default=7
        Size of the moving window (in rows). If you have daily data, 7 => 7-day rolling average.
    min_periods : int or None, default=None
        Minimum number of observations in the window required to have a value; if None, defaults to window.
    center : bool, default=False
        If True, the label is centered. E.g. the rolling window is equally distributed around each row.
    new_col_name : str or None, default=None
        If provided, the name of the new rolling average column. If None, defaults to 'rolling_avg_<window>'.

    Returns
    -------
    pd.DataFrame
        A new df with an additional column for the rolling average.

    Example
    -------
    - daily_df = calculate_rolling_averages(daily_df, "sales", window=7)
      => adds daily_df["rolling_avg_7"].

    Notes
    -----
    - If you have a DateTimeIndex, you can do time-based windows like window="7D" => 7 days.
      This is another approach in pandas. 
    - If the DataFrame is not sorted by date, do so before calling this.
    """
    df = df.copy()
    if min_periods is None:
        min_periods = window
    if new_col_name is None:
        new_col_name = f"rolling_avg_{window}"

    df[new_col_name] = df[value_col].rolling(
        window=window,
        min_periods=min_periods,
        center=center
    ).mean()

    return df

def calculate_slope_of_time_series(
    df: pd.DataFrame, 
    value_col: str = "value",
    date_col: Optional[str] = None
) -> float:
    """
    Perform a simple linear regression to find the overall slope of the time-series.

    If date_col is provided, we convert it to a numeric scale (e.g. days since first date).
    Otherwise, we just use row indices (0..N-1).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the numeric column value_col, and optionally date_col.
    value_col : str, default='value'
        Column representing the metric of interest.
    date_col : str or None, default=None
        If not None, we parse date_col as datetime, sort by it, and convert
        to an integer offset for regression. E.g. day 0..N-1 from the earliest date.

    Returns
    -------
    float
        The slope from linregress. Interpreted as 'units of value_col per day' if date_col is daily,
        or 'units per row' if date_col is None.

    Notes
    -----
    - We do a simple linear regression: slope, intercept, r_value, p_value, std_err = linregress(x, y).
    - If there's fewer than 2 data points, we return 0.0 or np.nan.
    """
    df = df.copy().dropna(subset=[value_col])
    if len(df) < 2:
        return 0.0

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        # create an integer offset
        min_date = df[date_col].iloc[0]
        df["_x"] = (df[date_col] - min_date).dt.total_seconds() / (3600*24)  # convert to days
    else:
        # use row index
        df.reset_index(drop=True, inplace=True)
        df["_x"] = df.index

    slope, intercept, r_val, p_val, std_err = linregress(df["_x"], df[value_col])
    return slope

def calculate_cumulative_growth(
    df: pd.DataFrame, 
    value_col: str = "value",
    sort_by_date: Optional[str] = None,
    method: str = "sum",
    base_index: float = 100.0,
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Transform raw values into a cumulative series. By default, a simple cumsum.
    Alternatively, use method='product' for compounding or 'index' for a 100-based index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric column value_col.
    value_col : str, default='value'
        Column to accumulate or index.
    sort_by_date : str or None, default=None
        If provided, sort the DataFrame by that column ascending first.
    method : str, default='sum'
        - 'sum': df[new_col] = cumsum(value_col)
        - 'product': df[new_col] = cumprod(1 + value_col) (like returns)
        - 'index': df[new_col] = base_index * (1 + cumsum(value_col / ???)) or 
          domain-specific approach
    base_index : float, default=100.0
        If method='index', we start the index at base_index on the first row.
    new_col_name : str or None, default=None
        Column name for the resulting cumulative series. If None, set automatically.

    Returns
    -------
    pd.DataFrame
        A new df with the cumulative growth column appended.

    Notes
    -----
    - For 'product', we assume 'value_col' is a daily return (like 0.02 => 2%).
      Then we do rolling product of (1 + return).
    - For 'index', we might do a more advanced approach if we interpret value_col as 
      daily returns. We multiply base_index * the product of (1+ returns).
      Implementation can vary widely based on your domain.
    """
    df = df.copy()
    if sort_by_date:
        df.sort_values(sort_by_date, inplace=True)
    if new_col_name is None:
        new_col_name = f"cumulative_{method}"

    if method == "sum":
        df[new_col_name] = df[value_col].cumsum()
    elif method == "product":
        # interpret value_col as daily return => e.g. 0.02 => 2% 
        # then the new column is cumulative product of (1 + value_col)
        df[new_col_name] = (1 + df[value_col]).cumprod() - 1
        # or if you want the final as e.g. total growth factor
    elif method == "index":
        # interpret value_col as daily return, 
        # index_0 = base_index, index_t = index_{t-1} * (1+value_col_t)
        # do a cumprod but multiply by base_index
        growth_factor = (1 + df[value_col]).cumprod()
        df[new_col_name] = base_index * growth_factor
    else:
        raise ValueError(f"Unknown method: {method}")

    return df