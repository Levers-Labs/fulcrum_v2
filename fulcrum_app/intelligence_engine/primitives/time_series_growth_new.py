# =============================================================================
# TimeSeriesGrowth
#
# This file includes primitives for time-series growth calculations such as
# period-over-period growth, partial-to-date comparisons, rolling averages,
# slopes, and cumulative growth.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - scipy (for linear regression, if needed)
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import linregress

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _safe_shift(series, periods=1):
    return series.shift(periods)

# -----------------------------------------------------------------------------
# Main Analysis Functions
# -----------------------------------------------------------------------------

def calculate_pop_growth(df, date_col='date', value_col='value'):
    """
    Purpose: Compute period-over-period growth (day-over-day, week-over-week, etc.).

    Implementation Details:
    1. Accept a DataFrame with [date, value].
    2. Sort by date; shift the series by 1 to get the 'previous' value.
    3. growth_rate = (current - prev)/prev * 100.
    4. Returns a new DataFrame with an additional 'pop_growth' column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [date_col, value_col].
    date_col : str
    value_col : str

    Returns
    -------
    pd.DataFrame
        Original columns plus a 'pop_growth' column (%).
    """
    df = df.sort_values(by=date_col).copy()
    df['prev_value'] = _safe_shift(df[value_col], 1)
    df['pop_growth'] = ((df[value_col] - df['prev_value']) / df['prev_value']) * 100
    df['pop_growth'] = df['pop_growth'].replace([np.inf, -np.inf], np.nan)
    return df


def calculate_to_date_growth_rates(current_df, prior_df):
    """
    Purpose: Compare partial-to-date vs. prior partial (MTD, WTD, QTD, etc.)

    Implementation Details:
    1. For the current period, sum or average the metric up to a specific date.
    2. For the prior period, do the same partial window.
    3. growth = (current_partial - old_partial)/old_partial * 100.

    Parameters
    ----------
    current_df : pd.DataFrame
        The current period data with columns [date, value].
    prior_df : pd.DataFrame
        The prior period data with columns [date, value].

    Returns
    -------
    float
        The partial-to-date growth rate in percent.
    """
    curr_val = current_df['value'].sum()  # or .mean() depending on desired aggregator
    prev_val = prior_df['value'].sum()
    if prev_val == 0:
        return np.nan
    growth = ((curr_val - prev_val) / prev_val) * 100.0
    return growth


def calculate_average_growth(df, date_col='date', value_col='value'):
    """
    Purpose: Compute the average % growth over multiple consecutive periods.

    Implementation Details:
    1. Sort by date ascending.
    2. For each consecutive row, (v[i] - v[i-1]) / v[i-1].
    3. Take mean of those % changes.
    4. Return single average growth rate in percent.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    value_col : str

    Returns
    -------
    float
        Average growth rate in percent over the entire series.
    """
    df = df.sort_values(by=date_col).copy()
    df['prev_value'] = df[value_col].shift(1)
    df['pct_change'] = ((df[value_col] - df['prev_value']) / df['prev_value']) * 100
    df['pct_change'] = df['pct_change'].replace([np.inf, -np.inf], np.nan)
    mean_growth = df['pct_change'].mean()
    return mean_growth


def calculate_rolling_averages(df, value_col='value', windows=[7, 28]):
    """
    Purpose: Create rolling means for smoothing out fluctuations.

    Implementation Details:
    1. Sort ascending by the DataFrame's index or by date if date is the index.
    2. For each window in `windows`, compute df['rolling_avg_{w}'] = rolling mean.
    3. Return updated DataFrame with new columns.

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str
        Column with numeric data.
    windows : list of int
        Rolling windows, e.g. [7, 28].

    Returns
    -------
    pd.DataFrame
        Same input with rolling average columns appended.
    """
    df = df.copy()
    df = df.sort_index()  # or sort_values by date if needed
    for w in windows:
        col_name = f'rolling_avg_{w}'
        df[col_name] = df[value_col].rolling(window=w).mean()
    return df


def calculate_slope_of_time_series(df, date_col='date', value_col='value'):
    """
    Purpose: Fit a linear regression to find overall slope of the series.

    Implementation Details:
    1. Convert date to a numeric offset (e.g., day 1, day 2...).
    2. Use scipy.stats.linregress(x, y).
    3. Return slope, intercept, r_value, p_value, std_err.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    value_col : str

    Returns
    -------
    dict
        {'slope': float, 'intercept': float, 'r_value': float, 'p_value': float, 'std_err': float}
    """
    df = df.sort_values(by=date_col).copy()
    df['time_index'] = (df[date_col] - df[date_col].min()).dt.days
    regression_result = linregress(df['time_index'], df[value_col])
    return {
        'slope': regression_result.slope,
        'intercept': regression_result.intercept,
        'r_value': regression_result.rvalue,
        'p_value': regression_result.pvalue,
        'std_err': regression_result.stderr
    }


def calculate_cumulative_growth(df, date_col='date', value_col='value', base_index=100.0):
    """
    Purpose: Transform the series into a cumulative index from a baseline.

    Implementation Details:
    1. Sort by date ascending.
    2. Start index = base_index (100 by default).
    3. For each row, index[i] = index[i-1] * (1 + daily_growth).
       daily_growth = (value[i] - value[i-1]) / value[i-1].
    4. Return DataFrame with 'cumulative_index' column.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    value_col : str
    base_index : float

    Returns
    -------
    pd.DataFrame
        Original plus 'cumulative_index' column.
    """
    df = df.sort_values(by=date_col).copy()
    df['cumulative_index'] = np.nan
    df.reset_index(drop=True, inplace=True)

    df.at[0, 'cumulative_index'] = base_index
    for i in range(1, len(df)):
        prev_val = df.at[i-1, value_col]
        curr_val = df.at[i, value_col]
        if prev_val and prev_val != 0:
            daily_growth = (curr_val - prev_val) / prev_val
        else:
            daily_growth = 0
        df.at[i, 'cumulative_index'] = df.at[i-1, 'cumulative_index'] * (1 + daily_growth)

    return df
