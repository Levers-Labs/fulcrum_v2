# =============================================================================
# DataQuality
#
# This file includes primitives for checking missing data, detecting data spikes,
# scoring overall data quality, etc.
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

def _pct_missing(series):
    return series.isna().mean() * 100

# -----------------------------------------------------------------------------
# Main Analysis Functions
# -----------------------------------------------------------------------------

def check_missing_data(df):
    """
    Purpose: Calculate % of missing or null observations in the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
        Each column's percent missing, indexed by column name.
    """
    return df.isna().mean() * 100


def detect_data_spikes(df, date_col='date', value_col='value', threshold=200.0):
    """
    Purpose: Identify suspicious spikes that may be data-ingestion errors.

    Implementation Details:
    1. Compare day-to-day or period-to-period jumps.
    2. If jump > threshold% from prior, label suspicious.
    3. Return DataFrame with 'suspicious_spike' bool.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    value_col : str
    threshold : float
        e.g. 200 means 200% jump

    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values(by=date_col).copy()
    df['prev_value'] = df[value_col].shift(1)
    df['pct_change'] = np.where(df['prev_value'] != 0,
                                (df[value_col] - df['prev_value']) / df['prev_value'] * 100,
                                np.nan)
    df['suspicious_spike'] = df['pct_change'].abs() > threshold
    return df


def score_data_quality(df, value_col='value'):
    """
    Purpose: Compute an overall data quality score based on coverage, outliers, noise, etc.

    Implementation Details:
    1. Evaluate % missing, some measure of outliers or spikes, etc.
    2. Produce a weighted combination => final score in [0..1].
    3. Return {'quality_score': float, 'missing_percent': float, 'spike_percent': float}

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str

    Returns
    -------
    dict
    """
    # Example placeholders
    missing_pct = df[value_col].isna().mean() * 100
    # Let's define outliers as points beyond 3 std
    mean_ = df[value_col].mean()
    std_ = df[value_col].std()
    if std_ == 0 or np.isnan(std_):
        outlier_count = 0
    else:
        outlier_count = df[(df[value_col] > mean_ + 3*std_) | (df[value_col] < mean_ - 3*std_)].shape[0]
    spike_pct = outlier_count / len(df) * 100 if len(df) > 0 else 0

    # A naive scoring approach: perfect = 1.0 if missing_pct=0 and spike_pct=0
    # We'll degrade the score linearly with missing/spike
    score = 1.0 - (missing_pct/100 * 0.5) - (spike_pct/100 * 0.5)
    score = max(0.0, min(1.0, score))

    return {
        'quality_score': score,
        'missing_percent': missing_pct,
        'spike_percent': spike_pct
    }
