import pandas as pd
import numpy as np
from typing import Optional
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL

def decompose_metric_change(
    val_t0: float, 
    val_t1: float, 
    factors: dict,
    relationship: str = "additive"
) -> dict:
    """
    Decompose the change in a metric (val_t0 -> val_t1) among multiple factors.

    Parameters
    ----------
    val_t0 : float
        The metric value at time T0.
    val_t1 : float
        The metric value at time T1.
    factors : dict
        A dictionary of factor_name -> (factor_val_t0, factor_val_t1).
        E.g. {"A": (A0, A1), "B":(B0,B1)} if M = A + B, or M = f(A,B).
    relationship : str, default='additive'
        - 'additive': We assume M = sum of factors. 
          The total change is sum of factor changes. 
        - 'multiplicative': We assume M ~ product or ratio of factors, 
          and we do a log-based partial approach. (Simplified example shown below.)
    
    Returns
    -------
    dict
        {
          'total_change': val_t1 - val_t0,
          'factors': {
             factor_name: {
               'delta': factor_val_t1 - factor_val_t0,
               'contribution_abs': ...,
               'contribution_pct': ...
             }, ...
          },
          'residual': ...
        }
        The 'residual' can capture any mismatch for more complex relationships 
        or if sum of factor changes doesn't exactly equal total_change.

    Example
    -------
    If M0=100, M1=130, and factors={"A":(60,80), "B":(40,50)}, 
    then sum_of_changes=(20 + 10)=30 => matches M1-M0=30 => each factor's share is 2/3 and 1/3 respectively.
    """
    total_change = val_t1 - val_t0

    if relationship == "additive":
        # sum factor changes
        factor_changes = {}
        sum_of_changes = 0.0

        for f_name, (f0, f1) in factors.items():
            delta = f1 - f0
            factor_changes[f_name] = {"delta": delta}
            sum_of_changes += delta

        # if sum_of_changes != total_change, we have a residual
        residual = total_change - sum_of_changes

        # compute contribution_abs, contribution_pct
        for f_name, fdict in factor_changes.items():
            delta = fdict["delta"]
            if total_change != 0:
                abs_contribution = delta
                pct_contribution = (delta / total_change) * 100.0
            else:
                abs_contribution = 0.0
                pct_contribution = 0.0
            fdict["contribution_abs"] = abs_contribution
            fdict["contribution_pct"] = pct_contribution

        return {
            "total_change": total_change,
            "factors": factor_changes,
            "residual": residual
        }

    elif relationship == "multiplicative":
        # For a simplified approach, interpret M ~ product of factors => 
        # log(M) ~ log(A) + log(B). Then the partial effect is 
        # factor_i's share ~ (log(f1_i/f0_i))/ log(M1/M0).
        # If any factor=0 => handle carefully.
        if val_t0 <= 0 or val_t1 <= 0:
            # domain meltdown => return additive fallback or error
            return {"error": "multiplicative assumed >0 metric, but got zero or negative."}

        import math
        ratio_M = val_t1 / val_t0
        if ratio_M <= 0:
            return {"error": "cannot do multiplicative with non-positive ratio."}

        log_M = math.log(ratio_M)
        sum_of_factor_logs = 0.0
        factor_changes = {}
        
        for f_name, (f0, f1) in factors.items():
            if f0 <= 0 or f1 <= 0:
                factor_changes[f_name] = {"delta": None, "contribution_abs": None, "contribution_pct": None}
                continue
            ratio_f = f1 / f0
            log_f = math.log(ratio_f)
            sum_of_factor_logs += log_f
            factor_changes[f_name] = {"delta": ratio_f - 1.0, "log_change": log_f}

        # total_change = val_t1 - val_t0, but partial effect is about log ratios
        # We'll define an absolute change as factor_share_of_log * total_change
        # (like partial derivative in log space).
        residual = log_M - sum_of_factor_logs

        for f_name, fdict in factor_changes.items():
            if "log_change" not in fdict or fdict["log_change"] is None:
                # skip
                continue

            log_f = fdict["log_change"]
            if log_M != 0:
                ratio_share = log_f / log_M  # fraction of log change
            else:
                ratio_share = 0.0
            # so partial absolute contribution in original metric space:
            # we approximate: partial_abs = ratio_share * (val_t1 - val_t0)
            partial_abs = ratio_share * (val_t1 - val_t0)
            fdict["contribution_abs"] = partial_abs
            if (val_t1 - val_t0) != 0:
                fdict["contribution_pct"] = (partial_abs / (val_t1 - val_t0)) * 100.0
            else:
                fdict["contribution_pct"] = 0.0

        return {
            "total_change": total_change,
            "factors": factor_changes,
            "residual_log": residual
        }

    else:
        return {"error": f"Unknown relationship: {relationship}"}

def calculate_component_drift(
    df: pd.DataFrame,
    formula: str,
    id_col: str = "component",
    value_col_t0: str = "value_t0",
    value_col_t1: str = "value_t1"
) -> pd.DataFrame:
    """
    Evaluate how each operand in a formula changed from T0->T1.

    Parameters
    ----------
    df : pd.DataFrame
        Each row is a component or operand with columns [id_col, value_col_t0, value_col_t1].
        E.g. [component='B',value_t0=..., value_t1=...] etc.
    formula : str
        A textual representation like "A = B + C" or "A = (X + Y) / Z".
        In practice, you'd parse or store the formula references to see how to combine components.
    id_col : str, default='component'
        The column that identifies each operand. (e.g. 'B','C','X','Y','Z')
    value_col_t0 : str, default='value_t0'
    value_col_t1 : str, default='value_t1'

    Returns
    -------
    pd.DataFrame
        Contains rows for each component, plus columns for 'delta' and possibly 'partial_effect'.
        If the formula is additive, partial_effect = delta. If the formula is multiplicative or ratio-based,
        you might do partial derivative logic.

    Notes
    -----
    - This is heavily domain-specific. 
    - For advanced usage, parse the formula to decide how each component's drift impacts the outcome.
    """
    # For demonstration, let's do a naive approach for additive formulas only.
    # We'll parse or just detect if there's a '+' sign in formula => additive.
    if "+" in formula and "=" in formula:
        # Let's assume left side is the metric, right side has operands separated by '+'
        # Very naive approach
        right = formula.split("=")[1].strip()
        operands = [s.strip(" ()") for s in right.split("+")]
        # Filter df to only these operands
        sub = df[df[id_col].isin(operands)].copy()
        sub["delta"] = sub[value_col_t1] - sub[value_col_t0]
        # partial_effect is just delta in additive
        sub["partial_effect"] = sub["delta"]
        return sub
    else:
        # stub for advanced logic
        return df.assign(delta=None, partial_effect=None)

def analyze_dimension_impact(
    df_t0: pd.DataFrame, 
    df_t1: pd.DataFrame, 
    slice_col: str = "segment",
    value_col: str = "value"
) -> pd.DataFrame:
    """
    Summation of slice-level changes from T0->T1. 
    If metric = sum of slices, then total delta = sum of each slice's delta.

    Parameters
    ----------
    df_t0 : pd.DataFrame
        Must have columns [slice_col, value_col].
    df_t1 : pd.DataFrame
        Same structure. Each slice's T0, T1 aggregated values.
    slice_col : str, default='segment'
    value_col : str, default='value'

    Returns
    -------
    pd.DataFrame
        Columns: [slice_col, value_col+'_t0', value_col+'_t1', 'delta', 'pct_of_total_delta'] etc.

    Example
    -------
    If T0 => slice A=100, B=50, T1 => slice A=120, B=45 => delta=15 + (-5)=10 => total delta=10 => slice A is +15 => 150% of total, slice B is -5 => -50% of total.
    """
    # Similar approach to your composition change function
    import pandas as pd

    t0 = df_t0.copy()
    t1 = df_t1.copy()

    # rename
    t0.rename(columns={value_col: f"{value_col}_t0"}, inplace=True)
    t1.rename(columns={value_col: f"{value_col}_t1"}, inplace=True)

    merged = pd.merge(
        t0[[slice_col, f"{value_col}_t0"]],
        t1[[slice_col, f"{value_col}_t1"]],
        on=slice_col,
        how="outer"
    ).fillna(0)

    merged["delta"] = merged[f"{value_col}_t1"] - merged[f"{value_col}_t0"]
    total_delta = merged["delta"].sum()
    merged["pct_of_total_delta"] = merged["delta"].apply(lambda x: (x / total_delta * 100) if total_delta != 0 else 0)

    merged.sort_values("delta", ascending=False, inplace=True, ignore_index=True)
    return merged

def get_regression_model(
    X, 
    y, 
    model_type: str = "linear", 
    fit_intercept: bool = True, 
    **kwargs
):
    """
    Fit a regression model linking driver metrics (X) to output metric (y).

    Parameters
    ----------
    X : pd.DataFrame or np.array
        Shape (n_samples, n_features). The driver metrics.
    y : pd.Series or np.array
        Shape (n_samples,). The target metric.
    model_type : str, default='linear'
        Could be 'linear', 'lasso', 'ridge', etc. In this snippet, we handle 'linear' only 
        for demonstration. Expand as needed.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    **kwargs : 
        Additional params passed to the chosen model (e.g. 'alpha' for Lasso).

    Returns
    -------
    model
        A fitted scikit-learn estimator with .coef_, .intercept_, .predict(...).

    Example
    -------
    drivers = df[['driverA','driverB']]
    target = df['output_metric']
    model = get_regression_model(drivers, target, model_type='linear')
    """
    if model_type == "linear":
        model = LinearRegression(fit_intercept=fit_intercept, **kwargs)
    else:
        # optional expansions for Lasso, Ridge, etc.
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X, y)
    return model

def influence_attribution(
    model, 
    X_t0: np.array, 
    X_t1: np.array, 
    y_change: float, 
    driver_names: list
) -> dict:
    """
    Attribute driver changes T0->T1 to an output's total delta (y_change).

    We use a linear model y = intercept + sum_j(beta_j * X_j). 
    Then partial_effect_j = beta_j * (X1_j - X0_j).
    The sum of partial_effect_j for all j = model_estimated_delta, 
    which may differ from the actual y_change => 'residual'.

    Parameters
    ----------
    model : a fitted scikit-learn linear model
        Must have .coef_ for each driver.
    X_t0 : np.array, shape=(n_features,)
        The driver values at T0.
    X_t1 : np.array, shape=(n_features,)
        The driver values at T1.
    y_change : float
        The actual observed change in y (y1 - y0).
    driver_names : list
        Names corresponding to each feature in model.coef_.

    Returns
    -------
    dict
        {
         'estimated_delta': float,
         'residual': float,
         'drivers': {
           driver_name: {
             'delta_x': float,
             'partial_effect': float,
             'pct_of_estimated': float,
             'pct_of_actual': float
           }
         }
        }

    Example
    -------
    model.coef_ = [1.5, -0.3], X0=[10,5], X1=[13,2] => 
    partial_effect_driver1 = 1.5*(13-10)=4.5,
    partial_effect_driver2 = -0.3*(2-5)=0.9 => sum=5.4 => if y_change=5 => residual=-0.4
    """
    if X_t0.shape != X_t1.shape:
        return {"error": "X_t0 and X_t1 must have the same shape."}

    betas = model.coef_
    if len(betas) != len(X_t0):
        return {"error": "model.coef_ length mismatch with X_t0 dimension."}

    deltas = X_t1 - X_t0
    partials = betas * deltas
    estimated_delta = partials.sum()

    driver_details = {}
    for i, dname in enumerate(driver_names):
        delta_x = deltas[i]
        p_effect = partials[i]
        # fraction of the model_estimated_delta
        if abs(estimated_delta) > 1e-9:
            pct_est = (p_effect / estimated_delta) * 100.0
        else:
            pct_est = 0.0

        # fraction of the actual y_change
        if abs(y_change) > 1e-9:
            pct_act = (p_effect / y_change) * 100.0
        else:
            pct_act = 0.0

        driver_details[dname] = {
            "delta_x": float(delta_x),
            "partial_effect": float(p_effect),
            "pct_of_estimated": float(pct_est),
            "pct_of_actual": float(pct_act)
        }

    residual = y_change - estimated_delta
    return {
        "estimated_delta": float(estimated_delta),
        "residual": float(residual),
        "drivers": driver_details
    }

def influence_drift(
    model_t0, 
    model_t1, 
    driver_names: list
) -> dict:
    """
    Compare driver coefficients from two models (e.g., T0 window vs T1 window) 
    to see if the influence of each driver changed significantly.

    Parameters
    ----------
    model_t0 : a fitted model for T0 window
    model_t1 : a fitted model for T1 window
    driver_names : list
        Names for each driver in the same order as model_t0.coef_, model_t1.coef_.

    Returns
    -------
    dict
        {
         driver_name: {
           'coef_t0': float,
           'coef_t1': float,
           'delta_coef': float
         }
        }

    Notes
    -----
    - In practice, you might also want to compare confidence intervals or do a statistical test 
      (like a Chow test) to see if the difference is significant.
    """
    betas0 = model_t0.coef_
    betas1 = model_t1.coef_

    if len(betas0) != len(betas1):
        return {"error": "Models have different coefficient dimensions."}

    results = {}
    for i, dname in enumerate(driver_names):
        c0 = betas0[i]
        c1 = betas1[i]
        results[dname] = {
            "coef_t0": float(c0),
            "coef_t1": float(c1),
            "delta_coef": float(c1 - c0)
        }
    return results

def evaluate_seasonality_effect(
    df: pd.DataFrame, 
    date_col: str = "date", 
    value_col: str = "value",
    period: Optional[int] = None
) -> dict:
    """
    Estimate how much T0->T1 difference might be from seasonality.

    We run an STL decomposition (trend, seasonal, residual) if the data is a time series. 
    Then compare the seasonal component at T0 vs T1.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [date_col, value_col]. Sorted ascending by date_col.
    date_col : str, default='date'
    value_col : str, default='value'
    period : int or None
        The seasonal period (e.g., 7 for weekly seasonality if daily data). 
        If None, we might guess or require the user to specify.

    Returns
    -------
    dict
        {
          'seasonal_diff': float,
          'fraction_of_total_diff': float,
          ...
        }

    Notes
    -----
    - This is a simplistic approach. Real usage might require more robust or custom seasonality detection.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)

    if period is None:
        # user must supply or we guess a default
        return {"error": "Must specify period for STL."}

    # We'll assume the time series is contiguous. If not, ensure frequency set or resample.
    y = df[value_col].to_numpy()
    stl = STL(y, period=period, robust=True)
    res = stl.fit()
    # seasonal, trend, resid
    seasonal = res.seasonal
    # let's define T0 => earliest, T1 => latest
    val_t0 = seasonal[0]
    val_t1 = seasonal[-1]
    seasonal_diff = val_t1 - val_t0
    total_diff = y[-1] - y[0]
    fraction = 0.0
    if abs(total_diff) > 1e-9:
        fraction = seasonal_diff / total_diff

    return {
        "seasonal_diff": float(seasonal_diff),
        "total_diff": float(total_diff),
        "fraction_of_total_diff": float(fraction)
    }

def quantify_event_impact(
    df: pd.DataFrame, 
    event_df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    event_date_col: str = "event_date",
    window_before: int = 7,
    window_after: int = 7
) -> pd.DataFrame:
    """
    Approximate each event's impact by comparing avg metric 
    in [event_date - window_before, event_date-1] vs 
    [event_date, event_date+window_after].

    Parameters
    ----------
    df : pd.DataFrame
        The metric time series [date_col, value_col].
    event_df : pd.DataFrame
        The events [event_date_col, 'event_name', etc.].
    date_col : str, default='date'
        Metric date column.
    value_col : str, default='value'
        Metric numeric column.
    event_date_col : str, default='event_date'
        The column in event_df marking the event date.
    window_before : int, default=7
        Number of days (or periods) before the event date to compute "before" average.
    window_after : int, default=7
        Number of days (or periods) after (inclusive) to compute "after" average.

    Returns
    -------
    pd.DataFrame
        For each event, columns: [event_name?, event_date, before_avg, after_avg, impact, 
                                  event_label, etc.].
        'impact' = after_avg - before_avg. 

    Notes
    -----
    - This is very naive. Real analysis might do local regression or difference-in-differences with a control group.
    - The timescale depends on the date frequency. If daily, window_before=7 => last 7 days.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)

    # Make a quick index for searching
    df.set_index(date_col, inplace=True, drop=False)

    results = []
    for _, row in event_df.iterrows():
        e_date = pd.to_datetime(row[event_date_col])
        # define the window
        start_before = e_date - pd.Timedelta(days=window_before)
        end_before = e_date - pd.Timedelta(days=1)
        start_after = e_date
        end_after = e_date + pd.Timedelta(days=window_after)

        before_slice = df.loc[start_before:end_before, value_col]
        after_slice = df.loc[start_after:end_after, value_col]

        if len(before_slice) == 0 or len(after_slice) == 0:
            results.append({
                event_date_col: e_date,
                "before_avg": None,
                "after_avg": None,
                "impact": None
            })
            continue

        bavg = before_slice.mean()
        aavg = after_slice.mean()
        impact = aavg - bavg
        result_row = {
            event_date_col: e_date,
            "before_avg": bavg,
            "after_avg": aavg,
            "impact": impact
        }
        # if there's an event_name or ID, pass it along
        for col in event_df.columns:
            if col not in [event_date_col]:
                result_row[col] = row[col]
        results.append(result_row)

    return pd.DataFrame(results)