# =============================================================================
# Forecasting
#
# This file includes primitives for statistical/driver-based forecasting,
# scenario bounding (best/worst case), pacing, target achievement checks,
# and forecast accuracy evaluation.
#
# Dependencies:
#   - pandas as pd
#   - numpy as np
#   - statsmodels.tsa.holtwinters
#   - pmdarima for auto_arima
# =============================================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pmdarima import arima as pmd_arima
import math

def _convert_grain_to_freq(grain: str) -> str:
    """
    Convert a textual grain like 'day','daily','week','weekly','month','monthly','quarter','quarterly'
    into a pandas frequency alias.
    """
    g = grain.lower()
    if g in ["day", "daily"]:
        return "D"
    elif g in ["week", "weekly"]:
        return "W-MON"
    elif g in ["month", "monthly"]:
        return "MS"  # month start
    elif g in ["quarter", "quarterly"]:
        return "QS"  # quarter start
    else:
        raise ValueError(f"Unsupported grain '{grain}'.")

def simple_forecast(
    df: pd.DataFrame,
    value_col: str = "value",
    periods: int = 7,              # UPDATED: replaced 'steps' with 'periods'
    method: str = "ses",
    seasonal_periods: int = None,
    date_col: str = None,
    freq: str = None,
    grain: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Produce a forecast using one of ['naive','ses','holtwinters','auto_arima'],
    with optional resampling by freq or grain.

    UPDATED:
     - (Suggested Update #9) replaced steps->periods
     - (Suggested Update #10) use _convert_grain_to_freq if grain provided

    Returns
    -------
    pd.DataFrame with columns ['date','forecast'].
    """
    dff = df.copy()
    if date_col is None:
        # fallback: naive approach if less than 2 data points
        series = dff[value_col].dropna()
        if len(series) < 2 and method != "naive":
            method = "naive"
        if method == "naive":
            last_val = series.iloc[-1] if len(series) > 0 else 0.0
            idx_future = np.arange(len(series), len(series) + periods)
            return pd.DataFrame({"date": idx_future, "forecast": [last_val]*periods})
    else:
        # convert date
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff.sort_values(date_col, inplace=True)
        dff.set_index(date_col, inplace=True)

        # If user passed 'grain', override freq param
        if grain is not None:
            freq_alias = _convert_grain_to_freq(grain)
            freq = freq_alias
        # Resample if freq is specified
        if freq is not None:
            dff = dff[value_col].resample(freq).sum()
            dff = dff.fillna(0)
            series = dff
        else:
            series = dff[value_col].dropna()

        if len(series) < 2 and method != "naive":
            method = "naive"

    # do forecasting
    if method == "naive":
        last_val = series.iloc[-1]
        if date_col and freq and not series.index.empty:
            last_date = series.index[-1]
            future_idx = pd.date_range(last_date, periods=periods+1, freq=freq)[1:]
            fc_vals = [last_val]*periods
            return pd.DataFrame({"date": future_idx, "forecast": fc_vals})
        else:
            idx_future = np.arange(len(series), len(series)+periods)
            fc_vals = [last_val]*periods
            return pd.DataFrame({"date": idx_future, "forecast": fc_vals})

    elif method == "ses":
        model = SimpleExpSmoothing(series, initialization_method="estimated")
        fit = model.fit(**kwargs)
        fc_vals = fit.forecast(periods)

    elif method == "holtwinters":
        model = ExponentialSmoothing(
            series,
            trend=kwargs.pop("trend", None),
            seasonal=kwargs.pop("seasonal", None),
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        fit = model.fit(**kwargs)
        fc_vals = fit.forecast(periods)

    elif method == "auto_arima":
        do_seasonal = (seasonal_periods is not None and seasonal_periods>1)
        default_arima_kwargs = dict(
            start_p=1, start_q=1, max_p=5, max_q=5,
            seasonal=do_seasonal,
            m=seasonal_periods if do_seasonal else 1,
            stepwise=True, error_action='ignore', suppress_warnings=True
        )
        for k,v in default_arima_kwargs.items():
            kwargs.setdefault(k,v)
        model = pmd_arima.auto_arima(series, **kwargs)
        fc_vals = model.predict(n_periods=periods)

    else:
        raise ValueError(f"Unknown forecast method '{method}'.")

    if date_col:
        # Build date index for future
        if freq is None:
            # fallback: numeric range for future
            idx_future = np.arange(len(series), len(series)+periods)
            return pd.DataFrame({"date": idx_future, "forecast": fc_vals}).reset_index(drop=True)
        else:
            last_idx = series.index[-1]
            future_idx = pd.date_range(last_idx, periods=periods+1, freq=freq)[1:]
            return pd.DataFrame({"date": future_idx, "forecast": fc_vals}).reset_index(drop=True)
    else:
        # numeric fallback
        idx_future = np.arange(len(series), len(series)+periods)
        return pd.DataFrame({"date": idx_future, "forecast": fc_vals}).reset_index(drop=True)

def forecast_upstream_metrics(drivers_dict, periods=7, method="ses", date_col=None, freq=None, grain=None, **kwargs):
    """
    For each driver => run simple_forecast. Updated param name 'periods' to match.
    """
    results = {}
    for driver_id, driver_df in drivers_dict.items():
        fc_df = simple_forecast(
            driver_df,
            value_col="value",
            periods=periods,
            method=method,
            date_col=date_col,
            freq=freq,
            grain=grain,
            **kwargs
        )
        results[driver_id] = fc_df
    return results

def forecast_metric_dimensions(dim_dfs, periods=7, method="ses", freq=None, grain=None, date_col="date", value_col="value", **kwargs):
    results = {}
    for slice_val, sub_df in dim_dfs.items():
        fc = simple_forecast(
            sub_df,
            value_col=value_col,
            periods=periods,
            method=method,
            date_col=date_col,
            freq=freq,
            grain=grain,
            **kwargs
        )
        results[slice_val] = fc
    return results

def forecast_best_and_worst_case(forecast_df, buffer_pct=10.0, forecast_col="forecast"):
    dff = forecast_df.copy()
    dff["best_case"] = dff[forecast_col] * (1.0 + buffer_pct/100.0)
    dff["worst_case"] = dff[forecast_col] * (1.0 - buffer_pct/100.0)
    return dff

def forecast_target_achievement(forecast_df, target_df, forecast_col="forecast", target_col="target", date_col="date"):
    merged = pd.merge(
        forecast_df[[date_col, forecast_col]],
        target_df[[date_col, target_col]],
        on=date_col, how="left"
    )
    merged["abs_diff"] = merged[forecast_col] - merged[target_col]

    def safe_pct(row):
        t = row[target_col]
        if pd.isna(t) or t == 0:
            return None
        return (row["abs_diff"] / abs(t)) * 100

    merged["pct_diff"] = merged.apply(safe_pct, axis=1)
    merged["on_track"] = merged.apply(
        lambda r: (r[target_col] is not None) and (r[forecast_col] >= r[target_col]),
        axis=1
    )
    return merged

def calculate_forecast_accuracy(actual_df, forecast_df, date_col="date", actual_col="actual", forecast_col="forecast"):
    dfa = actual_df[[date_col, actual_col]].copy()
    dff = forecast_df[[date_col, forecast_col]].copy()
    merged = pd.merge(dfa, dff, on=date_col, how="inner").dropna(subset=[actual_col, forecast_col])
    if merged.empty:
        return {"rmse": None, "mae": None, "mape": None, "n": 0}

    errors = merged[forecast_col] - merged[actual_col]
    abs_errors = errors.abs()
    rmse = float(np.sqrt((errors**2).mean()))
    mae = float(abs_errors.mean())

    def safe_ape(row):
        if row[actual_col] == 0:
            return None
        return abs(row[forecast_col] - row[actual_col])/abs(row[actual_col])*100.0

    merged["ape"] = merged.apply(safe_ape, axis=1)
    mape = merged["ape"].mean(skipna=True)
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": float(mape) if pd.notna(mape) else None,
        "n": len(merged)
    }

def assess_forecast_uncertainty(forecast_df, lower_col="lower", upper_col="upper", forecast_col="forecast"):
    dff = forecast_df.dropna(subset=[lower_col, upper_col, forecast_col]).copy()
    if dff.empty:
        return {"mean_interval_width": None, "max_interval_width": None, "mean_relative_width": None}

    dff["interval_width"] = dff[upper_col] - dff[lower_col]
    mean_width = dff["interval_width"].mean()
    max_width = dff["interval_width"].max()

    def safe_rel(row):
        denom = abs(row[forecast_col])
        return row["interval_width"]/denom if denom>1e-9 else None

    dff["relative_width"] = dff.apply(safe_rel, axis=1)
    rel_width = dff["relative_width"].mean(skipna=True)

    return {
        "mean_interval_width": float(mean_width),
        "max_interval_width": float(max_width),
        "mean_relative_width": float(rel_width) if pd.notna(rel_width) else None
    }

def decompose_forecast_drivers(driver_forecasts, driver_coefs, base_intercept=0.0):
    all_dates = set()
    for df_fc in driver_forecasts.values():
        all_dates.update(df_fc["date"].unique())
    all_dates = sorted(all_dates)

    rows = []
    for d in all_dates:
        row_data = {"date": d}
        total = base_intercept
        for drv, df_fc in driver_forecasts.items():
            sub = df_fc.loc[df_fc["date"] == d, "forecast"]
            if len(sub)==1:
                forecast_val = sub.iloc[0]
            else:
                forecast_val = 0
            coeff = driver_coefs.get(drv, 0.0)
            contrib = forecast_val*coeff
            row_data[f"{drv}_contribution"] = contrib
            total += contrib
        row_data["total_forecast"] = total
        rows.append(row_data)

    return pd.DataFrame(rows)
