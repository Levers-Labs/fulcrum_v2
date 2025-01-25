import pandas as pd
from typing import Dict, Any

def calculate_descriptive_stats(df: pd.DataFrame, value_col: str = "value") -> Dict[str, Any]:
    """
    Compute basic summary stats (min, max, mean, median, std, etc.) over the entire DataFrame.
    Returns a dictionary { 'min':..., 'max':..., 'mean':..., 'median':..., 'std':... }
    """
    series = df[value_col].dropna()
    if len(series) == 0:
        return {
            "min": None, "max": None, "mean": None, "median": None, "std": None
        }
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std())
    }
