import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.root_cause import (
    decompose_metric_change,
    analyze_dimension_impact,
    influence_attribution
)

class RootCausePattern:
    """
    Tries to explain why a metric changed from T0->T1. 
    We might do dimension-level breakdown plus a driver-based approach if we have a model.
    """

    PATTERN_NAME = "RootCause"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        data_t0: pd.DataFrame,
        data_t1: pd.DataFrame,
        analysis_window: Dict[str, str],
        driver_model=None,
        driver_t0: pd.DataFrame = None,
        driver_t1: pd.DataFrame = None
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        data_t0 : pd.DataFrame
            aggregated data at time T0 (or we might do final_value at T0).
        data_t1 : pd.DataFrame
            aggregated data at time T1.
        analysis_window : dict
        driver_model : optional, a regression model
        driver_t0 : optional, driver values at T0
        driver_t1 : optional, driver values at T1

        Returns
        -------
        PatternOutput
            results={
              "dimension_impact": <list or summary>,
              "driver_attribution": <list or summary>,
              "seasonality_effect": ... (if used),
              "residual": ...
            }
        """
        # dimension impact
        dim_impact = []
        if not data_t0.empty and not data_t1.empty:
            # e.g. we have a slice_col
            col = "dimension" if "dimension" in data_t0.columns else None
            if col:
                dim_df = analyze_dimension_impact(data_t0, data_t1, slice_col=col)
                dim_impact = dim_df.to_dict("records")

        driver_att = []
        if driver_model is not None and driver_t0 is not None and driver_t1 is not None:
            # we only do an example if we have a single row with driver values
            # or we do an aggregated approach
            y_change = 0.0  # or from the main metric
            if not data_t0.empty and not data_t1.empty:
                y0 = data_t0["value"].sum()
                y1 = data_t1["value"].sum()
                y_change = y1 - y0
            # do a single row approach for X_t0, X_t1
            X0 = driver_t0.drop(columns=["date"], errors="ignore").values.reshape(1,-1)
            X1 = driver_t1.drop(columns=["date"], errors="ignore").values.reshape(1,-1)
            driver_names = list(driver_t0.drop(columns=["date"],errors="ignore").columns)
            att_res = influence_attribution(model=driver_model, X_t0=X0[0], X_t1=X1[0], y_change=y_change, driver_names=driver_names)
            driver_att = att_res

        results = {
            "dimension_impact": dim_impact,
            "driver_attribution": driver_att,
            "seasonality_effect": 0.0  # placeholder
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
