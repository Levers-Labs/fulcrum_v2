import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.leverage_scenario import rank_drivers_by_leverage

class LeveragePattern:
    """
    Ranks drivers by leverage or scenario potential. 
    We'll do a simple approach: call rank_drivers_by_leverage from the model.
    """

    PATTERN_NAME = "Leverage"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        model,  # a fitted linear model
        current_point: Dict[str, float],
        cost_map: Dict[str, float],
        analysis_window: Dict[str, str]
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        model : a fitted linear model
        current_point : dict, driver-> current_value
        cost_map : dict, driver-> cost_per_unit
        analysis_window : dict

        Returns
        -------
        PatternOutput
            results={
              "top_drivers": <list of driver->roi sorted desc>
            }
        """
        df = rank_drivers_by_leverage(model, current_point, cost_map)
        top_drivers = df.to_dict("records")

        results = {
            "top_drivers": top_drivers
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
