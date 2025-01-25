import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.performance import calculate_required_growth

class GoalSettingPattern:
    """
    Helps define future targets from top-down or bottom-up feasibility.
    """

    PATTERN_NAME = "GoalSetting"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        current_value: float,
        target_value: float,
        periods_left: int,
        analysis_window: Dict[str, str]
    ) -> PatternOutput:
        """
        We call a function to find the required compound growth rate to reach target_value from current_value 
        over periods_left.

        Returns
        -------
        PatternOutput
            results={
              "current_value": float,
              "target_value": float,
              "periods_left": int,
              "needed_growth_rate": float
            }
        """
        growth_rate = calculate_required_growth(current_value, target_value, periods_left)
        results = {
            "current_value": current_value,
            "target_value": target_value,
            "periods_left": periods_left,
            "needed_growth_rate": growth_rate
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
