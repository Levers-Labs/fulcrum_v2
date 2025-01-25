import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.time_series_growth import calculate_pop_growth, calculate_rolling_averages
from ..primitives.descriptive_stats import calculate_descriptive_stats

class HistoricalPerformancePattern:
    """
    Answers: how has this metric performed over time, period-over-period growth, trend shifts, etc.
    """
    PATTERN_NAME = "HistoricalPerformance"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, data: pd.DataFrame, analysis_window: Dict[str, str]) -> PatternOutput:
        if data.empty:
            return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, {"message":"no_data"})

        # 1) PoP growth
        pop_df = calculate_pop_growth(data, value_col="value")
        # 2) Rolling averages
        roll_df = calculate_rolling_averages(pop_df, value_col="value", window=7)
        # 3) Basic stats
        stats = calculate_descriptive_stats(data, value_col="value")

        results = {
            "pop_growth": roll_df["pop_growth"].tolist(),
            "rolling_avg": roll_df["rolling_avg"].tolist(),
            "descriptive_stats": stats
        }
        return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, results)
