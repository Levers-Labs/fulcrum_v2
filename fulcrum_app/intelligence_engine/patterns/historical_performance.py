import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.time_series_growth import calculate_pop_growth, calculate_rolling_averages
from ..primitives.descriptive_stats import calculate_descriptive_stats

class HistoricalPerformancePattern:
    """
    Analyzes the metric's historical record for growth rates, rolling averages, 
    and basic descriptive stats.
    """

    PATTERN_NAME = "HistoricalPerformance"
    PATTERN_VERSION = "1.0"

    def run(
        self, 
        metric_id: str, 
        data: pd.DataFrame, 
        analysis_window: Dict[str, str],
        pop_window: int = 1,
        rolling_window: int = 7
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        data : pd.DataFrame
            Must have ['date','value']. We'll sort by date ascending.
        analysis_window : dict
        pop_window : int, default=1
            If we want day-over-day or month-over-month, typically pop_window=1. 
            But we might do a weekly approach if the data is weekly.
        rolling_window : int, default=7
            For rolling average smoothing.

        Returns
        -------
        PatternOutput
            results={
               "pop_growth_series": list of pop_growth values (same length as data),
               "rolling_avg": list of rolling means,
               "descriptive_stats": {min, max, mean, etc.}
            }
        """
        if data.empty:
            return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, 
                                 results={"message":"no_data"})

        data = data.sort_values("date")
        # compute pop growth
        pop_df = calculate_pop_growth(data, value_col="value")  # pop_window=1 is baked in; you can adapt if needed
        # compute rolling average
        roll_df = calculate_rolling_averages(pop_df, value_col="value", window=rolling_window)

        # stats
        stats = calculate_descriptive_stats(data, value_col="value")

        # assemble results
        # we might store the entire pop_growth as a list, or just the final. We'll store the entire.
        results = {
            "pop_growth_series": roll_df["pop_growth"].tolist(),
            "rolling_avg": roll_df["rolling_avg"].tolist(),
            "descriptive_stats": stats
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
