import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.forecasting import simple_forecast

class ForecastPattern:
    PATTERN_NAME = "Forecast"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, data: pd.DataFrame, analysis_window: Dict[str, str], forecast_periods: int=7) -> PatternOutput:
        fc_df = simple_forecast(data, value_col="value", periods=forecast_periods)
        results = {
            "forecast_periods": forecast_periods,
            "forecast_data": fc_df.to_dict("records")
        }
        return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, results)
