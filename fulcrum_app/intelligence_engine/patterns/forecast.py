import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.forecasting import simple_forecast

class ForecastPattern:
    """
    Generates a forecast (ARIMA, Holt-Winters, or simpler).
    """

    PATTERN_NAME = "Forecast"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str],
        periods: int = 7,
        method: str = "ses",
        date_col: str = "date",
        freq: str = "D"
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        data : pd.DataFrame
            Should have columns [date_col, 'value'].
        analysis_window : dict
        periods : int
            how many future periods to forecast
        method : str, default='ses'
            can be 'naive','ses','holtwinters','auto_arima', etc.
        date_col : str, default='date'
        freq : str, default='D'

        Returns
        -------
        PatternOutput
            results={
               "forecast_periods": int,
               "forecast_data": list of {date, forecast} dicts
            }
        """
        fc_df = simple_forecast(data, value_col="value", periods=periods, method=method, date_col=date_col, freq=freq)
        results = {
            "forecast_periods": periods,
            "forecast_data": fc_df.to_dict("records")
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
