import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

class BaseQueryManager(ABC):
    """
    An abstract class that describes the common interface for a Query Manager.
    """

    @abstractmethod
    def fetch_time_series(
        self, metric_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns at least [date, value].
        Optionally filter by date range.
        """
        pass

    @abstractmethod
    def fetch_dimension_breakdown(
        self, metric_id: str, dimension_name: str, start_date: Optional[str]=None, end_date: Optional[str]=None
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns [dimension_value, date, value].
        E.g. if dimension_name='region', dimension_value might be 'APAC','EMEA', etc.
        """
        pass
