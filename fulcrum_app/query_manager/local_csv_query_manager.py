import os
import pandas as pd
from typing import Optional
from .base_query_manager import BaseQueryManager

class LocalCSVQueryManager(BaseQueryManager):
    """
    Reads local CSV files, one per metric_id. 
    Each CSV must have at least columns: [date, value, dimension_col(s)? ...].
    If you want dimension breakdown, you either store them in separate CSV or add a col for dimension_name.
    """

    def __init__(self, data_folder: str, dimension_col_mapping: dict = None):
        """
        :param data_folder: directory that holds CSV files named {metric_id}.csv
        :param dimension_col_mapping: maps dimension_name-> column name in CSV.
          e.g. {"region": "region_col"} 
        """
        self.data_folder = data_folder
        self.dimension_col_mapping = dimension_col_mapping or {}

    def fetch_time_series(
        self, metric_id: str, start_date: Optional[str]=None, end_date: Optional[str]=None
    ) -> pd.DataFrame:
        df = self._read_csv(metric_id)
        # Filter by date
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df[["date", "value"]].copy()

    def fetch_dimension_breakdown(
        self, metric_id: str, dimension_name: str, start_date: Optional[str]=None, end_date: Optional[str]=None
    ) -> pd.DataFrame:
        df = self._read_csv(metric_id)
        dim_col = self.dimension_col_mapping.get(dimension_name, dimension_name)
        if dim_col not in df.columns:
            raise ValueError(f"Dimension column '{dim_col}' not found in CSV for metric {metric_id}.")

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        return df[[dim_col, "date", "value"]].rename(columns={dim_col: "dimension_value"})

    def _read_csv(self, metric_id: str) -> pd.DataFrame:
        csv_path = os.path.join(self.data_folder, f"{metric_id}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No CSV found for metric_id='{metric_id}' at {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        return df