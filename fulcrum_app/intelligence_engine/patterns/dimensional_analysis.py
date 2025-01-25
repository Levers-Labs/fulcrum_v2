import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.dimension_analysis import calculate_slice_metrics, compute_slice_shares, rank_metric_slices

class DimensionAnalysisPattern:
    PATTERN_NAME = "DimensionAnalysis"
    PATTERN_VERSION = "1.0"

    def run(self, metric_id: str, data: pd.DataFrame, analysis_window: Dict[str, str], slice_col: str="segment") -> PatternOutput:
        if data.empty:
            return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, {"message":"no_data"})

        # Expect data has columns: [slice_col, "value"].
        agg_df = calculate_slice_metrics(data, slice_col, "value", agg="sum")
        shares_df = compute_slice_shares(agg_df, slice_col, val_col="aggregated_value")
        top_slices = rank_metric_slices(shares_df, val_col="aggregated_value", top_n=3, ascending=False)

        results = {
            "slice_metrics": agg_df.to_dict("records"),
            "slice_shares": shares_df.to_dict("records"),
            "top_slices": top_slices.to_dict("records")
        }
        return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, results)
