import pandas as pd
from typing import Dict
from ..data_structures import PatternOutput
from ..primitives.dimension_analysis import (
    calculate_slice_metrics, 
    compute_slice_shares, 
    rank_metric_slices
)

class DimensionAnalysisPattern:
    """
    Examines a metrics dimensional breakdown, returning top slices, share distribution, etc.
    """

    PATTERN_NAME = "DimensionAnalysis"
    PATTERN_VERSION = "1.0"

    def run(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str],
        slice_col: str = "dimension",
        agg: str = "sum",
        top_n: int = 5
    ) -> PatternOutput:
        """
        Parameters
        ----------
        metric_id : str
        data : pd.DataFrame
            Must have columns [slice_col, "value"] plus optional others. 
            Typically we sum or average them by slice_col.
        analysis_window : dict
        slice_col : str, default='dimension'
        agg : str, default='sum'
            e.g. 'sum','mean','count' etc. 
        top_n : int, default=5
            Show top 5 slices.

        Returns
        -------
        PatternOutput
            results={
              "slice_metrics": <list of dicts for each slice>,
              "slice_shares": <list of dicts with share_pct>,
              "top_slices": <list of dicts for the topN slices>
            }
        """
        if data.empty:
            return PatternOutput(self.PATTERN_NAME, self.PATTERN_VERSION, metric_id, analysis_window, 
                                 {"message":"no_data"})

        # aggregator
        agg_df = calculate_slice_metrics(data, slice_col=slice_col, value_col="value", agg=agg)
        shares_df = compute_slice_shares(agg_df, slice_col, val_col="aggregated_value")
        top_slices = rank_metric_slices(shares_df, val_col="aggregated_value", top_n=top_n, ascending=False)

        results = {
            "slice_metrics": agg_df.to_dict("records"),
            "slice_shares": shares_df.to_dict("records"),
            "top_slices": top_slices.to_dict("records")
        }
        return PatternOutput(
            self.PATTERN_NAME,
            self.PATTERN_VERSION,
            metric_id,
            analysis_window,
            results
        )
