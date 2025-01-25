import pandas as pd
from typing import List, Dict
from .data_structures import PatternOutput
from .patterns.performance_status import PerformanceStatusPattern
from .patterns.root_cause import RootCausePattern
from .caching_service import PatternCacheService

class PatternsManager:
    """
    Orchestrates running multiple patterns on a single metric.
    """

    def __init__(self, cache_svc: PatternCacheService):
        self.cache_svc = cache_svc
        self.performance_status_pattern = PerformanceStatusPattern()
        self.root_cause_pattern = RootCausePattern()

    def run_patterns_for_metric(
        self,
        metric_id: str,
        data: pd.DataFrame,
        analysis_window: Dict[str, str]
    ) -> List[PatternOutput]:
        """
        Given a metric_id and a DataFrame of data, run multiple patterns
        (PerformanceStatus, RootCause, etc.) and return their outputs.

        We'll also store the results in the cache so we can retrieve them later.
        """
        outputs = []

        # 1) PerformanceStatus
        # Check if we have a cached result
        cached_output = self.cache_svc.get_cached_pattern_result(
            metric_id,
            "performance_status",
            analysis_window
        )
        if cached_output:
            outputs.append(cached_output)
        else:
            pattern_output = self.performance_status_pattern.run(
                metric_id, data, analysis_window
            )
            self.cache_svc.store_pattern_result(pattern_output)
            outputs.append(pattern_output)

        # 2) RootCause (stub)
        cached_output = self.cache_svc.get_cached_pattern_result(
            metric_id,
            "root_cause",
            analysis_window
        )
        if cached_output:
            outputs.append(cached_output)
        else:
            pattern_output = self.root_cause_pattern.run(
                metric_id, data, analysis_window
            )
            self.cache_svc.store_pattern_result(pattern_output)
            outputs.append(pattern_output)

        return outputs