from typing import Dict, Tuple, Optional
from .data_structures import PatternOutput


class PatternCacheService:
    """
    In-memory caching for pattern outputs, keyed by:
     (metric_id, pattern_name, analysis_window(start_date, end_date))
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, str, str, str], PatternOutput] = {}

    def store_pattern_result(self, pattern_output: PatternOutput):
        key = (
            pattern_output.metric_id,
            pattern_output.pattern_name,
            pattern_output.analysis_window.get("start_date", ""),
            pattern_output.analysis_window.get("end_date", "")
        )
        self._cache[key] = pattern_output

    def get_cached_pattern_result(
        self,
        metric_id: str,
        pattern_name: str,
        analysis_window: dict
    ) -> Optional[PatternOutput]:
        key = (
            metric_id,
            pattern_name,
            analysis_window.get("start_date", ""),
            analysis_window.get("end_date", "")
        )
        return self._cache.get(key, None)
