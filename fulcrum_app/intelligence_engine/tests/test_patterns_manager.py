import pandas as pd
from fulcrum_app.intelligence_engine.caching_service import PatternCacheService
from fulcrum_app.intelligence_engine.patterns_manager import PatternsManager

def test_patterns_manager_caching():
    cache_service = PatternCacheService()
    pm = PatternsManager(cache_service)

    data = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "value": [105, 110],
        "target": [100, 100]
    })
    analysis_window = {"start_date": "2025-01-01", "end_date": "2025-01-02"}

    # Run patterns for metric "m1"
    outputs1 = pm.run_patterns_for_metric("m1", data, analysis_window)
    assert len(outputs1) == 2  # performance_status + root_cause

    # Run again, should retrieve from cache
    outputs2 = pm.run_patterns_for_metric("m1", data, analysis_window)
    # We expect the same PatternOutput objects from the cache
    assert outputs2[0] is outputs1[0]
    assert outputs2[1] is outputs1[1]
