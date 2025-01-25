import pandas as pd
from fulcrum_app.intelligence_engine.patterns.performance_status import PerformanceStatusPattern

def test_performance_status_pattern():
    pattern = PerformanceStatusPattern()

    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "value": [95, 105],
        "target": [100, 100]
    })

    analysis_window = {"start_date": "2025-01-01", "end_date": "2025-01-02"}
    output = pattern.run("my_metric", df, analysis_window)

    # final row is 105 vs target 100 => on_track
    assert output.results["final_status"] == "on_track"
    assert output.results["final_value"] == 105
    assert output.results["final_target"] == 100
    assert output.pattern_name == "performance_status"
    assert output.pattern_version == "1.0"
    assert output.metric_id == "my_metric"
