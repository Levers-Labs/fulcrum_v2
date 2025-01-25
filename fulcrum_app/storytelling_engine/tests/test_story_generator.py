import pytest
from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from fulcrum_app.storytelling_engine.story_generator import StoryGenerator

def test_generate_stories_on_track():
    generator = StoryGenerator()

    pattern_out = PatternOutput(
        pattern_name="performance_status",
        pattern_version="1.0",
        metric_id="my_metric",
        analysis_window={"start_date":"2025-01-01","end_date":"2025-01-02"},
        results={
            "final_status": "on_track",
            "final_value": 105,
            "final_target": 100,
            "threshold": 0.05
        }
    )

    stories = generator.generate_stories_from_pattern(pattern_out)
    assert len(stories) == 1
    s = stories[0]
    assert "my_metric" in s.title
    assert s.genre == "Performance"
    assert s.theme == "Goal vs Actual"
    assert s.payload["final_value"] == 105
