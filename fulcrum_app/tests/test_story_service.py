import pytest
from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from fulcrum_app.storytelling_engine.story_service import StoryService

def test_story_service_basic():
    service = StoryService()

    # Two pattern outputs
    p_out_1 = PatternOutput(
        pattern_name="performance_status",
        pattern_version="1.0",
        metric_id="metric_a",
        analysis_window={"start_date":"2025-01-01","end_date":"2025-01-02"},
        results={
            "final_status": "off_track",
            "final_value": 90,
            "final_target": 100,
            "threshold": 0.05
        }
    )

    p_out_2 = PatternOutput(
        pattern_name="performance_status",
        pattern_version="1.0",
        metric_id="metric_b",
        analysis_window={"start_date":"2025-01-01","end_date":"2025-01-02"},
        results={
            "final_status": "on_track",
            "final_value": 120,
            "final_target": 100,
            "threshold": 0.05
        }
    )

    all_stories = service.generate_stories_for_outputs([p_out_1, p_out_2])
    assert len(all_stories) == 2

    off_track_story = [s for s in all_stories if "metric_a" in s.title][0]
    assert off_track_story.title == "metric_a is Off Track"

    on_track_story = [s for s in all_stories if "metric_b" in s.title][0]
    assert on_track_story.title == "metric_b is On Track"
