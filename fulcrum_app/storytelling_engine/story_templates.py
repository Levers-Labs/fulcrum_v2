from typing import Dict, Any
from .data_structures import Story

def performance_status_on_track(result: Dict[str, Any]) -> Story:
    """
    Example template for a metric that is on track vs. its target.
    We'll pull details from 'result' which is the .results from PatternOutput.
    """
    title = f"{result['metric_id']} is On Track"
    body = (
        f"Your metric '{result['metric_id']}' is currently above or near its target. "
        f"Value: {result['final_value']}, Target: {result['final_target']}."
    )
    # In real usage, you might reference 'analysis_window' or incorporate more detail.
    return Story(
        story_id=None,
        title=title,
        body=body,
        date="2025-01-02",  # Hard-coded for demo; typically you'd pass in from pattern output
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={
            "final_value": result["final_value"],
            "final_target": result["final_target"],
            "threshold": result["threshold"]
        }
    )

def performance_status_off_track(result: Dict[str, Any]) -> Story:
    title = f"{result['metric_id']} is Off Track"
    body = (
        f"Your metric '{result['metric_id']}' is currently below target. "
        f"Value: {result['final_value']}, Target: {result['final_target']}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date="2025-01-02",
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={
            "final_value": result["final_value"],
            "final_target": result["final_target"]
        }
    )


# Optional template for no_data
def performance_status_no_data(result: Dict[str, Any]) -> Story:
    return Story(
        story_id=None,
        title=f"No Data for {result['metric_id']}",
        body="No data is available in the timeframe specified.",
        date="2025-01-02",
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={}
    )

# Registry approach
STORY_TEMPLATE_REGISTRY = {
    # Key: ("pattern_name", "scenario"), Value: a callable
    ("performance_status", "on_track"): performance_status_on_track,
    ("performance_status", "off_track"): performance_status_off_track,
    ("performance_status", "no_data"): performance_status_no_data
}
