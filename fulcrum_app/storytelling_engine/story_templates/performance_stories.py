from typing import Dict, Any
from ..data_structures import Story

def performance_status_on_track(pattern_results: Dict[str, Any]) -> Story:
    metric_id = pattern_results["metric_id"]
    val = pattern_results["final_value"]
    tgt = pattern_results["final_target"]
    title = f"{metric_id} is On Track"
    body = f"{metric_id} is at {val}, above the target {tgt}."
    return Story(
        story_id=None,
        title=title,
        body=body,
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={"final_value": val, "final_target": tgt}
    )

def performance_status_off_track(pattern_results: Dict[str, Any]) -> Story:
    metric_id = pattern_results["metric_id"]
    val = pattern_results["final_value"]
    tgt = pattern_results["final_target"]
    title = f"{metric_id} is Off Track"
    body = f"{metric_id} is at {val}, below the target {tgt}."
    return Story(
        story_id=None,
        title=title,
        body=body,
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={"final_value": val, "final_target": tgt}
    )

def performance_status_no_data(pattern_results: Dict[str, Any]) -> Story:
    metric_id = pattern_results["metric_id"]
    title = f"No Data for {metric_id}"
    body = f"There's no data available for {metric_id} in this time window."
    return Story(
        story_id=None,
        title=title,
        body=body,
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={}
    )
