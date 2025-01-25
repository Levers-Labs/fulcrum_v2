# fulcrum_app/storytelling_engine/story_templates/master_story_templates.py

from typing import Dict, Any
from ..data_structures import Story

#############################################
# PERFORMANCE (Goal vs Actual)
#############################################

def story_on_track(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    val = data.get("final_value")
    tgt = data.get("final_target")
    return Story(
        story_id=None,
        title=f"{metric} is On Track",
        body=f"{metric} is at {val}, beating its target {tgt}.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={"value": val, "target": tgt}
    )

def story_off_track(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    val = data.get("final_value")
    tgt = data.get("final_target")
    return Story(
        story_id=None,
        title=f"{metric} is Off Track",
        body=f"{metric} is at {val}, missing its target {tgt}.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Goal vs Actual",
        payload={"value": val, "target": tgt}
    )

def story_improving_status(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} has newly improved status",
        body=f"{metric} is newly on-track after being off-track previously.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Status Change",
        payload={}
    )

def story_worsening_status(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} has newly worsened status",
        body=f"{metric} is newly off-track after being on-track previously.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Status Change",
        payload={}
    )

def story_forecasted_on_track(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} is forecasted to beat target",
        body=f"{metric} is forecasted to end the period above the target.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Likely Status",
        payload={}
    )

def story_forecasted_off_track(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} is forecasted to miss target",
        body=f"{metric} is forecasted to end the period below the target.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Likely Status",
        payload={}
    )

def story_required_performance(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    needed_growth = data.get("needed_growth_rate", 0)
    return Story(
        story_id=None,
        title=f"{metric} must grow {needed_growth*100:.2f}% to meet target",
        body=f"{metric} must achieve {needed_growth*100:.2f}% growth over the next period to reach its goal.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Required Performance",
        payload={}
    )

def story_hold_steady(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} just needs to hold steady",
        body=f"{metric} is already at target and needs only to maintain its current level.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Likely Status",
        payload={}
    )

#############################################
# GROWTH
#############################################

def story_accelerating_growth(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    growth_rate = data.get("growth_rate")
    return Story(
        story_id=None,
        title=f"{metric} Growth is Accelerating",
        body=f"{metric} growth is now {growth_rate}%, up from previous periods.",
        date="2025-01-01",
        grain="Day",
        genre="Growth",
        theme="Growth Rates",
        payload={}
    )

def story_slowing_growth(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} Growth is Slowing",
        body=f"{metric} is showing slower growth compared to past intervals.",
        date="2025-01-01",
        grain="Day",
        genre="Growth",
        theme="Growth Rates",
        payload={}
    )

#############################################
# TRENDS (Long-Range, Trend Changes, etc.)
#############################################

def story_improving_performance_long_range(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} improved over the past period",
        body=f"Over the past few intervals, {metric} has had consistent growth or improvements.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Long-Range",
        payload={}
    )

def story_worsening_performance_long_range(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} has worsened over the past period",
        body=f"{metric} is showing a negative trend over the last intervals.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Long-Range",
        payload={}
    )

def story_stable_trend(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} trend is stable",
        body=f"The trend for {metric} has remained consistent with minimal fluctuations.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Trend Changes",
        payload={}
    )

def story_new_upward_trend(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"New upward trend for {metric}",
        body=f"{metric} has reversed a prior downward or flat trend and is now rising.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Trend Changes",
        payload={}
    )

def story_new_downward_trend(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"New downward trend for {metric}",
        body=f"{metric} recently flipped from stable or upward to downward.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Trend Changes",
        payload={}
    )

def story_performance_plateau(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} has plateaued",
        body=f"{metric} is hovering in a narrow band, indicating a potential plateau.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Trend Changes",
        payload={}
    )

def story_spike(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    val = data.get("current_value")
    return Story(
        story_id=None,
        title=f"{metric} spiked above trend",
        body=f"{metric} is currently {val}, which is above its normal range.",
        date="2025-01-01",
        grain="Day",
        genre="Trends",
        theme="Trend Exceptions",
        payload={}
    )

def story_drop(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    val = data.get("current_value")
    return Story(
        story_id=None,
        title=f"{metric} dropped below trend",
        body=f"{metric} is currently {val}, below its normal range.",
        date="2025-01-01",
        grain="Day",
        genre="Trends",
        theme="Trend Exceptions",
        payload={}
    )

def story_new_strongest_segment(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    seg = data.get("segment")
    return Story(
        story_id=None,
        title=f"{seg} is now the best-performing segment for {metric}",
        body=f"{seg} just overtook others to become the best-performing segment for {metric}.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Segment Changes",
        payload={}
    )

def story_new_weakest_segment(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    seg = data.get("segment")
    return Story(
        story_id=None,
        title=f"{seg} is now the worst-performing segment for {metric}",
        body=f"{seg} fell below others and became the weakest segment.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Segment Changes",
        payload={}
    )

def story_new_largest_segment(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    seg = data.get("segment")
    share = data.get("share_pct", 0)
    return Story(
        story_id=None,
        title=f"{seg} is now the largest slice of {metric}",
        body=f"{seg} now comprises {share}% of total {metric}, surpassing the previous leader.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Segment Changes",
        payload={}
    )

def story_new_smallest_segment(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    seg = data.get("segment")
    share = data.get("share_pct", 0)
    return Story(
        story_id=None,
        title=f"{seg} is now the smallest slice of {metric}",
        body=f"{seg} dropped to {share}% of {metric}, becoming the smallest segment.",
        date="2025-01-01",
        grain="Week",
        genre="Trends",
        theme="Segment Changes",
        payload={}
    )

def story_record_high(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    val = data.get("current_value")
    n = data.get("n_periods", 0)
    return Story(
        story_id=None,
        title=f"Record High for {metric}",
        body=f"{metric} reached {val}, the highest in the past {n} periods.",
        date="2025-01-01",
        grain="Day",
        genre="Trends",
        theme="Record Values",
        payload={}
    )

def story_record_low(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    val = data.get("current_value")
    n = data.get("n_periods", 0)
    return Story(
        story_id=None,
        title=f"Record Low for {metric}",
        body=f"{metric} dropped to {val}, the lowest in the past {n} periods.",
        date="2025-01-01",
        grain="Day",
        genre="Trends",
        theme="Record Values",
        payload={}
    )

def story_performance_against_historical_benchmarks(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} vs. Historical Benchmarks",
        body=f"{metric} is performing [X]% higher than last year and [Y]% vs. two years ago.",
        date="2025-01-01",
        grain="Day",
        genre="Trends",
        theme="Benchmark Comparisons",
        payload={}
    )


##################################
# ROOT CAUSES & SEGMENT DRIFT
##################################

def story_primary_root_cause_factor(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    factor = data.get("top_factor", "some factor")
    change_pct = data.get("change_pct", 0)
    return Story(
        story_id=None,
        title=f"Primary Driver of {metric}'s change is {factor}",
        body=f"{factor} contributed {change_pct}% to the change in {metric}.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Root Cause Summary",
        payload={}
    )

def story_growing_segment(data: Dict[str, Any]) -> Story:
    seg = data.get("segment")
    metric = data.get("metric_id")
    old_share = data.get("old_share", 0)
    new_share = data.get("new_share", 0)
    return Story(
        story_id=None,
        title=f"Growing {seg} segment for {metric}",
        body=f"{seg} share increased from {old_share}% to {new_share}%, driving {metric} upward.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Segment Drift",
        payload={}
    )

def story_shrinking_segment(data: Dict[str, Any]) -> Story:
    seg = data.get("segment")
    metric = data.get("metric_id")
    old_share = data.get("old_share", 0)
    new_share = data.get("new_share", 0)
    return Story(
        story_id=None,
        title=f"Shrinking {seg} segment for {metric}",
        body=f"{seg} share fell from {old_share}% to {new_share}%, reducing {metric}.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Segment Drift",
        payload={}
    )

def story_improving_segment(data: Dict[str, Any]) -> Story:
    seg = data.get("segment")
    metric = data.get("metric_id")
    improvement = data.get("improvement_pct", 0)
    return Story(
        story_id=None,
        title=f"Stronger {seg} segment for {metric}",
        body=f"{seg} improved by {improvement}%, helping push {metric} higher.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Segment Drift",
        payload={}
    )

def story_worsening_segment(data: Dict[str, Any]) -> Story:
    seg = data.get("segment")
    metric = data.get("metric_id")
    dip = data.get("dip_pct", 0)
    return Story(
        story_id=None,
        title=f"Weaker {seg} segment for {metric}",
        body=f"{seg} declined by {dip}%, dragging {metric} down.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Segment Drift",
        payload={}
    )

def story_improving_component(data: Dict[str, Any]) -> Story:
    comp = data.get("component", "component")
    metric = data.get("metric_id")
    inc_pct = data.get("inc_pct", 0)
    return Story(
        story_id=None,
        title=f"Growth in {comp} for {metric}",
        body=f"{comp} increased by {inc_pct}%, boosting {metric}.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Component Drift",
        payload={}
    )

def story_worsening_component(data: Dict[str, Any]) -> Story:
    comp = data.get("component", "component")
    metric = data.get("metric_id")
    dec_pct = data.get("dec_pct", 0)
    return Story(
        story_id=None,
        title=f"Decline in {comp} for {metric}",
        body=f"{comp} declined by {dec_pct}%, pulling {metric} down.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Component Drift",
        payload={}
    )

def story_stronger_influence_relationship(data: Dict[str, Any]) -> Story:
    infl = data.get("influence_metric")
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"Stronger Influence of {infl} on {metric}",
        body=f"{infl}'s influence on {metric} has grown significantly.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Influence Drift",
        payload={}
    )

def story_weaker_influence_relationship(data: Dict[str, Any]) -> Story:
    infl = data.get("influence_metric")
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"Weaker Influence of {infl} on {metric}",
        body=f"{infl}'s impact on {metric} has diminished significantly.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Influence Drift",
        payload={}
    )

def story_improving_influence_metric(data: Dict[str, Any]) -> Story:
    infl = data.get("influence_metric")
    metric = data.get("metric_id")
    inc_pct = data.get("inc_pct", 0)
    return Story(
        story_id=None,
        title=f"Growth in {infl} is driving up {metric}",
        body=f"{infl} increased by {inc_pct}%, which helped {metric}.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Influence Drift",
        payload={}
    )

def story_worsening_influence_metric(data: Dict[str, Any]) -> Story:
    infl = data.get("influence_metric")
    metric = data.get("metric_id")
    dec_pct = data.get("dec_pct", 0)
    return Story(
        story_id=None,
        title=f"Decline in {infl} is dragging down {metric}",
        body=f"{infl} dropped by {dec_pct}%, hurting {metric}.",
        date="2025-01-01",
        grain="Day",
        genre="Root Causes",
        theme="Influence Drift",
        payload={}
    )

##################################
# HEADWINDS / TAILWINDS
##################################

def story_unfavorable_driver_trend(data: Dict[str, Any]) -> Story:
    driver = data.get("driver_metric", "some driver")
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"Unfavorable Driver Trend for {metric}",
        body=f"{driver} is trending negatively, posing a risk for {metric}.",
        date="2025-01-01",
        grain="Day",
        genre="Headwinds/Tailwinds",
        theme="Headwind: Leading Indicators",
        payload={}
    )

def story_favorable_driver_trend(data: Dict[str, Any]) -> Story:
    driver = data.get("driver_metric", "some driver")
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"Favorable Driver Trend for {metric}",
        body=f"{driver} is trending positively, offering potential upside for {metric}.",
        date="2025-01-01",
        grain="Day",
        genre="Headwinds/Tailwinds",
        theme="Tailwind: Leading Indicators",
        payload={}
    )

def story_unfavorable_seasonality(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"Challenging seasonal period ahead for {metric}",
        body=f"Historically, {metric} dips during this period, posing a headwind.",
        date="2025-01-01",
        grain="Month",
        genre="Headwinds/Tailwinds",
        theme="Headwind: Seasonality",
        payload={}
    )

def story_favorable_seasonality(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"Favorable seasonal period ahead for {metric}",
        body=f"Historically, {metric} sees a boost during this period, a potential tailwind.",
        date="2025-01-01",
        grain="Month",
        genre="Headwinds/Tailwinds",
        theme="Tailwind: Seasonality",
        payload={}
    )

def story_volatility_alert(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    jump = data.get("vol_jump_pct", 0)
    return Story(
        story_id=None,
        title=f"Volatility Alert for {metric}",
        body=f"{metric} volatility jumped by {jump}%, indicating instability and forecast uncertainty.",
        date="2025-01-01",
        grain="Day",
        genre="Headwinds/Tailwinds",
        theme="Risk: Volatility",
        payload={}
    )

def story_concentration_risk(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    share = data.get("top_segments_share", 0)
    return Story(
        story_id=None,
        title=f"Concentration Risk in {metric}",
        body=f"Over {share}% of {metric} is driven by a few segments, magnifying downside risk if they falter.",
        date="2025-01-01",
        grain="Day",
        genre="Headwinds/Tailwinds",
        theme="Risk: Concentration",
        payload={}
    )

##################################
# PORTFOLIO-LEVEL STORIES
##################################

def story_portfolio_status_overview(data: Dict[str, Any]) -> Story:
    metrics_on_track = data.get("metrics_on_track", 0)
    metrics_newly_on = data.get("metrics_newly_on", 0)
    metrics_newly_off = data.get("metrics_newly_off", 0)
    return Story(
        story_id=None,
        title="Portfolio Status Overview",
        body=f"{metrics_on_track}% of metrics on track, {metrics_newly_on}% newly on-track, {metrics_newly_off}% newly off-track.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Overall GvA",
        payload={}
    )

def story_portfolio_performance_overview(data: Dict[str, Any]) -> Story:
    improving = data.get("improving_count", 0)
    stable = data.get("stable_count", 0)
    declining = data.get("declining_count", 0)
    return Story(
        story_id=None,
        title="Portfolio Performance Overview",
        body=f"Among the portfolio, {improving} metrics are improving, {stable} stable, {declining} declining.",
        date="2025-01-01",
        grain="Day",
        genre="Performance",
        theme="Overall Performance",
        payload={}
    )

##################################
# SEASONAL PATTERNS
##################################

def story_seasonal_pattern_match(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"{metric} following usual seasonal pattern",
        body=f"{metric} is aligning within historical seasonal norms for this period.",
        date="2025-01-01",
        grain="Day",
        genre="Trends",
        theme="Seasonal Patterns",
        payload={}
    )

def story_seasonal_pattern_break(data: Dict[str, Any]) -> Story:
    metric = data.get("metric_id")
    return Story(
        story_id=None,
        title=f"Unexpected seasonal deviation for {metric}",
        body=f"{metric} is diverging from its usual seasonal pattern by X%.",
        date="2025-01-01",
        grain="Day",
        genre="Trends",
        theme="Seasonal Pattern Break",
        payload={}
    )
