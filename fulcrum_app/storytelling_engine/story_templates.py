"""
story_templates_part2.py

Implements the second half (stories 26-50) from the table:
New Largest Segment, New Smallest Segment, Record High, Record Low, 
Benchmarks, Primary Root Cause Factor, etc.

Combine with story_templates_part1 for a full set.
"""

from typing import Dict, Any
from .story_data_structures import Story

def _resolve_date_and_grain(analysis_window: Dict[str, str]) -> (str, str):
    grain = analysis_window.get("grain","Day")
    start_date = analysis_window.get("start_date","")
    end_date = analysis_window.get("end_date","")

    if grain.lower() == "day":
        date_str = end_date if end_date else start_date
        grain_str = "Day"
    elif grain.lower() == "week":
        date_str = f"Week ending {end_date}" if end_date else "Unknown week"
        grain_str = "Week"
    elif grain.lower() == "month":
        date_str = end_date[:7] if len(end_date)>=7 else end_date
        grain_str = "Month"
    else:
        date_str = end_date
        grain_str = grain.capitalize()

    return date_str, grain_str

##############################################################################
# 26) New Largest Segment
##############################################################################

def story_new_largest_segment_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    seg = results.get("segment","SegmentA")
    dimension = results.get("dimension","Dimension")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    prev_leader = results.get("previous_leader","SegmentB")
    z_val = results.get("z",0.0)

    metric = results.get("metric","Metric")
    date_str, grain_str = _resolve_date_and_grain(analysis_window)

    title = f"{seg} is now the most represented segment"
    body = (
        f"{seg} now comprises the largest share of {dimension}, at {x_val}%, up from {y_val}% the prior {grain_str}. "
        f"This surpasses the previously most represented segment {prev_leader}, which now comprises {z_val}% of {dimension}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Trends",
        theme="Segment Changes",
        payload={
            "segment": seg,
            "dimension": dimension,
            "x": x_val,
            "y": y_val,
            "previous_leader": prev_leader,
            "z": z_val
        }
    )

##############################################################################
# 27) New Smallest Segment
##############################################################################

def story_new_smallest_segment_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    seg = results.get("segment","SegmentA")
    dimension = results.get("dimension","Dimension")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    prev_smallest = results.get("previous_smallest","SegmentB")
    z_val = results.get("z",0.0)
    a_val = results.get("a",0.0)

    metric = results.get("metric","Metric")
    date_str, grain_str = _resolve_date_and_grain(analysis_window)

    title = f"{seg} is now the least represented segment"
    body = (
        f"{seg} now comprises the smallest share of {dimension}, at {x_val}%, down from {y_val}% in the prior {grain_str}. "
        f"The previously least represented segment {prev_smallest} now comprises {z_val}% of {dimension}, "
        f"up from {a_val}%."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Trends",
        theme="Segment Changes",
        payload={
            "segment": seg,
            "dimension": dimension,
            "x": x_val,
            "y": y_val,
            "previous_smallest": prev_smallest,
            "z": z_val,
            "a": a_val
        }
    )

##############################################################################
# 28) Record High
##############################################################################

def story_record_high_portfolio_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    y_rank = results.get("y_rank",1)
    period_label = results.get("period_label","Day")
    value_val = results.get("value",0.0)
    n_val = results.get("n",1)
    range_label = results.get("range_label","days")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Record High"
    body = (
        f"The {grain_str} value for {metric} of {value_val} is now the {y_rank}th highest {period_label} value "
        f"in the past {n_val} {range_label}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Trends",
        theme="Record Values",
        payload={
            "metric": metric,
            "value": value_val,
            "y_rank": y_rank,
            "period_label": period_label,
            "n": n_val,
            "range_label": range_label
        }
    )

##############################################################################
# 29) Record Low
##############################################################################

def story_record_low_portfolio_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    y_rank = results.get("y_rank",1)
    period_label = results.get("period_label","Day")
    value_val = results.get("value",0.0)
    n_val = results.get("n",1)
    range_label = results.get("range_label","days")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Record Low"
    body = (
        f"The {grain_str} value for {metric} of {value_val} is now the {y_rank}th lowest {period_label} value "
        f"in {n_val} {range_label}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Trends",
        theme="Record Values",
        payload={
            "metric": metric,
            "value": value_val,
            "y_rank": y_rank,
            "period_label": period_label,
            "n": n_val,
            "range_label": range_label
        }
    )

##############################################################################
# 30) Benchmarks
##############################################################################

def story_benchmarks_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    reference1 = results.get("ref1","last year")
    z_val = results.get("z",0.0)
    reference2 = results.get("ref2","two years ago")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Performance Against Historical Benchmarks"
    body = (
        f"This day marks the {y_val}th highest-performing {grain_str} in the past [n] {grain_str}(s), "
        f"with the current {grain_str}'s performance of {metric} at {x_val} coming in {y_val}% "
        f"[higher|lower] than this time {reference1} and {z_val}% [higher|lower] than this time {reference2}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Trends",
        theme="Benchmark Comparisons",
        payload={
            "metric": metric,
            "x": x_val,
            "y": y_val,
            "ref1": reference1,
            "z": z_val,
            "ref2": reference2
        }
    )

##############################################################################
# 31) Primary Root Cause Factor
##############################################################################

def story_primary_root_cause_factor_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    x_val = results.get("x",0.0)
    metric = results.get("metric","Metric")
    factor1 = results.get("factor1","Driver1")
    factor1_dir = results.get("factor1_dir","increase")
    factor1_val = results.get("factor1_val",0.0)
    factor1_pct = results.get("factor1_pct",0.0)
    second_factor = results.get("factor2","Driver2")
    second_dir = results.get("factor2_dir","increase")
    second_val = results.get("factor2_val",0.0)
    second_pct = results.get("factor2_pct",0.0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Primary Root Cause Factor"
    body = (
        f"The primary driver of the {x_val}% change in {metric} was the {factor1_val}% {factor1_dir} in {factor1}, "
        f"which drove {factor1_pct}% of the change. The second most significant driver was the "
        f"{second_val}% {second_dir} in {second_factor}, which drove {second_pct}% of the change."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Root Cause Summary",
        payload={
            "x": x_val,
            "metric": metric,
            "factor1": factor1,
            "factor1_dir": factor1_dir,
            "factor1_val": factor1_val,
            "factor1_pct": factor1_pct,
            "factor2": second_factor,
            "factor2_dir": second_dir,
            "factor2_val": second_val,
            "factor2_pct": second_pct
        }
    )

##############################################################################
# 32) Growing Segment
##############################################################################

def story_growing_segment_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    slice_ = results.get("slice","SliceA")
    dimension = results.get("dimension","Dimension")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    z_val = results.get("z",0.0)
    direction = results.get("direction","upward")  # "upward" or "downward"

    metric = results.get("metric","Metric")
    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Key Driver: Growing slice share"
    body = (
        f"The share of {dimension} that is {slice_} increased from {x_val}% to {y_val}% over the past {grain_str}. "
        f"This increase contributed {z_val}% {direction} pressure on {metric}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Segment Drift",
        payload={
            "slice": slice_,
            "dimension": dimension,
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "direction": direction,
            "metric": metric
        }
    )

##############################################################################
# 33) Shrinking Segment
##############################################################################

def story_shrinking_segment_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    slice_ = results.get("slice","SliceA")
    dimension = results.get("dimension","Dimension")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    z_val = results.get("z",0.0)
    direction = results.get("direction","downward")

    metric = results.get("metric","Metric")
    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Key Driver: Falling slice share"
    body = (
        f"For {metric}, the share of {dimension} that is {slice_} has decreased from {x_val}% to {y_val}% "
        f"over the past {grain_str}. This decrease contributed {z_val}% {direction} pressure on {metric}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Segment Drift",
        payload={
            "slice": slice_,
            "dimension": dimension,
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "direction": direction,
            "metric": metric
        }
    )

##############################################################################
# 34) Improving Segment
##############################################################################

def story_improving_segment_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    dimension = results.get("dimension","Dimension")
    slice_ = results.get("slice","SliceA")
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    z_val = results.get("z",0.0)
    direction = results.get("direction","upward")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = f"Key Driver: Stronger {slice_} segment"
    body = (
        f"Over the past {grain_str}, when {dimension} is {slice_}, {metric} is {x_val}. This is an increase of {y_val}% "
        f"relative to the prior range, and this increase contributed {z_val}% {direction} pressure on {metric}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Segment Drift",
        payload={
            "dimension": dimension,
            "slice": slice_,
            "metric": metric,
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "direction": direction
        }
    )

##############################################################################
# 35) Worsening Segment
##############################################################################

def story_worsening_segment_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    dimension = results.get("dimension","Dimension")
    slice_ = results.get("slice","SliceA")
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    z_val = results.get("z",0.0)
    direction = results.get("direction","downward")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = f"Key Driver: Weaker {slice_} segment"
    body = (
        f"Over the past {grain_str}, when {dimension} is {slice_}, {metric} is {x_val}. This is a decrease of {y_val}% "
        f"relative to the prior {grain_str}, and this decrease contributed {z_val}% {direction} pressure on {metric}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Segment Drift",
        payload={
            "dimension": dimension,
            "slice": slice_,
            "metric": metric,
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "direction": direction
        }
    )

##############################################################################
# 36) Improving Component
##############################################################################

def story_improving_component_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    component = results.get("component","ComponentA")
    x_val = results.get("x",0.0)
    metric = results.get("metric","Metric")
    y_val = results.get("y",0.0)
    direction = results.get("direction","upward")
    z_val = results.get("z",0.0)
    period_label = results.get("period_label","period")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = f"Key Driver: Growth in {component}"
    body = (
        f"The {x_val}% increase in {component} over the past {period_label} contributed {y_val}% {direction} pressure on {metric} "
        f"and accounts for {z_val}% of its overall change."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Component Drift",
        payload={
            "component": component,
            "x": x_val,
            "metric": metric,
            "y": y_val,
            "direction": direction,
            "z": z_val,
            "period_label": period_label
        }
    )

##############################################################################
# 37) Worsening Component
##############################################################################

def story_worsening_component_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    component = results.get("component","ComponentA")
    x_val = results.get("x",0.0)
    metric = results.get("metric","Metric")
    y_val = results.get("y",0.0)
    direction = results.get("direction","downward")
    z_val = results.get("z",0.0)
    period_label = results.get("period_label","period")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = f"Key Driver: Decline in {component}"
    body = (
        f"The {x_val}% decrease in {component} over the past {period_label} contributed {y_val}% {direction} pressure on {metric} "
        f"and accounts for {z_val}% of its overall change."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Component Drift",
        payload={
            "component": component,
            "x": x_val,
            "metric": metric,
            "y": y_val,
            "direction": direction,
            "z": z_val,
            "period_label": period_label
        }
    )

##############################################################################
# 38) Stronger Influence Relationship
##############################################################################

def story_stronger_influence_relationship_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    infl = results.get("influence","Influence Metric")
    out = results.get("output_metric","Output Metric")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    inc_dec = results.get("inc_dec","increase")
    z_val = results.get("z",0.0)
    period_label = results.get("period_label","period")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = f"Key Driver: Stronger influence of {infl}"
    body = (
        f"The influence of {infl} on {out} is growing stronger. A {x_val}% {inc_dec} in {infl} "
        f"is associated with a {y_val}% {inc_dec} in {out} — up from {z_val}% the prior {period_label}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Influence Drift",
        payload={
            "influence": infl,
            "output_metric": out,
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "inc_dec": inc_dec,
            "period_label": period_label
        }
    )

##############################################################################
# 39) Weaker Influence Relationship
##############################################################################

def story_weaker_influence_relationship_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    infl = results.get("influence","Influence Metric")
    out = results.get("output_metric","Output Metric")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    inc_dec = results.get("inc_dec","increase")
    z_val = results.get("z",0.0)
    period_label = results.get("period_label","period")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = f"Key Driver: Weaker influence of {infl}"
    body = (
        f"The influence of {infl} on {out} is getting weaker. A {x_val}% {inc_dec} in {infl} "
        f"is associated with a {y_val}% {inc_dec} in {out} — down from {z_val}% the prior {period_label}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Influence Drift",
        payload={
            "influence": infl,
            "output_metric": out,
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "inc_dec": inc_dec,
            "period_label": period_label
        }
    )

##############################################################################
# 40) Improving Influence Metric
##############################################################################

def story_improving_influence_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    infl = results.get("influence","Influence Metric")
    out = results.get("output_metric","Output Metric")
    x_val = results.get("x",0.0)
    direction = results.get("direction","upward")
    z_val = results.get("z",0.0)
    date_str, grain_str = _resolve_date_and_grain(analysis_window)

    title = f"Key Driver: Growth in {infl}"
    body = (
        f"The {x_val}% increase in {infl} over the past {grain_str} contributed {z_val}% {direction} pressure on {out} "
        f"and accounts for a significant portion of its overall change."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Influence Drift",
        payload={
            "influence": infl,
            "output_metric": out,
            "x": x_val,
            "direction": direction,
            "z": z_val
        }
    )

##############################################################################
# 41) Worsening Influence Metric
##############################################################################

def story_worsening_influence_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    infl = results.get("influence","Influence Metric")
    out = results.get("output_metric","Output Metric")
    x_val = results.get("x",0.0)
    direction = results.get("direction","downward")
    z_val = results.get("z",0.0)
    date_str, grain_str = _resolve_date_and_grain(analysis_window)

    title = f"Key Driver: Decline in {infl}"
    body = (
        f"The {x_val}% decrease in {infl} over the past {grain_str} contributed {z_val}% {direction} pressure on {out} "
        f"and accounts for a significant portion of its overall change."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Root Causes",
        theme="Influence Drift",
        payload={
            "influence": infl,
            "output_metric": out,
            "x": x_val,
            "direction": direction,
            "z": z_val
        }
    )

##############################################################################
# 42) Unfavorable Driver Trend (Headwind: Leading Indicators)
##############################################################################

def story_unfavorable_driver_trend_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    driver = results.get("driver_metric","Driver Metric")
    out = results.get("metric","Metric")
    x_val = results.get("x",0.0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Leading indicator suggests potential risk"
    body = (
        f"{driver} is showing a negative {grain_str}-over-{grain_str} trend that may pose a risk to {out}. "
        f"If this continues, {out} could see up to {x_val}% downside relative to its current trajectory."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Headwinds/Tailwinds",
        theme="Headwind: Leading Indicators",
        payload={
            "driver_metric": driver,
            "metric": out,
            "x": x_val
        }
    )

##############################################################################
# 43) Favorable Driver Trend (Tailwind: Leading Indicators)
##############################################################################

def story_favorable_driver_trend_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    driver = results.get("driver_metric","Driver Metric")
    out = results.get("metric","Metric")
    x_val = results.get("x",0.0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Leading indicator suggests potential upside"
    body = (
        f"{driver} is showing a positive {grain_str}-over-{grain_str} trend that may lift {out}. "
        f"If sustained, {out} could see up to {x_val}% upside."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Headwinds/Tailwinds",
        theme="Tailwind: Leading Indicators",
        payload={
            "driver_metric": driver,
            "metric": out,
            "x": x_val
        }
    )

##############################################################################
# 44) Unfavorable Seasonal Trend (Headwind: Seasonality)
##############################################################################

def story_unfavorable_seasonal_trend_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    n_val = results.get("n",1)
    end_date = results.get("end_date","some date")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Challenging seasonal period ahead"
    body = (
        f"Historically, {metric} dips by {x_val}% over the coming {n_val} {grain_str}(s). "
        f"If the pattern holds, we may see a slowdown through {end_date}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Headwinds/Tailwinds",
        theme="Headwind: Seasonality",
        payload={
            "metric": metric,
            "x": x_val,
            "n": n_val,
            "end_date": end_date
        }
    )

##############################################################################
# 45) Favorable Seasonal Trend (Tailwind: Seasonality)
##############################################################################

def story_favorable_seasonal_trend_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    n_val = results.get("n",1)
    end_date = results.get("end_date","some date")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Beneficial seasonal period ahead"
    body = (
        f"Historically, {metric} increases by {x_val}% over the coming {n_val} {grain_str}(s). "
        f"If this trend repeats, expect a lift through {end_date}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Headwinds/Tailwinds",
        theme="Tailwind: Seasonality",
        payload={
            "metric": metric,
            "x": x_val,
            "n": n_val,
            "end_date": end_date
        }
    )

##############################################################################
# 46) Volatility Alert
##############################################################################

def story_volatility_alert_portfolio_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    n_val = results.get("n",1)
    y_val = results.get("y",0.0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Significant increase in metric volatility"
    body = (
        f"{metric} has experienced a {x_val}% jump in volatility over the last {n_val} {grain_str}(s), up from {y_val}%. "
        f"This indicates potential instability and elevated forecasting uncertainty."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Headwinds/Tailwinds",
        theme="Risk: Volatility",
        payload={
            "metric": metric,
            "x": x_val,
            "n": n_val,
            "y": y_val
        }
    )

##############################################################################
# 47) Concentration Risk
##############################################################################

def story_concentration_risk_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    segA = results.get("segA","SegmentA")
    segB = results.get("segB","SegmentB")
    segC = results.get("segC","SegmentC")

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Heavy reliance on a few segments"
    body = (
        f"Over {x_val}% of {metric} comes from {segA}, {segB}, and {segC}, signaling a concentration risk "
        f"that could magnify negative impacts if these segments falter."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Headwinds/Tailwinds",
        theme="Risk: Concentration",
        payload={
            "metric": metric,
            "x": x_val,
            "segA": segA,
            "segB": segB,
            "segC": segC
        }
    )

##############################################################################
# 48) Portfolio Status Overview
##############################################################################

def story_portfolio_status_overview_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    z_val = results.get("z",0.0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Summary of portfolio health"
    body = (
        f"Overall, {x_val}% of metrics on track vs target, with {y_val}% newly On-Track and {z_val}% newly Off-Track."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Performance",
        theme="Overall GvA",
        payload={
            "pct_on_track": x_val,
            "pct_newly_on": y_val,
            "pct_newly_off": z_val
        }
    )

##############################################################################
# 49) Portfolio Performance Overview
##############################################################################

def story_portfolio_performance_overview_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    x_val = results.get("x",0)
    y_val = results.get("y",0)
    z_val = results.get("z",0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Summary of portfolio momentum"
    body = (
        f"Overall, the momentum across the portfolio of metrics is positive, with {x_val} metrics improving, "
        f"{y_val} stable, and {z_val} declining."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Performance",
        theme="Overall Performance",
        payload={
            "improving": x_val,
            "stable": y_val,
            "declining": z_val
        }
    )

##############################################################################
# 50) Seasonal Pattern Match
##############################################################################

def story_seasonal_pattern_match_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Expected seasonal behavior"
    body = (
        f"{metric} is following its usual seasonal pattern, aligning within {x_val}% of historical norms for this {grain_str}. "
        f"No unexpected variation is evident."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Trends",
        theme="Seasonal Patterns",
        payload={
            "metric": metric,
            "x": x_val
        }
    )

##############################################################################
# 51) Seasonal Pattern Break
##############################################################################

def story_seasonal_pattern_break_metric(results: Dict[str,Any], analysis_window: Dict[str,str]) -> Story:
    metric = results.get("metric","Metric")
    x_val = results.get("x",0.0)
    y_val = results.get("y",0.0)
    z_val = results.get("z",0.0)

    date_str, grain_str = _resolve_date_and_grain(analysis_window)
    title = "Unexpected seasonal deviation"
    body = (
        f"{metric} is diverging from its usual seasonal trend by {x_val}%. Historically, we'd expect a seasonal {y_val}% "
        f"this period, but we're currently seeing {z_val}."
    )
    return Story(
        story_id=None,
        title=title,
        body=body,
        date=date_str,
        grain=grain_str,
        genre="Trends",
        theme="Seasonal Pattern Break",
        payload={
            "metric": metric,
            "x": x_val,
            "y": y_val,
            "z": z_val
        }
    )

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
