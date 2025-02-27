# README: Storytelling Engine

---

## Overview

The **Storytelling Engine** is designed to:

- **Consume** outputs from **Patterns** (analytical modules that produce ‘PatternOutput‘ objects).
    
    ‘PatternOutput‘`PatternOutput`
    
- **Generate** user-facing “Stories” by mapping pattern data to textual and visual narratives.
- **Handle** different time grains (daily, weekly, monthly) with distinct logic if needed.
- **Support** salience filters (e.g., only show new or critical stories).

**Why** we built it:

- **Patterns** produce structured data (`PatternOutput`) describing broad analytical tasks.
- **Analysts** and **operators** need that data turned into clear, easily-digestible narratives—**Stories**—with a title, body, date, grain, etc.
- This engine centralizes the transformation from “analysis data” → “user-facing story” so developers can easily add or modify the logic.

---

## Key Files & Responsibilities

Below is a table summarizing each of the four primary files in the **Storytelling Engine**:

| **File** | **Purpose** | **Key Components** |
| --- | --- | --- |
| `story_data_structures.py` | Defines the fundamental `Story` class—how each story is represented (title, body, genre, date, grain, etc.). | - `Story` (dataclass): the minimal container for story text, metadata, and a structured payload. |
| `story_templates.py` | Houses the actual **template functions** that turn pattern results/scenarios into textual stories. Also holds an optional **template registry** for quick look-ups. | - A set of functions like `performance_status_on_track(...)` returning a single `Story` object. - A dictionary `STORY_TEMPLATE_REGISTRY` mapping `(pattern_name, scenario)` → function. |
| `story_generator.py` | Orchestrates how each **`PatternOutput`** is turned into one or multiple **`Story`** objects by referencing the templates. | - `StoryGenerator.generate_stories_from_pattern(...)`: reads the pattern name & scenario, calls the matching template function. |
| `story_service.py` | Provides a **high-level** API to generate stories for a list of `PatternOutput` objects. Optionally handles salience filters, deduplication, or time-grain logic. | - `StoryService.generate_stories_for_outputs(...)`: main entry point. - `_apply_salience_filters(...)`: example hook for filtering. |

---

## Data Flow & Architecture

1. **Patterns** produce **`PatternOutput`** objects. Each `PatternOutput` includes:
    - **`pattern_name`**: e.g. `"performance_status"`, `"root_cause"`.
    - **`results`**: a dictionary with fields like `{"status":"on_track","final_value":105,"final_target":100}`.
    - **`analysis_window`**: e.g. `{"start_date":"2025-01-01","end_date":"2025-01-31","grain":"Day"}`.
2. The **`StoryService`** is called with a list of these `PatternOutput`s.
3. The **`StoryService`** calls its **`generator.generate_stories(...)`** method, which:
    - Iterates each `PatternOutput`.
    - Calls **`generator.generate_stories_from_pattern(...)`** for each, which:
        - Looks up **`pattern_output.pattern_name`** + scenario in the **`STORY_TEMPLATE_REGISTRY`** (or a switch-case).
        - Calls the appropriate **template function** in `story_templates.py`.
        - Returns one or more **`Story`** objects, each describing a single narrative.
4. The final **`List[Story]`** is returned, potentially after applying salience or grain-based filters.

**Diagram** (conceptual):

```
 PatternOutput (PerformanceStatus)
   -> StoryService -> StoryGenerator ->
       story_templates -> [Story("On Track", ...)]

 PatternOutput (DimensionAnalysis)
   -> StoryService -> StoryGenerator ->
       story_templates -> [Story("Top 3 Segments", ...)]

```

---

## Story Object Structure

**`Story`** is the minimal container for a user-facing insight. Key fields:

| **Field** | **Type** | **Meaning** |
| --- | --- | --- |
| `story_id` | `Optional[str]` | Could be `None` or an auto-generated UUID for deduplication, referencing, or linking. |
| `title` | `str` | The high-level heading or short phrase describing the story. |
| `body` | `str` | The main text. Typically references the metric ID, the scenario, any relevant numbers (like “Value: X, Target: Y”). |
| `date` | `str` | The date relevant for the story. Often derived from `analysis_window["end_date"]` or set to something like “Week ending 2025-01-08.” |
| `grain` | `str` | `'Day'`, `'Week'`, or `'Month'`. Used for grouping or consistent display logic. |
| `genre` | `str` | A broad classification: `'Performance'`, `'Growth'`, `'Trends'`, `'Root Causes'`, `'Headwinds/Tailwinds'`, etc. |
| `theme` | `str` | A finer sub-classification: `'Goal vs Actual'`, `'Status Change'`, `'Dimension Drilldown'`, `'Forecast'`, etc. |
| `payload` | `Dict[str, Any]` | A structured JSON-like dictionary for visuals or detailed data. e.g. `{"final_value":105,"target":100,"analysis_window":{...}}`. |

This design ensures that any front-end or consumer can easily parse textual fields plus a structured `payload` for deeper interactions.

---

## Using the Storytelling Engine

Below is a step-by-step usage outline:

1. **Obtain** a list of `PatternOutput` objects (e.g., from your Orchestrator, which runs daily or weekly).
2. **Instantiate** a `StoryService`:
    
    ```
    from fulcrum_app.storytelling_engine.story_service import StoryService
    story_service = StoryService()
    ```
    
3. **Call** `story_service.generate_stories_for_outputs(pattern_outputs)`. For example:
    
    ```
    all_stories = story_service.generate_stories_for_outputs(my_pattern_outputs)
    ```
    
4. **Receive** a list of `Story` objects. Each has `title`, `body`, `grain`, etc. You might:
    
    ```
    for s in all_stories:
        print(f"Title: {s.title}\nBody: {s.body}\n---")
    ```
    
5. We can then **render** these stories in the UI, or store them in the DB for later display. We might also apply own additional filtering or grouping by `s.grain`.

### Basic Example

```
pattern_outputs = [
    PatternOutput(
       pattern_name="performance_status",
       pattern_version="1.0",
       metric_id="my_metric",
       analysis_window={"start_date":"2025-01-01","end_date":"2025-01-02","grain":"Day"},
       results={"status":"on_track","final_value":105,"final_target":100}
    )
]

all_stories = story_service.generate_stories_for_outputs(pattern_outputs)
for story in all_stories:
    print(story.title, story.body)

```

If the template registry has a match for `("performance_status","on_track")`, it returns a single “On Track” story.

---

## Adding New Stories

**When** you add a new scenario in a **Pattern** (e.g., you discover “Top 4 segments changed order” in `DimensionAnalysisPattern`), you do:

1. **Identify** a scenario name or code. For instance, `"segment_reorder"`.
2. **Add** logic in your `StoryGenerator.generate_stories_from_pattern(...)` to detect that scenario in `pattern_output.results`:
    
    ```
    if pattern_output.pattern_name == "dimension_analysis":
        if "reordered_segments" in pattern_output.results and pattern_output.results["reordered_segments"] is True:
            scenario = "segment_reorder"
            ...
    ```
    
3. **Create** a new function in `story_templates.py` that receives the partial data and returns a `Story`. Example:
    
    ```
    def dimension_segment_reorder(results: Dict[str, Any], analysis_window: Dict[str, str]) -> Story:
        ... # build a Title, Body
        return Story(...)
    ```
    
4. **Register** it in `STORY_TEMPLATE_REGISTRY`:
    
    ```
    STORY_TEMPLATE_REGISTRY[( "dimension_analysis", "segment_reorder" )] = dimension_segment_reorder
    ```
    

Hence, the next time your pattern sets `results["reordered_segments"]=True`, the generator picks up that scenario, calls the new template, and yields your story.

---

## Salience & Grain Logic

### Salience

You might want to **skip** repetitive or low-value stories. E.g.:

- If you reported “on track” for the last 3 days in a row, skip.
- If the difference from target is less than 1%, skip.

**Implementation** tip:

- Store a method `_apply_salience_filters(stories: List[Story]) -> List[Story]` in `StoryService`.
- For each story, check if it meets your thresholds. Remove or keep it.

### Grain

We handle daily vs. weekly vs. monthly logic typically in:

- The **`analysis_window`** dict includes `"grain":"Day"` or `"grain":"Week"` or `"grain":"Month"`.
- In **`story_templates.py`** or the generator, we interpret that to build the correct date label or apply different thresholds (like for weekly, skip small changes; for monthly, skip ephemeral day-to-day changes).

You can maintain a table of thresholds:

| **Grain** | **Threshold** | **Cool-Off** |
| --- | --- | --- |
| Day | e.g. 1% | 2 days |
| Week | e.g. 2% | 1 week |
| Month | e.g. 3% | 1 month |

So if you see `grain="Week"`, you apply the week threshold or skip stories that are below 2% difference from last story.

---

## Example Walkthrough

Below is a short table showing a typical usage scenario:

| **Step** | **Action** | **Data** |
| --- | --- | --- |
| 1. Patterns run | The Orchestrator calls `PerformanceStatusPattern` (and others). They produce a `PatternOutput` with `{"status":"off_track","final_value":90,"final_target":100}`, etc. | `PatternOutput(pattern_name="performance_status", results={"status":"off_track","final_value":90,"final_target":100}, analysis_window={"grain":"Day",...})` |
| 2. Pass to StoryService | We do `story_service.generate_stories_for_outputs([...])`. | `[PatternOutput(...), ...]` is our input array. |
| 3. Generator sees pattern_name= "performance_status" & results["status"]="off_track" | In `story_generator.py`, it looks up `("performance_status","off_track")` in `STORY_TEMPLATE_REGISTRY`. | Finds `performance_status_off_track(...)` function. |
| 4. Template builds story | The template merges the final value & target into a textual message. Returns a `Story`. | Example: `title="my_metric is Off Track"`, `body="Your metric 'my_metric' is below target. Value=90, Target=100."` |
| 5. Stories returned | The final `[Story(...)]` is returned to the caller. | The UI or consumer can then display that story under "Performance -> Goal vs Actual." |

And that’s it! The process repeats for each pattern, potentially producing multiple stories if you define multiple scenarios in your generator.