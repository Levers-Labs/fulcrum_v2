# Storytelling Engine

The **Storytelling Engine** converts structured outputs from the Intelligence Engine (Patterns) into end-user “Stories.” Each “Story” is textual plus some structured payload for rendering visualizations.

## Architecture

1. **Data Structures** (`data_structures.py`):
   - `Story`: a dataclass with `title`, `body`, `genre`, etc.

2. **Story Templates** (`story_templates.py`):
   - A registry of Python functions that map from pattern results to textual content.

3. **Story Generator** (`story_generator.py`):
   - Reads a `PatternOutput` (from Intelligence Engine).
   - Determines which template(s) to apply based on pattern name and scenario (e.g., “on_track” vs. “off_track”).
   - Returns one or more `Story` objects.

4. **Story Service** (`story_service.py`):
   - Higher-level orchestration. Allows you to feed in multiple pattern outputs and get back a consolidated list of stories.

## Typical Flow

1. **Obtain Pattern Outputs**:
   - e.g., from the Orchestrator, which has just run the Intelligence Engine on multiple metrics.

2. **Generate Stories**:
   ```python
   from storytelling_engine.story_service import StoryService

   service = StoryService()
   stories = service.generate_stories_for_outputs(pattern_outputs)
