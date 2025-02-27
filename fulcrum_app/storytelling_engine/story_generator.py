# fulcrum_app/storytelling_engine/story_generator.py

from typing import List
from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from .story_data_structures import Story
from .story_templates import STORY_TEMPLATE_REGISTRY

class StoryGenerator:
    """
    Takes PatternOutput objects and produces user-facing Story objects
    by mapping to known templates or fallback logic.
    """

    def __init__(self):
        pass

    def generate_stories_from_pattern(self, pattern_output: PatternOutput) -> List[Story]:
        """
        Convert a single PatternOutput into zero or more Story objects.
        """
        pattern_name = pattern_output.pattern_name
        results = {
            **pattern_output.results,
            "metric_id": pattern_output.metric_id  # so templates can reference
        }
        analysis_window = pattern_output.analysis_window

        if pattern_name == "performance_status":
            # typical scenario is results["status"] => 'on_track','off_track','no_data' ...
            final_status = results.get("status", "no_data")
            # find a matching template in STORY_TEMPLATE_REGISTRY
            key = (pattern_name, final_status)
            if key in STORY_TEMPLATE_REGISTRY:
                story_fn = STORY_TEMPLATE_REGISTRY[key]
                story = story_fn(results, analysis_window)
                return [story]
            else:
                # no matching template => produce fallback or empty
                return []
        else:
            # Patterns we haven't mapped => produce fallback or empty
            return []

    def generate_stories(self, pattern_outputs: List[PatternOutput]) -> List[Story]:
        """
        Convert a list of PatternOutput objects into a list of Story objects
        by calling generate_stories_from_pattern on each.
        """
        all_stories = []
        for p_out in pattern_outputs:
            st_list = self.generate_stories_from_pattern(p_out)
            all_stories.extend(st_list)
        return all_stories
