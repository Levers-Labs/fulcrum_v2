from typing import List
from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from .data_structures import Story
from .story_templates import STORY_TEMPLATE_REGISTRY

class StoryGenerator:
    """
    Takes in PatternOutput objects and produces user-facing stories
    by applying template logic.
    """

    def __init__(self):
        pass

    def generate_stories_from_pattern(self, pattern_output: PatternOutput) -> List[Story]:
        """
        Convert a single PatternOutput into zero or more Story objects.
        """
        # 1) Identify pattern name
        pname = pattern_output.pattern_name

        # 2) Extract results we need
        results_dict = {
            **pattern_output.results,
            "metric_id": pattern_output.metric_id,
            "analysis_window": pattern_output.analysis_window,
        }

        # We'll create logic for performance_status as an example
        if pname == "performance_status":
            final_status = results_dict.get("final_status", "no_data")
            if final_status == "no_data":
                scenario = "no_data"
            elif final_status == "on_track":
                scenario = "on_track"
            elif final_status == "off_track":
                scenario = "off_track"
            else:
                scenario = "off_track"  # default if unknown

            # Attempt to find a matching template
            tpl = STORY_TEMPLATE_REGISTRY.get((pname, scenario))
            if tpl:
                story = tpl(results_dict)
                return [story]
            else:
                # No matching template => skip or produce fallback story
                return []
        
        elif pname == "root_cause":
            # Example: you might produce multiple stories from root causes
            # We'll skip for now:
            return []
        else:
            # Patterns we don't have a template for
            return []

    def generate_stories(self, pattern_outputs: List[PatternOutput]) -> List[Story]:
        """
        Convert a list of PatternOutput objects into a list of Story objects.
        """
        all_stories = []
        for p_out in pattern_outputs:
            st_list = self.generate_stories_from_pattern(p_out)
            all_stories.extend(st_list)
        return all_stories
