# fulcrum_app/storytelling_engine/story_service.py

from typing import List
from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from .story_data_structures import Story
from .story_generator import StoryGenerator

class StoryService:
    """
    A higher-level service that orchestrates story generation,
    applying any final salience or grain logic.
    """

    def __init__(self):
        self.generator = StoryGenerator()
        # If needed, store additional config (like cool_off_periods).

    def generate_stories_for_outputs(
        self, 
        pattern_outputs: List[PatternOutput]
    ) -> List[Story]:
        """
        Main entry point. Takes multiple PatternOutputs, calls the generator,
        then optionally filters or merges stories.
        
        1) For each PatternOutput, get stories via generator
        2) Possibly handle grain-based logic
        3) Possibly apply salience heuristics or 'cool_off' logic
        4) Return final stories
        """
        raw_stories = self.generator.generate_stories(pattern_outputs)

        # You could do post-processing, e.g. filter out low-salience or repetitive stories.
        final_stories = self._apply_salience_filters(raw_stories)

        return final_stories

    def _apply_salience_filters(self, stories: List[Story]) -> List[Story]:
        """
        A placeholder approach that might remove duplicates or skip low-signal stories 
        if they fail 'salience_heuristics.' 
        We'll just return them as-is for now.
        """
        return stories
