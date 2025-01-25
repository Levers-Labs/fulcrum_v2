from typing import List
from fulcrum_app.intelligence_engine.data_structures import PatternOutput
from .data_structures import Story
from .story_generator import StoryGenerator

class StoryService:
    """
    A higher-level service that orchestrates story generation,
    salience filtering, and caching if desired.
    """

    def __init__(self):
        self.generator = StoryGenerator()
        # Could have a story cache or DB here if you want

    def generate_stories_for_outputs(self, pattern_outputs: List[PatternOutput]) -> List[Story]:
        """
        Main entry point. Takes multiple PatternOutputs, generates stories, and returns them.
        """
        stories = self.generator.generate_stories(pattern_outputs)
        # Optionally filter or rank them by salience
        # For now, we'll just return them all
        return stories
