# fulcrum_app/storytelling_engine/story_data_structures.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Story:
    """
    Represents a single story to be displayed to users.
    """
    story_id: Optional[str]       # Could be auto-generated or None
    title: str
    body: str
    grain: str                    # 'Day', 'Week', or 'Month'
    date: str                     # The date for which the story is relevant, i.e. the beginning of the relevant grain.
    genre: str                    # e.g. 'Performance', 'Growth', 'Root Cause', ...
    theme: str                    # finer classification, e.g. 'Goal vs Actual', 'Status Change'
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "body": self.body,
            "date": self.date,
            "grain": self.grain,
            "genre": self.genre,
            "theme": self.theme,
            "payload": self.payload
        }
