from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Story:
    """
    Represents a single story displayed to users.
    """
    story_id: Optional[str]   # Could be an auto-generated UUID or None
    title: str
    body: str
    date: str
    grain: str
    genre: str
    theme: str
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