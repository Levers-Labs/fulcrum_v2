from dataclasses import dataclass, field
from typing import List

@dataclass
class FunnelStepDefinition:
    """
    A single step in a funnel, referencing a metric by ID.
    Example: 
      name="Signup", metric_id="signups"
    """
    name: str
    metric_id: str

@dataclass
class FunnelDefinition:
    """
    A funnel is an ordered list of steps. Each step references a metric ID.
    """
    funnel_id: str
    label: str
    steps: List[FunnelStepDefinition] = field(default_factory=list)

@dataclass
class FunnelCollection:
    """
    A container for multiple funnels.
    """
    funnels: List[FunnelDefinition] = field(default_factory=list)
