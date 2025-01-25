from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InfluenceDefinition:
    """A non-deterministic relationship: metric A influences metric B."""
    source: str     # which metric ID is the driver?
    strength: float
    confidence: float

@dataclass
class DimensionDefinition:
    """Represents a dimension that can slice a metric."""
    id: str
    label: str
    reference: str
    cube: str
    member_type: str

@dataclass
class FormulaDefinition:
    """
    A formula for this metric, e.g. {AcceptOpps} * {SQOToWinRate}.
    In the future, we may parse this expression to discover references.
    """
    expression_str: str

@dataclass
class MetricDefinition:
    """
    A single metric definition from the TOML.
    """
    id: str
    label: str
    definition: str
    unit: str
    owner_team: str

    # Optional formula-based relationships
    formula: Optional[FormulaDefinition] = None

    # List of influences
    influences: List[InfluenceDefinition] = field(default_factory=list)

    # Dimensions for slicing
    dimensions: List[DimensionDefinition] = field(default_factory=list)

    # Semantic layer references
    cube: Optional[str] = None
    member: Optional[str] = None
    member_type: Optional[str] = None
    time_dimension: Optional[str] = None

@dataclass
class MetricGraph:
    """
    A container for all metric definitions.
    """
    metrics: List[MetricDefinition] = field(default_factory=list)

    def get_metric_ids(self) -> List[str]:
        """Returns the list of all metric IDs in the graph."""
        return [m.id for m in self.metrics]
