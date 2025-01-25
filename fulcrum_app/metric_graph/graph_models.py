from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FormulaDefinition:
    expression_str: str

@dataclass
class InfluenceDefinition:
    source: str     # driver metric ID
    strength: float
    confidence: float

@dataclass
class DimensionDefinition:
    id: str
    label: str
    reference: str
    cube: str
    member_type: str

@dataclass
class MetricDefinition:
    id: str
    label: str
    definition: str
    unit: str
    owner_team: str
    formula: Optional[FormulaDefinition] = None

    # The references we parsed from formula, e.g. ["AcceptOpps","SQOToWinRate"]
    formula_references: List[str] = field(default_factory=list)

    influences: List[InfluenceDefinition] = field(default_factory=list)
    dimensions: List[DimensionDefinition] = field(default_factory=list)

    # Semantic layer references
    cube: Optional[str] = None
    member: Optional[str] = None
    member_type: Optional[str] = None
    time_dimension: Optional[str] = None

@dataclass
class MetricGraph:
    metrics: List[MetricDefinition] = field(default_factory=list)

    def get_metric_ids(self) -> List[str]:
        return [m.id for m in self.metrics]