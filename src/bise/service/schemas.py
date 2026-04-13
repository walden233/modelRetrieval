from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class QueryRequest:
    top_k: int = 10
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryResponse:
    sample_ids: List[str]
    scores: List[float]
