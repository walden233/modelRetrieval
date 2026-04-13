from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class EmbeddingSample:
    sample_id: str
    task_id: str
    scene_id: str
    video_embedding: List[float] = field(default_factory=list)
    trajectory_embedding: List[float] = field(default_factory=list)
    text_embedding: List[float] = field(default_factory=list)
    label_embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
