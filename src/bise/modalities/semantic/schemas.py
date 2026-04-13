from dataclasses import dataclass, field
from typing import List


@dataclass
class VLMAnnotation:
    task_label: str
    task_description: str
    capability_tags: List[str] = field(default_factory=list)
