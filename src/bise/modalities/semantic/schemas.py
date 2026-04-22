from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SemanticManifestRecord:
    sample_id: str
    pair_id: str
    task_id: str
    scene_id: str
    dataset_name: str
    video_role: str
    video_path: str
    paired_video_path: str = ""
    cam_id: str = ""
    description_prompt_version: str = "description_prompt_v1"
    label_prompt_version: str = "label_prompt_with_taxonomy_v1"
    joint_prompt_version: str = "joint_prompt_with_taxonomy_v1"
    taxonomy_version: str = "taxonomy_v1"
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SemanticManifestRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            pair_id=str(payload.get("pair_id", payload["sample_id"])),
            task_id=str(payload["task_id"]),
            scene_id=str(payload["scene_id"]),
            dataset_name=str(payload["dataset_name"]),
            video_role=str(payload["video_role"]),
            video_path=str(payload["video_path"]),
            paired_video_path=str(payload.get("paired_video_path", "")),
            cam_id=str(payload.get("cam_id", "")),
            description_prompt_version=str(payload.get("description_prompt_version", "description_prompt_v1")),
            label_prompt_version=str(payload.get("label_prompt_version", "label_prompt_with_taxonomy_v1")),
            joint_prompt_version=str(payload.get("joint_prompt_version", "joint_prompt_with_taxonomy_v1")),
            taxonomy_version=str(payload.get("taxonomy_version", "taxonomy_v1")),
            status=str(payload.get("status", "pending")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class SemanticAnnotation:
    sample_id: str
    pair_id: str
    task_id: str
    scene_id: str
    dataset_name: str
    video_role: str
    video_path: str
    paired_video_path: str
    cam_id: str
    task_description: str
    capability_tags: List[str]
    task_complexity: str
    environment_tags: List[str]
    scene_category: str
    label_canonical_text: str
    status: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SemanticAnnotation":
        return cls(
            sample_id=str(payload["sample_id"]),
            pair_id=str(payload.get("pair_id", payload["sample_id"])),
            task_id=str(payload["task_id"]),
            scene_id=str(payload["scene_id"]),
            dataset_name=str(payload["dataset_name"]),
            video_role=str(payload["video_role"]),
            video_path=str(payload["video_path"]),
            paired_video_path=str(payload.get("paired_video_path", "")),
            cam_id=str(payload.get("cam_id", "")),
            task_description=str(payload["task_description"]).strip(),
            capability_tags=[str(tag).strip() for tag in payload.get("capability_tags", []) if str(tag).strip()],
            task_complexity=str(payload.get("task_complexity", "unknown")).strip(),
            environment_tags=[str(tag).strip() for tag in payload.get("environment_tags", []) if str(tag).strip()],
            scene_category=str(payload.get("scene_category", "unknown")).strip(),
            label_canonical_text=str(payload.get("label_canonical_text", "")).strip(),
            status=str(payload.get("status", "success")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class SemanticEmbeddingRecord:
    sample_id: str
    text_embedding: List[float]
    label_embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LabelEvaluationRecord:
    sample_id: str
    predicted_capability_tags: List[str]
    gold_capability_tags: List[str]
    capability_precision: float
    capability_recall: float
    capability_f1: float
    capability_exact_match: bool
    task_complexity_match: bool
    scene_category_match: bool
    environment_exact_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DescriptionReviewRecord:
    sample_id: str
    main_action_ok: bool
    object_ok: bool
    hallucination_free: bool
    reviewer: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptTemplate:
    version: str
    system_prompt: str
    user_template: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedLabelResult:
    capability_tags: List[str]
    task_complexity: str
    environment_tags: List[str]
    scene_category: str
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass
class ParsedSemanticResult:
    task_description: str
    capability_tags: List[str]
    task_complexity: str
    environment_tags: List[str]
    scene_category: str
    raw_payload: Optional[Dict[str, Any]] = None
