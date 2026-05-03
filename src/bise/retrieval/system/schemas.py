from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FeatureRecord:
    # 单条特征向量的索引记录：真正的向量在 features/*.npy 中，这里只保存数组路径和 row_index。
    # 这样可以避免在 manifest 里重复写大向量，同时支持一个样本拥有多个模态特征。
    feature_id: str
    entity_key: str
    domain: str
    modality: str
    array_path: str
    row_index: int
    task_id: str
    scene_id: str
    camera_id: str | None = None
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureRecord":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GalleryItem:
    # 机器人侧检索候选。生产系统只需要 gallery；实验时再额外保存 human query。
    # feature_ids 将 video / trajectory / semantic_text 等模态映射到对应 FeatureRecord。
    gallery_id: str
    entity_key: str
    domain: str
    task_id: str
    scene_id: str
    camera_id: str | None = None
    feature_ids: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GalleryItem":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryItem:
    # 离线评估用的人类侧 query。真实在线查询不要求 query 预先存在于检索库。
    query_id: str
    entity_key: str
    domain: str
    task_id: str
    scene_id: str
    camera_id: str | None = None
    feature_ids: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryItem":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalQuery:
    # 在线查询输入的统一表示：视频、轨迹、语义可以任意缺失，但至少需要一个 embedding。
    # raw video 会在脚本层先编码成 video_embedding，系统核心只消费向量。
    query_id: str | None = None
    video_embedding: list[float] | None = None
    trajectory_embedding: list[float] | None = None
    semantic_text_embedding: list[float] | None = None
    semantic_label_embedding: list[float] | None = None
    semantic_combined_embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_feature_map(self) -> dict[str, list[float]]:
        mapping = {
            "video": self.video_embedding,
            "trajectory": self.trajectory_embedding,
            "semantic_text": self.semantic_text_embedding,
            "semantic_label": self.semantic_label_embedding,
            "semantic_combined": self.semantic_combined_embedding,
        }
        return {name: values for name, values in mapping.items() if values is not None}


@dataclass
class RetrievalResult:
    # 单个候选的检索结果。保留 fused_score 和各模态分数，方便分析融合到底由哪个模态主导。
    query_id: str
    gallery_id: str
    entity_key: str
    task_id: str
    scene_id: str
    camera_id: str | None
    fused_score: float
    modality_scores: dict[str, float]
    modality_ranks: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
