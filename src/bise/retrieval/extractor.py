from typing import Any, Dict

from bise.common.schemas import EmbeddingSample


def build_embedding_sample(sample_id: str, task_id: str, scene_id: str, embeddings: Dict[str, Any], metadata: Dict[str, Any]):
    return EmbeddingSample(
        sample_id=sample_id,
        task_id=task_id,
        scene_id=scene_id,
        video_embedding=embeddings.get("video_embedding", []),
        trajectory_embedding=embeddings.get("trajectory_embedding", []),
        text_embedding=embeddings.get("text_embedding", []),
        label_embedding=embeddings.get("label_embedding", []),
        metadata=metadata,
    )
