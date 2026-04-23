import torch
from tqdm.auto import tqdm

from bise.retrieval.metrics import calculate_label_retrieval_metrics


def evaluate_video_retrieval(model, dataloader, device):
    # 评估阶段不算梯度，只做编码和检索指标统计。
    model.eval()
    human_features_list = []
    robot_features_list = []
    metadata = {
        "sample_ids": [],
        "pair_ids": [],
        "task_ids": [],
        "scene_ids": [],
        "camera_ids": [],
        "human_video_paths": [],
        "robot_video_paths": [],
        "task_indices": [],
        "scene_indices": [],
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Video"):
            human_videos = batch["human_pixel_values"].to(device)
            robot_videos = batch["robot_pixel_values"].to(device)
            outputs = model(human_videos, robot_videos)
            human_features_list.append(outputs["human_embeddings"].cpu())
            robot_features_list.append(outputs["robot_embeddings"].cpu())
            metadata["sample_ids"].extend(batch["sample_ids"])
            metadata["pair_ids"].extend(batch["pair_ids"])
            metadata["task_ids"].extend(batch["task_ids"])
            metadata["scene_ids"].extend(batch["scene_ids"])
            metadata["camera_ids"].extend(batch["camera_ids"])
            metadata["human_video_paths"].extend(batch["human_video_paths"])
            metadata["robot_video_paths"].extend(batch["robot_video_paths"])
            metadata["task_indices"].append(batch["task_indices"].cpu())
            metadata["scene_indices"].append(batch["scene_indices"].cpu())

    all_human_features = torch.cat(human_features_list, dim=0)
    all_robot_features = torch.cat(robot_features_list, dim=0)
    # 因为 embedding 都做了 L2 normalize，这里的点积就等价于余弦相似度。
    similarity_matrix = torch.matmul(all_human_features, all_robot_features.T).numpy()
    task_labels = torch.cat(metadata["task_indices"], dim=0).numpy()
    scene_labels = torch.cat(metadata["scene_indices"], dim=0).numpy()
    # 同时汇报两种检索方向、两种正样本粒度：
    # 1. human -> robot / robot -> human
    # 2. task-level / scene-level
    metrics = {
        "human_to_robot": {
            "task": calculate_label_retrieval_metrics(similarity_matrix, task_labels, task_labels),
            "scene": calculate_label_retrieval_metrics(similarity_matrix, scene_labels, scene_labels),
        },
        "robot_to_human": {
            "task": calculate_label_retrieval_metrics(similarity_matrix.T, task_labels, task_labels),
            "scene": calculate_label_retrieval_metrics(similarity_matrix.T, scene_labels, scene_labels),
        },
    }
    return {
        "metrics": metrics,
        "similarity_matrix": similarity_matrix,
        "human_embeddings": all_human_features.numpy(),
        "robot_embeddings": all_robot_features.numpy(),
        "metadata": metadata,
    }


def build_retrieval_cases(result: dict, direction: str = "human_to_robot", label_level: str = "task", top_k: int = 5):
    # 这个函数不参与训练，只是把检索结果整理成可读案例，方便看 Top-K 是否合理。
    similarity_matrix = result["similarity_matrix"]
    metadata = result["metadata"]
    if direction == "robot_to_human":
        similarity_matrix = similarity_matrix.T
        query_paths = metadata["robot_video_paths"]
        gallery_paths = metadata["human_video_paths"]
    else:
        query_paths = metadata["human_video_paths"]
        gallery_paths = metadata["robot_video_paths"]

    label_key = "task_ids" if label_level == "task" else "scene_ids"
    query_labels = metadata[label_key]
    gallery_labels = metadata[label_key]
    cases = []
    for query_index, query_label in enumerate(query_labels):
        ranked_indices = similarity_matrix[query_index].argsort()[::-1][:top_k]
        cases.append(
            {
                "query_index": query_index,
                "query_label": query_label,
                "query_path": query_paths[query_index],
                "retrieved": [
                    {
                        "index": int(candidate_index),
                        "label": gallery_labels[int(candidate_index)],
                        "path": gallery_paths[int(candidate_index)],
                        "score": float(similarity_matrix[query_index][int(candidate_index)]),
                        "is_positive": bool(gallery_labels[int(candidate_index)] == query_label),
                    }
                    for candidate_index in ranked_indices
                ],
            }
        )
    return cases
