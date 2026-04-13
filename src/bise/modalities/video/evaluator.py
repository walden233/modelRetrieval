import torch
from tqdm.auto import tqdm

from bise.retrieval.metrics import calculate_retrieval_metrics_grouped


def evaluate_video_retrieval(model, dataloader, loss_fn, device, group_size: int):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    human_features_list = []
    robot_features_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Video"):
            human_videos = [tensor.to(device) for tensor in batch["human_pixel_values"]]
            robot_videos = [tensor.to(device) for tensor in batch["robot_pixel_values"]]
            all_videos = torch.cat(human_videos + robot_videos, dim=0)
            all_features = model(all_videos)
            human_features, robot_features = torch.chunk(all_features, 2, dim=0)
            human_features_list.append(human_features.cpu())
            robot_features_list.append(robot_features.cpu())
            total_loss += loss_fn(human_features, robot_features).item()
            total_batches += 1

    all_human_features = torch.cat(human_features_list, dim=0)
    all_robot_features = torch.cat(robot_features_list, dim=0)
    similarity_matrix = torch.matmul(all_human_features, all_robot_features.T).numpy()
    metrics = calculate_retrieval_metrics_grouped(similarity_matrix, group_size)
    return {
        "loss": total_loss / max(total_batches, 1),
        "metrics": metrics,
        "similarity_matrix": similarity_matrix,
    }
