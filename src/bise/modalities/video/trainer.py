import torch
from tqdm.auto import tqdm


def train_video_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Video"):
        human_videos = batch["human_pixel_values"].to(device)
        robot_videos = batch["robot_pixel_values"].to(device)
        all_videos = torch.cat([human_videos, robot_videos], dim=0)
        all_features = model(all_videos)
        human_features, robot_features = torch.chunk(all_features, 2, dim=0)
        loss = loss_fn(human_features, robot_features)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)
