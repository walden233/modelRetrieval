from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 16,
    strategy: str = "uniform",
    seed: int | None = None,
    stride: int | None = None,
):
    # 真正的视频读取入口：先解码，再按策略选帧，最后返回帧列表。
    try:
        video_reader = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(video_reader)
        if total_frames <= 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        indices = select_frame_indices(
            total_frames=total_frames,
            num_frames=num_frames,
            strategy=strategy,
            seed=seed,
            stride=stride,
        )
        return list(video_reader.get_batch(indices).asnumpy())
    except Exception as exc:
        print(f"Error reading video file {video_path}: {exc}")
        return None


def select_frame_indices(
    total_frames: int,
    num_frames: int,
    strategy: str = "uniform",
    seed: int | None = None,
    stride: int | None = None,
):
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    rng = np.random.default_rng(seed)
    strategy = str(strategy).strip().lower()

    if total_frames == 1:
        # 单帧视频无法采样，直接重复该帧补齐长度。
        return np.zeros(num_frames, dtype=int)

    if strategy == "uniform":
        # 最常见的评估策略：整段视频均匀覆盖。
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    if strategy == "center_clip":
        # 取中间连续片段，适合一些只关心中段动作的设置。
        if total_frames <= num_frames:
            return np.linspace(0, total_frames - 1, num_frames, dtype=int)
        start = max((total_frames - num_frames) // 2, 0)
        return np.arange(start, start + num_frames, dtype=int)

    if strategy == "stride":
        # 固定步长采样，偏向保持时间顺序的一致间隔。
        effective_stride = stride or max(total_frames // num_frames, 1)
        indices = np.arange(0, total_frames, effective_stride, dtype=int)
        if len(indices) >= num_frames:
            return indices[:num_frames]
        padded = np.pad(indices, (0, num_frames - len(indices)), mode="edge")
        return padded.astype(int)

    if strategy == "segment_random":
        # 训练时常用：把整段视频切成 num_frames 个时间段，每段随机抽一帧。
        # 这样既覆盖全局时间范围，又引入轻量时序扰动。
        segment_edges = np.linspace(0, total_frames, num_frames + 1, dtype=int)
        indices = []
        for start, end in zip(segment_edges[:-1], segment_edges[1:]):
            upper = max(end - 1, start)
            if upper <= start:
                indices.append(start)
            else:
                indices.append(int(rng.integers(start, upper + 1)))
        return np.asarray(indices, dtype=int)

    raise ValueError(f"Unsupported frame sampling strategy: {strategy}")
