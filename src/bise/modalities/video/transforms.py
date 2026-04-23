from typing import Any, Dict

import torch


def apply_video_transforms(
    pixel_values: torch.Tensor,
    config: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    # 这里的增强刻意保持轻量，目标是增强鲁棒性，而不是大幅破坏动作语义。
    if not config:
        return pixel_values

    transformed, layout = _to_tchw(pixel_values)
    generator = torch.Generator(device=transformed.device)
    if seed is not None:
        generator.manual_seed(seed)

    brightness_jitter = float(config.get("brightness_jitter", 0.0))
    if brightness_jitter > 0:
        # 对整段视频做统一亮度缩放，避免不同帧亮度变化破坏时序一致性。
        scale = torch.empty(1, device=transformed.device).uniform_(
            1.0 - brightness_jitter,
            1.0 + brightness_jitter,
            generator=generator,
        )
        transformed = transformed * scale

    noise_std = float(config.get("noise_std", 0.0))
    if noise_std > 0:
        transformed = transformed + torch.randn_like(transformed, generator=generator) * noise_std

    temporal_roll_max = int(config.get("temporal_roll_max", 0))
    if temporal_roll_max > 0 and transformed.shape[0] > 1:
        # 轻微时序平移，模拟起始点不完全对齐的情况。
        shift = int(
            torch.randint(
                -temporal_roll_max,
                temporal_roll_max + 1,
                (1,),
                generator=generator,
                device=transformed.device,
            ).item()
        )
        if shift != 0:
            transformed = torch.roll(transformed, shifts=shift, dims=0)

    if bool(config.get("clamp", True)):
        transformed = transformed.clamp(-3.0, 3.0)

    return _restore_layout(transformed, layout)


def _to_tchw(pixel_values: torch.Tensor):
    # 有些处理器输出是 [T, C, H, W]，有些是 [C, T, H, W]，这里统一成 TCHW 再做增强。
    if pixel_values.ndim != 4:
        raise ValueError("Video transforms expect a 4D tensor.")
    if pixel_values.shape[0] == 3 and pixel_values.shape[1] != 3:
        return pixel_values.permute(1, 0, 2, 3), "cthw"
    return pixel_values, "tchw"


def _restore_layout(pixel_values: torch.Tensor, layout: str):
    if layout == "cthw":
        return pixel_values.permute(1, 0, 2, 3)
    return pixel_values
