"""Honor an explicit experiment device instead of silently changing its budget."""

import torch


def experiment_device(config: dict) -> torch.device:
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run in a GPU-enabled torch2 session; "
                           "use device=cpu only for an explicit CPU verification batch.")
    return torch.device(device)
