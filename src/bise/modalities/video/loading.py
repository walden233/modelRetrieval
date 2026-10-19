"""CPU video loading with bounded prefetch and CUDA-safe worker startup."""

from .batch import collate_video_pairs


def video_loader_options(training: dict, device) -> dict:
    workers = int(training.get("num_workers", 0))
    options = {
        "num_workers": workers,
        "collate_fn": collate_video_pairs,
        "pin_memory": device.type == "cuda",
    }
    if workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=int(training.get("prefetch_factor", 2)),
            multiprocessing_context="spawn",
        )
    return options
