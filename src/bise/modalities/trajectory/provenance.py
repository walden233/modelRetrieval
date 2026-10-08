"""Keep evaluation inputs and checkpoint identity alongside metrics."""

import hashlib
import json
from pathlib import Path


def save_evaluation_provenance(output_dir: Path, config: dict, checkpoint: str, manifest: dict):
    info_path = output_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update({"checkpoint": str(Path(checkpoint).resolve()),
                 "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                 "config": config, "data_manifest_sha256": hashlib.sha256(
                     json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                 "unique_valid_scenes": len(manifest["scenes"]),
                 "candidate_unit": "camera_sequence", "metric_R_at_K": "multi_positive_hit_at_K"})
    info_path.write_text(json.dumps(info, indent=2))
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))


def validate_checkpoint_config(config: dict, checkpoint: str) -> None:
    path = Path(checkpoint).resolve().parent / "params.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text())
    for key in ("model_params", "data_params", "use_6_keypoints", "reference_frames",
                "geometry_hashes"):
        if saved.get(key) != config.get(key):
            raise ValueError(f"Evaluation {key} differs from checkpoint training config: {path}")
