"""Validate and fingerprint independent retrain evaluations without loading models."""

import json
from pathlib import Path

import numpy as np

from bise.common.fingerprints import file_hash, object_hash, write_json
from bise.common.paths import project_root, resolve_path
from bise.data.rh20t.splits import (
    PROTOCOL, load_protocol, load_split_map, protocol_hash, scene_id, validate_split,
)


ARTIFACTS = ("metrics.json", "metadata.json", "cases.json", "similarity_matrix.npy",
             "human_embeddings.npy", "robot_embeddings.npy", "data_manifest.json")


def resolve_eval_split(config: dict, checkpoint: str, explicit: str | None,
                       all_as_test: bool) -> tuple[dict, Path | None]:
    if all_as_test and explicit:
        raise ValueError("--all-as-test cannot be combined with --split-manifest")
    if explicit:
        path = resolve_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {path}")
        return {"manifest_path": str(path)}, path
    split = dict(config.get("split") or {})
    if all_as_test or split.get("all_as_test") or split.get("unit") == "all_test":
        return {"all_as_test": True}, None
    path = Path(checkpoint).resolve().parent / "split_manifest.json"
    if path.is_file():
        return {"manifest_path": str(path)}, path
    if config.get("protocol") == PROTOCOL:
        raise FileNotFoundError(f"A retrain evaluation requires a frozen split: {path}")
    return split, None


def bind_protocols(run: Path, config: dict) -> None:
    if config.get("protocol") != PROTOCOL:
        return
    sources = {cfg: {"path": str(resolve_path(path)), "sha256": protocol_hash(path)}
               for cfg, path in config["eval_manifests"].items()}
    path = run / "protocols.json"
    if path.exists() and json.loads(path.read_text()) != sources:
        raise ValueError(f"Run is bound to a different frozen protocol: {run}")
    write_json(path, sources)


def evaluator_hash(modality: str) -> str:
    root = project_root()
    dataset = "hand_3d_trajectory_dataset" if modality == "trajectory" else "video_dataset"
    paths = [root / "src/bise/retrieval/metrics.py",
             root / f"src/bise/modalities/{modality}/evaluator.py",
             root / f"src/bise/data/rh20t/{dataset}.py",
             root / "src/bise/common/evaluation.py"]
    paths += [root / "src/bise/data/rh20t/splits.py",
              root / f"src/bise/modalities/{modality}/factory.py"]
    paths += sorted((root / f"src/bise/modalities/{modality}/models").glob("*.py"))
    if modality == "trajectory":
        paths += [root / "src/bise/data/rh20t/human_representation.py",
                  root / "src/bise/data/rh20t/grasp_geometry.py",
                  root / "src/bise/modalities/trajectory/grasp.py",
                  root / "src/bise/modalities/trajectory/geometry.py",
                  root / "src/bise/data/rh20t/collate.py",
                  root / "runs/trajectory/evaluate_retrieval.py"]
    else:
        paths += [root / "src/bise/modalities/video/frame_sampling.py",
                  root / "src/bise/modalities/video/transforms.py",
                  root / "runs/video/evaluate_video.py",
                  root / "runs/video/evaluate_raw_video_backbone.py"]
    return object_hash({str(p.relative_to(root)): file_hash(p) for p in paths})


def effective_config(config: dict, modality: str, cfg: str) -> dict:
    result = json.loads(json.dumps(config))
    payload = load_protocol(config["eval_manifests"][cfg])
    if modality == "trajectory":
        result["data_root"] = payload["data_root"]
    else:
        result["dataset"]["root_dir"] = payload["data_root"]
    return result


def evaluation_inputs(config: dict, checkpoint: str | Path, modality: str,
                      manifest_path: str | Path, split: str = "test", *,
                      run_dir: str | Path | None = None) -> dict:
    checkpoint = Path(checkpoint).resolve()
    run = resolve_path(run_dir) if run_dir is not None else checkpoint.parent
    root = resolve_path(config["data_root"] if modality == "trajectory"
                        else config["dataset"]["root_dir"])
    cfg = root.name.removeprefix("RH20T_")
    protocol_path = resolve_path(config["eval_manifests"][cfg])
    payload = load_protocol(protocol_path, root, video_hashes=modality == "video")
    mapping = load_split_map(manifest_path)
    selected = mapping.get(split, [])
    if not selected:
        raise ValueError(f"Empty evaluation split: {manifest_path}:{split}")
    validate_split(mapping)
    saved = json.loads((run / "params.json").read_text())
    for key in (("model_params", "data_params", "use_6_keypoints", "reference_frames",
                 "geometry_hashes")
                if modality == "trajectory" else ("model", "dataset")):
        current, original = config.get(key), saved.get(key)
        if key == "dataset":
            current, original = dict(current), dict(original)
            current.pop("root_dir", None)
            original.pop("root_dir", None)
        if current != original:
            raise ValueError(f"Evaluation {key} differs from saved training configuration")
    if not config.get("raw_backbone"):
        training = load_split_map(run / "split_manifest.json")
        validate_split(training, saved["split"].get("unit", "scene"))
        saved_root = resolve_path(saved["data_root"] if modality == "trajectory"
                                  else saved["dataset"]["root_dir"])
    if modality == "trajectory":
        snapshot = run / "data_manifest.json"
        data = json.loads(snapshot.read_text()) if snapshot.is_file() else None
        if data is None or resolve_path(data["data_root"]) != root:
            data = json.loads(resolve_path(payload["data_manifest"]).read_text())
        canonical = json.loads(resolve_path(payload["data_manifest"]).read_text())
        if object_hash(data) != object_hash(canonical):
            raise ValueError("Run trajectory inventory differs from the bound evaluation protocol")
        entries = {x["scene_id"]: x for x in data["scenes"]}
        identities = []
        for identity in selected:
            identities.extend([[identity, camera]
                               for camera in sorted(entries[identity]["cameras"])])
    else:
        data = {"protocol": payload["protocol"], "data_root": str(root),
                "pair_manifest": str(protocol_path),
                "pair_manifest_sha256": file_hash(protocol_path),
                "video_pairs": payload["video_pairs"]}
        by_id = {pair["sample_id"]: [identity, pair["camera_id"]]
                 for identity, pairs in payload["video_pairs"].items() for pair in pairs}
        identities = []
        for identity in selected:
            if identity in payload["video_pairs"]:
                identities.extend([[identity, pair["camera_id"]]
                                   for pair in payload["video_pairs"][identity]])
            else:
                identities.append(by_id[identity])
    if len(identities) != len({tuple(x) for x in identities}):
        raise ValueError("Duplicate evaluation scene/camera identities")
    source = config.get("data_params", {}) if modality == "trajectory" else config["dataset"]
    keys = (("max_human_frames", "max_robot_frames", "preserve_timing") if modality == "trajectory"
            else ("num_frames", "sampling_strategy", "sampling_stride", "eval_augmentations"))
    sampling = {key: source.get(key) for key in keys}
    geometry = {}
    if modality == "trajectory" and (config["model_params"].get("grasp_geometry")
            or config["model_params"].get("use_gripper")):
        from bise.data.rh20t.grasp_geometry import geometry_digest
        geometry["geometry_sha256"] = geometry_digest(config, selected)
    return {"cfg": cfg, "split": split, "modality": modality, "protocol": PROTOCOL,
            "checkpoint_sha256": file_hash(checkpoint), "config_sha256": object_hash(config),
            "protocol_sha256": protocol_hash(protocol_path),
            "split_sha256": object_hash(mapping), "data_sha256": object_hash(data),
            "evaluator_sha256": evaluator_hash(modality), "sampling": sampling,
            "identities": identities,
            "scenes": len({x[0] for x in identities}),
            "tasks": len({x[0].split('/')[0] for x in identities}),
            "data_manifest": data, **geometry}


def metadata_identities(metadata: dict, modality: str, domain: str) -> list:
    if modality == "trajectory":
        return [list(x) for x in zip(metadata[f"{domain}_scene_ids"],
                                     metadata[f"{domain}_camera_ids"])]
    return [list(x) for x in zip(metadata["scene_ids"], metadata["camera_ids"])]


def check_artifacts(output: Path, inputs: dict) -> dict:
    for name in ARTIFACTS:
        if not (output / name).is_file():
            raise FileNotFoundError(f"Incomplete evaluation: {output / name}")
    metadata = json.loads((output / "metadata.json").read_text())
    for domain in ("human", "robot"):
        if metadata_identities(metadata, inputs["modality"], domain) != inputs["identities"]:
            raise ValueError(
                f"Evaluation {domain} inventory/order differs from frozen protocol: {output}"
            )
    human = np.load(output / "human_embeddings.npy")
    robot = np.load(output / "robot_embeddings.npy")
    matrix = np.load(output / "similarity_matrix.npy")
    count = len(inputs["identities"])
    if human.ndim != 2 or robot.ndim != 2 or human.shape[0] != count or robot.shape[0] != count:
        raise ValueError(f"Embedding counts differ from frozen inventory: {output}")
    if not all(np.isfinite(x).all() for x in (human, robot, matrix)):
        raise ValueError(f"Non-finite evaluation arrays: {output}")
    if matrix.shape != (count, count) or human.shape[1] != robot.shape[1]:
        raise ValueError(f"Evaluation matrix dimensions differ from frozen inventory: {output}")
    np.testing.assert_allclose(matrix, human @ robot.T, rtol=2e-5, atol=2e-6)
    data = json.loads((output / "data_manifest.json").read_text())
    if object_hash(data) != inputs["data_sha256"]:
        raise ValueError(f"Evaluation data inventory changed: {output}")
    return {name: file_hash(output / name) for name in ARTIFACTS}


def stamp_evaluation(output: Path, config: dict, checkpoint: str | Path,
                      modality: str, manifest_path: str | Path, split: str = "test", *,
                      run_dir: str | Path | None = None) -> None:
    inputs = evaluation_inputs(config, checkpoint, modality, manifest_path, split, run_dir=run_dir)
    write_json(output / "data_manifest.json", inputs.pop("data_manifest"))
    artifacts = check_artifacts(output, inputs)
    path = output / "run_info.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    info["evaluation"] = {**inputs, "artifacts": artifacts}
    write_json(path, info)


def validate_evaluation(output: Path, config: dict, checkpoint: str | Path,
                         modality: str, manifest_path: str | Path, *,
                         run_dir: str | Path | None = None) -> dict:
    inputs = evaluation_inputs(config, checkpoint, modality, manifest_path, run_dir=run_dir)
    inputs.pop("data_manifest")
    info = json.loads((output / "run_info.json").read_text()).get("evaluation")
    if info is None or any(info.get(key) != value for key, value in inputs.items()):
        raise ValueError(f"Stale evaluation; explicitly re-evaluate this run: {output}")
    if check_artifacts(output, inputs) != info["artifacts"]:
        raise ValueError(f"Evaluation artifacts changed: {output}")
    return info
