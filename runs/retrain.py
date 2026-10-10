"""Run the new RH20T batch, or one independent preparation/training/evaluation stage."""

import argparse
import copy
import datetime
import json
import os
import sys
from pathlib import Path

from _bootstrap import bootstrap
bootstrap()

from bise.common.fingerprints import write_json
from bise.common.paths import project_root, resolve_path
from bise.common.process import run_logged
from bise.data.rh20t.splits import PROTOCOL


def epoch_budget(config: dict, modality: str, budget: int | None) -> dict:
    result = copy.deepcopy(config)
    if budget is None:
        return result
    if budget < 1:
        raise ValueError("Epoch budget must be positive")
    if modality == "video":
        result["training"]["num_epochs"] = budget
    elif result["mode"] == "two_stage":
        if budget < 2:
            raise ValueError("T3 needs at least two total epochs")
        total = result["pretrain_epochs"] + result["finetune_epochs"]
        pretrain = max(1, min(budget - 1, round(budget * result["pretrain_epochs"] / total)))
        result["pretrain_epochs"], result["finetune_epochs"] = pretrain, budget - pretrain
    else:
        result["num_epochs"] = budget
    return result


def prepare_batch(config: dict, root: Path, modalities: list[str], seed: int,
                  budgets: dict, dry_run: bool = False,
                  device: str | None = None) -> tuple[dict, dict]:
    registries, runtime = {}, copy.deepcopy(config)
    manifest_dir = resolve_path(config["manifest_dir"]) / root.name
    runtime["manifest_dir"] = str(manifest_dir)
    runtime["eval_manifests"] = {
        cfg: str(manifest_dir / cfg / "metadata.json") for cfg in ("cfg2", "cfg3")
    }
    for modality in modalities:
        groups = {}
        for key, entry in config[modality].items():
            source = resolve_path(entry["config"])
            current = epoch_budget(json.loads(source.read_text()), modality, budgets[modality])
            current["seed"] = seed
            current["eval_manifests"] = {
                cfg: str(manifest_dir / cfg / f"split_manifest_{modality}.json")
                for cfg in ("cfg2", "cfg3")
            }
            if modality == "trajectory":
                current["data_manifests"] = {
                    f"RH20T_{cfg}": str(manifest_dir / cfg / "data_manifest.json")
                    for cfg in ("cfg2", "cfg3")
                }
                if (current["model_params"].get("grasp_geometry")
                        or current["model_params"].get("use_gripper")):
                    current["data_params"]["geometry_manifests"] = {
                        f"RH20T_{cfg}": str(resolve_path(config["dataset_root"])
                                           / f"RH20T_{cfg}" / "geometry.json")
                        for cfg in ("cfg2", "cfg3")
                    }
            else:
                current["dataset"]["pair_manifests"] = {
                    f"RH20T_{cfg}": path for cfg, path in runtime["eval_manifests"].items()
                }
            if current["split"].get("shared_manifest_path"):
                current["split"]["shared_manifest_path"] = current["eval_manifests"]["cfg2"]
            if device:
                current["device"] = device
            current["split"].setdefault("seed", seed)
            split_manifest = root / modality / key / "split_manifest.json"
            if split_manifest.is_file():
                current["split"]["manifest_path"] = str(split_manifest)
            destination = root / modality / "configs" / f"{key}.json"
            if not dry_run:
                write_json(destination, current)
            groups[key] = {**entry, "config": str(destination),
                           "run_dir": str(root / modality / key)}
            runtime[modality][key] = {**entry, "config": str(destination)}
        registry = {"protocol": PROTOCOL, "batch": root.name,
                    "modality": modality, "groups": groups}
        if not dry_run:
            path = root / modality / "runs.json"
            write_json(path, registry)
        registries[modality] = registry
    return registries, runtime


def archive_run(root: Path, modality: str, key: str, backup: Path) -> None:
    """Move the selected run and its stale comparisons outside the active batch."""
    run = root / modality / key
    if not run.exists():
        return
    destination = backup / modality
    destination.mkdir(parents=True, exist_ok=True)
    run.rename(destination / key)
    for name in ("charts", "summary.json"):
        current = root / modality / name
        if current.exists():
            current.rename(destination / name)
    for log in (root / modality / "logs").glob(f"{key}_*.log"):
        target = destination / "logs" / log.name
        target.parent.mkdir(parents=True, exist_ok=True)
        log.rename(target)
    manifest = root / "plots.json"
    if manifest.is_file():
        plots = json.loads(manifest.read_text())
        prefixes = (f"{modality}/{key}/", f"{modality}/charts/")
        write_json(manifest, {path: entry for path, entry in plots.items()
                              if not path.startswith(prefixes)})
    print(f"Archive {modality}/{key}: {destination / key}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain/run.json")
    parser.add_argument("--batch", help="Short batch name, for example s42 or smoke")
    parser.add_argument("--stage", default="all",
                        choices=("all", "prepare", "audit", "train", "eval", "charts", "robust"))
    parser.add_argument("--modality", nargs="+", choices=("trajectory", "video"),
                        default=["trajectory", "video"])
    parser.add_argument("--only", nargs="+", help="Selected training/evaluation groups")
    parser.add_argument("--seed", type=int,
                        help="Training seed; the shared protocol seed stays fixed")
    parser.add_argument("--traj-epochs", type=int)
    parser.add_argument("--video-epochs", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="Explicit execution device override")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse completed training; selected evaluations are always recomputed")
    parser.add_argument("--replace", action="store_true",
                        help="Archive and retrain --only groups in this batch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without creating any files")
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Expected configs/retrain/run.json protocol")
    os.chdir(project_root())
    seed = config["seed"] if args.seed is None else args.seed
    batch = args.batch or (config["batch"] if args.seed is None else f"s{seed}")
    if Path(batch).name != batch or batch in (".", ".."):
        raise ValueError("--batch must be a single directory name")
    root = resolve_path(config["output_root"]) / batch
    modalities = list(dict.fromkeys(args.modality))
    known = {key for modality in modalities for key in config[modality]}
    if args.only and set(args.only) - known:
        raise ValueError(f"Unknown selected groups: {sorted(set(args.only) - known)}")
    if args.replace:
        if not args.only or args.stage not in ("all", "train"):
            parser.error("--replace requires --only and --stage all/train")
        if args.skip_existing:
            parser.error("--replace and --skip-existing cannot be combined")
        raw = [key for modality in modalities for key, entry in config[modality].items()
               if key in args.only and entry.get("raw")]
        if raw:
            parser.error(f"Raw baselines have no training stage: {raw}")
    stages = ["prepare", "audit", "train", "eval"] if args.stage == "all" else [args.stage]
    if args.dry_run:
        for stage in stages:
            print(f"{stage}: batch={batch}, modalities={modalities}, groups={args.only or 'all'}")
        for modality in modalities:
            for key, entry in config[modality].items():
                if args.only and key not in args.only:
                    continue
                print(f"{modality}/{key}: {entry['config']} -> {root / modality / key}")
        if args.replace:
            print(f"Replace selected runs; archive under {root.parent / 'archive' / batch}")
        return
    budgets = {"trajectory": args.traj_epochs, "video": args.video_epochs}
    registries, runtime = prepare_batch(config, root, modalities, seed, budgets, device=args.device)
    backup = (root.parent / "archive" / batch
              / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    pipeline_path = root / "pipeline.json"
    history = (json.loads(pipeline_path.read_text()) if pipeline_path.exists()
               else {"protocol": PROTOCOL, "batch": batch, "steps": []})
    for stage in stages:
        record = {"stage": stage, "state": "running"}
        history["steps"].append(record)
        write_json(pipeline_path, history)
        try:
            if stage == "prepare":
                from bise.data.rh20t.splits import prepare_protocols
                prepare_protocols(config["dataset_root"], config.get("source_dir"),
                                  runtime["manifest_dir"], config["seed"])
                if "trajectory" in modalities:
                    selected = [entry for key, entry in registries["trajectory"]["groups"].items()
                                if not args.only or key in args.only]
                    for entry in selected:
                        current = json.loads(Path(entry["config"]).read_text())
                        if (current["model_params"].get("grasp_geometry")
                                or current["model_params"].get("use_gripper")):
                            from bise.data.rh20t.grasp_geometry import validate_dataset_geometry
                            validate_dataset_geometry(current)
                            break
            elif stage == "audit":
                from audit_splits import audit
                audit_config = {**runtime}
                for modality in ("trajectory", "video"):
                    if modality not in modalities:
                        audit_config[modality] = {}
                    elif args.only:
                        audit_config[modality] = {
                            key: entry for key, entry in runtime[modality].items()
                            if key in args.only
                        }
                report = audit(audit_config, files=True)
                write_json(root / "audit.json", report)
            elif stage == "train":
                from bise.common.device import experiment_device
                from bise.common.training import prepare_training_run
                for registry in registries.values():
                    for key, entry in registry["groups"].items():
                        if not entry.get("raw") and (not args.only or key in args.only):
                            experiment_device(json.loads(resolve_path(entry["config"]).read_text()))
                training = []
                for modality, registry in registries.items():
                    for key, entry in registry["groups"].items():
                        if args.only and key not in args.only:
                            continue
                        run = resolve_path(entry["run_dir"])
                        split_manifest = run / "split_manifest.json"
                        if entry.get("raw"):
                            metadata = ("params.json", "split_manifest.json", "data_manifest.json",
                                        "split_info.json", "protocols.json", "status.json")
                            if not all((run / name).is_file() for name in metadata):
                                source = (run / "params.json" if (run / "params.json").is_file()
                                          else resolve_path(entry["config"]))
                                current = json.loads(source.read_text())
                                current["raw_backbone"] = True
                                prepare_training_run(
                                    current, modality, run, split_manifest)
                            continue
                        if run.exists():
                            status_path = run / "status.json"
                            complete = (status_path.is_file()
                                        and json.loads(status_path.read_text()).get("state")
                                        == "complete")
                            if (args.skip_existing and complete
                                    and (run / "best_model.pth").is_file()
                                    and (run / "params.json").is_file()):
                                print(f"Reuse completed training {modality}/{key}", flush=True)
                                continue
                            archive_run(root, modality, key, backup)
                            split_manifest = backup / modality / key / "split_manifest.json"
                            record["archive"] = str(backup)
                            write_json(pipeline_path, history)
                        prepare_training_run(
                            json.loads(resolve_path(entry["config"]).read_text()),
                            modality, run, split_manifest)
                        print(f"Prepared {modality}/{key}: {run}", flush=True)
                        training.append((modality, key, run))
                print(f"Training metadata ready; start {len(training)} runs", flush=True)
                for modality, key, run in training:
                    script = f"runs/{modality}/train_{modality}.py"
                    command = [sys.executable, script, "--config", run / "params.json",
                               "--run-dir", run, "--prepared"]
                    run_logged(command,
                               root / modality / "logs" / f"{key}_train.log")
            elif stage in ("eval", "charts"):
                from compare import compare
                for modality, registry in registries.items():
                    only = ([key for key in args.only if key in registry["groups"]]
                            if args.only else None)
                    if args.only and not only:
                        continue
                    compare(root / modality / "runs.json", only=only,
                            charts_only=stage == "charts", skip_existing=args.skip_existing)
            elif stage == "robust":
                from bise.common.comparison import eval_manifest
                from bise.data.rh20t.splits import load_protocol
                if "trajectory" not in registries:
                    raise ValueError("Human robustness requires trajectory modality")
                for key, entry in registries["trajectory"]["groups"].items():
                    if not key.startswith("N") or (args.only and key not in args.only):
                        continue
                    run = resolve_path(entry["run_dir"])
                    params = json.loads((run / "params.json").read_text())
                    for cfg in ("cfg2", "cfg3"):
                        root_dir = load_protocol(params["eval_manifests"][cfg])["data_root"]
                        run_logged([sys.executable, "runs/trajectory/evaluate_human_robustness.py",
                                    "--config", run / "params.json",
                                    "--checkpoint", run / "best_model.pth",
                                    "--data-root", root_dir,
                                    "--split-manifest", eval_manifest(params, run, cfg),
                                    "--output-dir", run / "human_robustness" / cfg],
                                   root / "trajectory" / "logs" / f"{key}_{cfg}_robust.log")
            record["state"] = "complete"
        except BaseException as exc:
            record.update(state="failed", error=str(exc))
            write_json(pipeline_path, history)
            raise
        write_json(pipeline_path, history)
    print(f"Batch: {root}", flush=True)


if __name__ == "__main__":
    main()
