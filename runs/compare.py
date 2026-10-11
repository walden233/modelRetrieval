"""Re-evaluate selected retrain groups, then merge saved results into charts."""

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap
bootstrap()

from bise.common.comparison import (
    eval_manifest, export_comparison, export_curves, load_runs, prepare_raw_split,
)
from bise.common.evaluation import ARTIFACTS, effective_config
from bise.common.fingerprints import write_json
from bise.common.paths import resolve_path
from bise.common.process import run_logged


def compare(runs_path: str | Path, only: list[str] | None = None,
            cfgs: list[str] | None = None, charts_only: bool = False,
            skip_existing: bool = False) -> dict:
    registry = load_runs(runs_path)
    modality = registry["modality"]
    cfgs = cfgs or ["cfg2", "cfg3"]
    selected = only if only is not None else list(registry["groups"])
    unknown = set(selected) - set(registry["groups"])
    if unknown:
        raise ValueError(f"Unknown {modality} groups: {sorted(unknown)}")
    root = resolve_path(runs_path).parent
    if not charts_only:
        for key in selected:
            entry = registry["groups"][key]
            run = resolve_path(entry["run_dir"])
            params = run / "params.json"
            source = params if params.is_file() else resolve_path(entry["config"])
            if not params.is_file() and not entry.get("raw"):
                raise FileNotFoundError(f"Train {key} first: {params}")
            if entry.get("raw"):
                prepare_raw_split(json.loads(source.read_text()), run)
            for cfg in cfgs:
                config = effective_config(json.loads(source.read_text()), modality, cfg)
                manifest = eval_manifest(config, run, cfg)
                output = run / "evaluations" / cfg
                print(f"Evaluate {modality} {key} {cfg}: {output}", flush=True)
                data_root = (config["data_root"] if modality == "trajectory"
                             else config["dataset"]["root_dir"])
                common = ["--config", source, "--data-root", data_root,
                          "--split-manifest", manifest, "--split", "test"]
                if entry.get("raw"):
                    command = [sys.executable, "runs/video/evaluate_raw_video_backbone.py", *common,
                               "--output-dir", run, "--final-subdir", f"evaluations/{cfg}"]
                else:
                    checkpoint = run / "best_model.pth"
                    script = ("runs/trajectory/evaluate_retrieval.py" if modality == "trajectory"
                              else "runs/video/evaluate_video.py")
                    command = [sys.executable, script, *common, "--checkpoint", checkpoint,
                               "--output-dir", output]
                run_logged(command, root / "logs" / f"{key}_{cfg}.log")
                missing = [name for name in ARTIFACTS if not (output / name).is_file()]
                if missing:
                    raise RuntimeError(
                        f"Selected evaluation {modality} {key} {cfg} did not produce "
                        f"{', '.join(missing)} in {output}; "
                        f"see {root / 'logs' / f'{key}_{cfg}.log'}"
                    )
                print(f"Evaluated {modality} {key} {cfg}", flush=True)
                source = params
    summary = export_comparison(registry, cfgs, root / "charts")
    export_curves(registry, root / "charts")
    summary_path = root / "summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    previous.update({"protocol": registry["protocol"], "modality": modality,
                     "batch": registry["batch"], "runs": str(resolve_path(runs_path))})
    previous.setdefault("evaluations", {}).update(summary)
    write_json(summary_path, previous)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="New batch trajectory/video runs.json")
    parser.add_argument("--only", nargs="+",
                        help="Re-evaluate these groups; charts merge all available saved results")
    parser.add_argument("--cfg", nargs="+", choices=("cfg2", "cfg3"))
    parser.add_argument("--charts-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Compatibility option; selected evaluations are always overwritten")
    args = parser.parse_args()
    compare(args.runs, args.only, args.cfg, args.charts_only, args.skip_existing)


if __name__ == "__main__":
    main()
