"""Train N0-N2 and evaluate cfg2, cfg3, and fixed-gallery human SE(3) stability."""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


GROUPS = ("N0_hand_3d_base", "N1_hand_wrist_relative", "N2_hand_palm_relative")


def _run(command, log_path):
    print("RUN " + " ".join(map(str, command)), flush=True)
    args = [str(value) for value in command]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("wb") as handle:
        process = subprocess.Popen(args, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=0, env=environment)
        try:
            while chunk := os.read(process.stdout.fileno(), 4096):
                handle.write(chunk)
                handle.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, args)


def _read_retrieval(output):
    scores = np.load(output / "similarity_matrix.npy")
    human = np.load(output / "human_embeddings.npy")
    robot = np.load(output / "robot_embeddings.npy")
    np.testing.assert_allclose(scores, human @ robot.T, rtol=2e-5, atol=2e-6)
    metrics = json.loads((output / "metrics.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    for domain in ("human", "robot"):
        ids = list(zip(metadata[f"{domain}_scene_ids"], metadata[f"{domain}_camera_ids"]))
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {domain} scene/camera identity in evaluation")
    return {"queries": len(human), "candidates": len(robot),
            "scene": metrics["human_to_robot"]["scene"],
            "task": metrics["human_to_robot"]["task"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42, help="Training seed; the frozen split is unchanged")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--run-root")
    parser.add_argument("--manifest-dir", default="artifacts/manifests/trajectory_hand_3d_v1")
    parser.add_argument("--dataset-root", default="dataset/RH20T_subset")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(args.run_root or f"artifacts/runs/trajectory/hand_3d_n0_n2_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    manifests = Path(args.manifest_dir)
    required = [manifests / f"{cfg}_valid.json" for cfg in ("cfg2", "cfg3")]
    shared_test = Path("artifacts/manifests/rh20t_shared_v1/cfg2_fixed_test_scenes.json")
    legacy_split = manifests / "cfg2_split.json"
    if not shared_test.exists() and legacy_split.exists():
        shared_test.parent.mkdir(parents=True, exist_ok=True)
        shared_test.write_text(json.dumps({"test": json.loads(legacy_split.read_text())["test"]}, indent=2))
    required.append(shared_test)
    if not all(path.is_file() for path in required):
        _run([sys.executable, "runs/trajectory/prepare_hand_3d.py", "--dataset-root",
              args.dataset_root, "--output-dir", manifests], output / "prepare.log")
    summary = {
        "epochs": args.epochs,
        "seed": args.seed,
        "split_protocol": {
            "fixed_test_fraction": 0.1,
            "val_ratio_of_remaining": 1.0 / 9.0,
            "fixed_test_source": str(shared_test),
        },
        "groups": {},
    }
    reference_split = None
    for name in args.groups:
        config = json.loads(Path(f"configs/trajectory/{name}.json").read_text())
        config["seed"] = args.seed
        config["data_root"] = str(Path(args.dataset_root) / "RH20T_cfg2")
        config["split"] = {
            "fixed_test_manifest_path": str(shared_test),
            "fixed_test_fraction": 0.1,
            "val_ratio_of_remaining": 1.0 / 9.0,
            "unit": "scene",
            "seed": args.seed,
        }
        config["data_manifests"] = {f"RH20T_{cfg}": str(manifests / f"{cfg}_valid.json")
                                    for cfg in ("cfg2", "cfg3")}
        config_path = output / f"{name}_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        run = output / name
        _run([sys.executable, "runs/trajectory/train_trajectory.py", "--config", config_path,
              "--epochs", args.epochs, "--run-dir", run], output / f"{name}_train.log")
        run_split = json.loads((run / "split_manifest.json").read_text())
        if reference_split is None:
            reference_split = run_split
            summary["split_counts"] = {key: len(value) for key, value in run_split.items()}
        elif run_split != reference_split:
            raise AssertionError("N0-N2 did not use the same seeded train/val/test split")
        common = ["--config", run / "params.json", "--checkpoint", run / "best_model.pth"]
        result = {}
        for cfg in ("cfg2", "cfg3"):
            eval_dir = run / "evaluations" / cfg
            robust_dir = run / "human_robustness" / cfg
            extra = (["--data-root", str(Path(args.dataset_root) / "RH20T_cfg3"), "--all-as-test"]
                     if cfg == "cfg3" else ["--split", "test"])
            _run([sys.executable, "runs/trajectory/evaluate_retrieval.py", *common,
                  *extra, "--output-dir", eval_dir], output / f"{name}_{cfg}_eval.log")
            _run([sys.executable, "runs/trajectory/evaluate_human_robustness.py", *common,
                  *extra, "--output-dir", robust_dir], output / f"{name}_{cfg}_robustness.log")
            retrieval = _read_retrieval(eval_dir)
            robustness = json.loads((robust_dir / "summary.json").read_text())
            for variant in robustness["variants"].values():
                if variant["query_count"] != retrieval["queries"] or variant["gallery_count"] != retrieval["candidates"]:
                    raise ValueError("Fixed-gallery evaluation changed sample inventory")
            result[cfg] = {"retrieval": retrieval, "human_robustness": {
                key: {"max_abs_embedding_delta": value["max_abs_embedding_delta"],
                      "top1_same_candidate_rate": value["top1_same_candidate_rate"]}
                for key, value in robustness["variants"].items()}}
            expected = (("translation_x",) if name == "N1_hand_wrist_relative" else
                        tuple(robustness["variants"]) if name == "N2_hand_palm_relative" else ())
            for variant_name in expected:
                if robustness["variants"][variant_name]["max_abs_embedding_delta"] > 1e-4:
                    raise AssertionError(f"{name} failed {cfg} geometric invariance: {variant_name}")
        summary["groups"][name] = {"run_dir": str(run), **result}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"group": name, "cfg2": result["cfg2"]["retrieval"],
                          "cfg3": result["cfg3"]["retrieval"]}), flush=True)
    print(f"Completed N0-N2 closed loop: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
