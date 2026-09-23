"""Train and evaluate the four trajectory ablations with an auditable summary."""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

GROUPS = ["F0_valid_raw7", "F1_rot6d", "F2_relative", "F3_relative_time"]


def run_logged(command, log_path):
    print("RUN " + " ".join(map(str, command)), flush=True)
    with log_path.open("w") as log:
        subprocess.run([str(value) for value in command], stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def check_evaluation(path):
    metadata = json.loads((path / "metadata.json").read_text())
    matrix = np.load(path / "similarity_matrix.npy")
    human = np.load(path / "human_embeddings.npy")
    robot = np.load(path / "robot_embeddings.npy")
    np.testing.assert_allclose(matrix, human @ robot.T, rtol=2e-5, atol=2e-6)
    for domain in ("human", "robot"):
        identities = list(zip(metadata[f"{domain}_scene_ids"], metadata[f"{domain}_camera_ids"]))
        if len(set(identities)) != len(identities):
            raise AssertionError("Evaluation contains duplicate scene/camera identities")
    q, g = np.array(metadata["human_task_ids"]), np.array(metadata["robot_task_ids"])
    ranks = np.argmax(g[np.argsort(-matrix, axis=1)] == q[:, None], axis=1) + 1
    metrics = json.loads((path / "metrics.json").read_text())["human_to_robot"]["task"]
    np.testing.assert_allclose(metrics["R@1"], np.mean(ranks == 1))
    np.testing.assert_allclose(metrics["MRR"], np.mean(1 / ranks))
    return {"task_hit1": metrics["R@1"], "task_mrr": metrics["MRR"],
            "queries": matrix.shape[0], "candidates": matrix.shape[1],
            "unique_scenes": len(set(metadata["human_scene_ids"])),
            "artifact_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--run-root")
    parser.add_argument("--manifest-dir", default="artifacts/manifests/trajectory_first_batch")
    parser.add_argument("--dataset-root", default="dataset/RH20T_subset")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    out = Path(args.run_root or f"artifacts/runs/trajectory/first_batch_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=False)
    manifests = Path(args.manifest_dir)
    required = [manifests / name for name in ("cfg2_valid.json", "cfg3_valid.json", "cfg2_split.json")]
    if not all(p.exists() for p in required):
        run_logged([sys.executable, "runs/trajectory/prepare_first_batch.py", "--dataset-root",
                    args.dataset_root, "--output-dir", manifests], out / "prepare.log")
    summary = {"epochs": args.epochs, "purpose": "short_run_validation" if args.epochs < 60 else "training",
               "groups": {}}
    for group in args.groups:
        config = json.loads(Path(f"configs/trajectory/{group}.json").read_text())
        config["data_root"] = str(Path(args.dataset_root) / "RH20T_cfg2")
        config["split"] = {"manifest_path": str(manifests / "cfg2_split.json")}
        config["data_manifests"] = {f"RH20T_{cfg}": str(manifests / f"{cfg}_valid.json") for cfg in ("cfg2", "cfg3")}
        config_path = out / f"{group}_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        run = out / group
        run_logged([sys.executable, "runs/trajectory/train_trajectory.py", "--config", config_path,
                    "--epochs", args.epochs, "--run-dir", run], out / f"{group}_train.log")
        common = ["--config", run / "params.json", "--checkpoint", run / "best_model.pth"]
        results = {}
        for cfg in ("cfg2", "cfg3"):
            evaluation = run / "evaluations" / cfg
            command = [sys.executable, "runs/trajectory/evaluate_retrieval.py", *common,
                       "--split", "test", "--output-dir", evaluation]
            if cfg == "cfg3":
                command += ["--data-root", str(Path(args.dataset_root) / "RH20T_cfg3"), "--all-as-test"]
            run_logged(command, out / f"{group}_{cfg}_eval.log")
            results[cfg] = check_evaluation(evaluation)
        probe = run / "robustness"
        run_logged([sys.executable, "runs/trajectory/evaluate_robustness.py", *common,
                    "--output-dir", probe], out / f"{group}_robustness.log")
        results["robustness"] = json.loads((probe / "robustness.json").read_text())
        if group != "F0_valid_raw7":
            if results["robustness"]["negate_q"]["embedding_max_abs_delta"] > 1e-5:
                raise AssertionError("Quaternion sign invariance failed")
        if group in {"F2_relative", "F3_relative_time"}:
            for name in ("translate_x_0.1m", "rotate_z_90deg"):
                if results["robustness"][name]["embedding_max_abs_delta"] > 1e-4:
                    raise AssertionError(f"Relative pose invariance failed: {name}")
        summary["groups"][group] = {"run_dir": str(run), **results}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"group": group, "cfg2": results["cfg2"], "cfg3": results["cfg3"]}), flush=True)
    print(f"Completed closed loop: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
