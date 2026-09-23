"""Probe equivalent quaternion signs and rigid coordinate changes on robot inputs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

import numpy as np
import torch
from torch.utils.data import DataLoader
from bise.data.rh20t.collate import collate_trajectories, trajectory_context
from bise.modalities.trajectory.factory import (
    build_trajectory_dataset, build_trajectory_model, split_trajectory_dataset,
)
from bise.modalities.trajectory.geometry import transform_tcp
from bise.modalities.trajectory.provenance import save_evaluation_provenance
from bise.retrieval.metrics import calculate_label_retrieval_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--all-as-test", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.data_root:
        config["data_root"] = args.data_root
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_trajectory_dataset(config, args.checkpoint)
    split = ({"all_as_test": True} if args.all_as_test else
             {"manifest_path": str(Path(args.checkpoint).resolve().parent / "split_manifest.json")})
    loader = DataLoader(split_trajectory_dataset(dataset, split)["test"],
                        batch_size=config["batch_size"], collate_fn=collate_trajectories)
    from bise.modalities.trajectory.provenance import validate_checkpoint_config
    validate_checkpoint_config(config, args.checkpoint)
    model = build_trajectory_model(config, device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    human, labels = [], []
    robots = {name: [] for name in ("original", "negate_q", "translate_x_0.1m", "rotate_z_90deg")}
    with torch.inference_mode():
        for batch in loader:
            h, hm = batch["human_poses"].to(device), batch["human_mask"].to(device)
            r, rm = batch["tcp_bases"].to(device), batch["tcp_mask"].to(device)
            hc, rc = trajectory_context(batch, "human", device), trajectory_context(batch, "robot", device)
            human.append(model.forward_human(h, hm, hc).cpu().numpy())
            labels.extend(batch["human_task_ids"])
            for name in robots:
                v = r.clone()
                if name == "negate_q":
                    v[..., 3:] *= -1
                elif name == "translate_x_0.1m":
                    v[..., 0] += .1
                elif name == "rotate_z_90deg":
                    q = r.new_tensor([2**-.5, 0, 0, 2**-.5])
                    v = transform_tcp(r, q, r.new_zeros(3), model.quaternion_order)
                robots[name].append(model.forward_robot(v, rm, rc).cpu().numpy())
    human = np.concatenate(human)
    robots = {name: np.concatenate(values) for name, values in robots.items()}
    original = human @ robots["original"].T
    results = {}
    for name, embeddings in robots.items():
        similarity = human @ embeddings.T
        results[name] = {
            "task": calculate_label_retrieval_metrics(similarity, np.array(labels), np.array(labels)),
            "mean_cosine_to_original": float(np.sum(embeddings * robots["original"], axis=-1).mean()),
            "embedding_max_abs_delta": float(abs(embeddings - robots["original"]).max()),
            "similarity_max_abs_delta": float(abs(similarity - original).max()),
        }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "robustness.json").write_text(json.dumps(results, indent=2))
    save_evaluation_provenance(out, config, args.checkpoint, dataset.manifest)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
