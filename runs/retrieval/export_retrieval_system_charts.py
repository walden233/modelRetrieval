import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.retrieval.system.figures import METRIC_KEYS, plot_system_metrics
from bise.retrieval.system.io import save_json


def plot_label(label, weights):
    if not weights or "selected" not in label.lower():
        return label
    return (
        f"{label}\n"
        f"V={float(weights.get('video', 0)):.2f}, "
        f"T={float(weights.get('trajectory', 0)):.2f}, "
        f"S={float(weights.get('semantic_text', 0)):.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export retrieval system final charts.")
    parser.add_argument("--runs-json", required=True, help="Mapping from label to eval output directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--level", default="scene", choices=["scene", "task"])
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--data-name",
        default="system_metrics_comparison_data.json",
        help="Filename for the exported chart data JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = json.loads(Path(args.runs_json).read_text(encoding="utf-8"))
    series = []
    for label, eval_dir in runs.items():
        eval_path = Path(eval_dir)
        metrics = json.loads((eval_path / "metrics.json").read_text(encoding="utf-8"))[args.level]
        item = {"label": str(label), "eval_dir": str(eval_dir), "metrics": metrics}
        config_path = eval_path / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            weights = (config.get("fusion") or {}).get("weights")
            if weights:
                item["fusion_weights"] = weights
                item["plot_label"] = plot_label(item["label"], weights)
        series.append(item)
    plot_system_metrics(series, output_dir / "system_metrics_comparison.png", dpi=args.dpi)
    save_json(
        output_dir / args.data_name,
        {
            "level": args.level,
            "metric_keys": list(METRIC_KEYS),
            "series": series,
        },
    )


if __name__ == "__main__":
    main()
