import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.retrieval.system.figures import METRIC_KEYS, plot_system_metrics
from bise.retrieval.system.io import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Export retrieval system final charts.")
    parser.add_argument("--runs-json", required=True, help="Mapping from label to eval output directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--level", default="scene", choices=["scene", "task"])
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = json.loads(Path(args.runs_json).read_text(encoding="utf-8"))
    series = []
    for label, eval_dir in runs.items():
        metrics = json.loads((Path(eval_dir) / "metrics.json").read_text(encoding="utf-8"))[args.level]
        series.append({"label": str(label), "eval_dir": str(eval_dir), "metrics": metrics})
    plot_system_metrics(series, output_dir / "system_metrics_comparison.png", dpi=args.dpi)
    save_json(
        output_dir / "system_metrics_comparison_data.json",
        {
            "level": args.level,
            "metric_keys": list(METRIC_KEYS),
            "series": series,
        },
    )


if __name__ == "__main__":
    main()
