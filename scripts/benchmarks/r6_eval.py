"""r6_eval.py - R6 comparison and reporting for the evidence-state feature pipeline.

Minimal re-implementation of eval_v4_pipeline_dev.py that only runs R6-specific
comparison and reporting. It does NOT re-run any questions. Instead, it reads the
existing R0/R2/R3 execution JSON files and produces R6 features + comparison
metrics in one pass.

Usage:
    python r6_eval.py --cond R6 --r6-out benchmark_results/pipeline_v4/r6_r0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_features(items):
    """Compute aggregate R6 feature statistics across all evaluated questions."""
    if not items:
        return {}

    keys = set()
    for item in items:
        keys.update(item.get("r6_features", {}).keys())

    summary = {k: {"min": 0, "max": 0, "mean": 0.0, "count": 0} for k in keys}

    for item in items:
        feats = item.get("r6_features", {})
        for k, v in feats.items():
            s = summary[k]
            s["count"] += 1
            if isinstance(v, bool):
                # Bool is a 0/1 flag
                s["min"] = min(s["min"], 1 if v else 0)
            elif isinstance(v, (int, float)):
                s["min"] = min(s["min"], v)
            else:
                s["min"] = min(s["min"], 0)  # string fallback
            s["max"] = max(s["max"], v if isinstance(v, (int, float)) else 1)

    # After we have all values, compute means
    sums = {k: 0.0 for k in summary}
    for item in items:
        feats = item.get("r6_features", {})
        for k, v in feats.items():
            if isinstance(v, (int, float)):
                sums[k] += v

    for k, v in summary.items():
        if v["count"] > 0:
            v["mean"] = sums[k] / v["count"]

    # Remove items with no values
    return {k: v for k, v in summary.items() if v["count"] > 0}


def output_report(summary: dict, path: Path):
    """Placeholder: produce a text/JSON report. The full reporting will be added
    later when the evaluation model is introduced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[r6_eval] Summary written to {path}")


def main():
    parser = argparse.ArgumentParser(description="R6 evaluation utilities")
    parser.add_argument("--cond", required=True, help="Condition label, e.g. R6")
    parser.add_argument(
        "--r6-out",
        required=True,
        help="Path where the R6 evaluation result will be saved",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSON file (default: .json from a prior R0/R2/R3 run)",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_pipeline.json"
    if not input_path.exists():
        print(f"[r6_eval] ERROR: input file not found: {input_path}")
        sys.exit(1)

    data = load_results(input_path)
    results = data.get("results", [])

    print(f"[r6_eval] Loaded {len(results)} results from {input_path}")

    # Summarize R6 feature values
    summary = summarize_features(results)

    # Save
    output_report(summary, Path(args.r6_out))

    print(f"[r6_eval] Done. Total questions processed: {len(results)}")


if __name__ == "__main__":
    main()
