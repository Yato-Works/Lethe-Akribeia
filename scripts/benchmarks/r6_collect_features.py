"""r6_collect_features.py - Extract R6 deterministic evidence-state features from
existing R0/R2/R3 execution JSON files. No re-run of questions; just feature
unpacking and trimming.

Usage:
    python r6_collect_features.py --cond R0 --reader-model qwen2.5:7b-instruct
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


def extract_r6_features(item: dict, r6_features: dict, predicted: str, is_correct: bool, reader_model: str) -> dict:
    """Return a feature dict with the required keys plus metadata."""
    base = {
        "question_id": item.get("question_id"),
        "question_type": item.get("question_type"),
        "ground_truth": item.get("ground_truth"),
        "predicted": predicted,
        "is_correct": is_correct,
        "r6_features": r6_features,
        "reader_model": reader_model,
    }
    return base


def collect_features(src_path: Path, cond: str, reader_model: str, out_path: Path):
    data = load_results(src_path)
    results = data.get("results", [])

    if out_path is None:
        out_path = (
            REPO
            / "benchmark_results"
            / "pipeline_v4"
            / f"r6_features_{cond}_{reader_model.replace(':', '_')}.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for idx, item in enumerate(results):
        # Predicted answer / correctness are required for features (they are
        # added by the upstream R0/R2/R3 run). If missing, we fall back to
        # reading them from the original item.
        if "predicted_answer" in item:
            predicted = item["predicted_answer"]
        elif "predicted" in item:
            predicted = item["predicted"]
        else:
            predicted = ""

        if "is_correct" in item:
            is_correct = bool(item["is_correct"])
        elif "correct" in item:
            is_correct = bool(item["correct"])
        else:
            is_correct = ""

        r6_f = item.get("r6_features", {})

        entry = {
            "question_id": item.get("question_id"),
            "question_type": item.get("question_type"),
            "question": item.get("question"),
            "ground_truth": item.get("ground_truth"),
            "predicted": predicted,
            "is_correct": is_correct,
            "r6_features": r6_f,
            "reader_model": reader_model,
        }
        entries.append(entry)

        if (idx + 1) % 25 == 0:
            print(f"[r6_collect_features] Processed {idx + 1}/{len(results)}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"[r6_collect_features] Wrote {len(entries)} entries to {out_path}")
    return entries


def main():
    parser = argparse.ArgumentParser(description="R6 feature collector")
    parser.add_argument("--cond", required=True, help="Condition label, e.g. R0")
    parser.add_argument(
        "--reader-model",
        required=True,
        help="Reader model label to store with each entry",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Existing R0/R2/R3 run JSON file (default: R0 run file)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: benchmark_results/pipeline_v4/r6_features_<cond>_<reader_model>.json)",
    )
    args = parser.parse_args()

    input_path = (
        Path(args.input)
        if args.input
        else REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_pipeline.json"
    )

    if not input_path.exists():
        print(f"[r6_collect_features] ERROR: input file not found: {input_path}")
        sys.exit(1)

    out_path = Path(args.out) if args.out else None
    collect_features(input_path, args.cond, args.reader_model, out_path)


if __name__ == "__main__":
    main()
