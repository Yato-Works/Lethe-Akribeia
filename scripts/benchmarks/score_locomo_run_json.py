"""Run the PINNED official LoCoMo scorer over a single results JSON file.

Why this exists
    ``scripts/benchmarks/score_locomo_official.py`` expects a run *directory* with
    ``conv_*_results.json`` shards and the prediction stored under
    ``predicted_answer``.  The 1,540 memorised-context runner writes one JSON with
    the prediction under ``prediction``.

    This helper ONLY re-shapes stored predictions (it never regenerates, edits or
    drops a prediction) into that layout inside a scratch directory, then invokes
    the pinned scorer so the official metric (stemmed F1 for categories 1-4,
    literal refusal match for category 5) is computed by upstream code.

Usage:
    .venv-benchmarks/beam/Scripts/python.exe scripts/benchmarks/score_locomo_run_json.py \
        --results benchmark_results/locomo1540/locomo_1540_improved.json \
        --tag locomo1540_improved
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
NORMALIZED_DATASET = "benchmark_results/_official_scoring/locomo10_normalized.json"
SHARD_SIZE = 250


def main() -> int:
    parser = argparse.ArgumentParser(description="Official LoCoMo scoring for a single results JSON")
    parser.add_argument("--results", required=True, help="Results JSON produced by a LoCoMo runner")
    parser.add_argument("--tag", required=True, help="Tag for the scratch run dir and output file")
    parser.add_argument("--dataset", default=NORMALIZED_DATASET,
                        help="Dataset whose QA items carry an 'answer' key (see AM_APEX_STATUS.md)")
    args = parser.parse_args()

    results_path = REPO / args.results
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload)
    if not isinstance(rows, list) or not rows:
        print(f"No results found in {results_path}")
        return 1

    run_dir = REPO / "benchmark_results" / "_official_scoring" / f"run_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in run_dir.glob("conv_*_results.json"):
        stale.unlink()

    shards = 0
    for start in range(0, len(rows), SHARD_SIZE):
        chunk = []
        for row in rows[start:start + SHARD_SIZE]:
            prediction = row.get("predicted_answer", row.get("prediction", ""))
            chunk.append({
                "question_id": row["question_id"],
                "category": row["category"],
                "predicted_answer": prediction,
                "is_correct": row.get("is_correct"),
                "tokens_used": row.get("tokens_used", row.get("tokens")),
                "latency_ms": row.get("latency_ms"),
            })
        out = run_dir / f"conv_{shards}_results.json"
        out.write_text(json.dumps({"results": chunk}, ensure_ascii=False, indent=1), encoding="utf-8")
        shards += 1

    print(f"re-shaped {len(rows)} stored predictions into {shards} shard(s): {run_dir.relative_to(REPO)}")

    out_json = f"benchmark_results/official_locomo_score_{args.tag}.json"
    cmd = [
        str(REPO / ".venv-benchmarks" / "beam" / "Scripts" / "python.exe"),
        str(REPO / "scripts" / "benchmarks" / "score_locomo_official.py"),
        "--run", str(run_dir.relative_to(REPO)).replace("\\", "/"),
        "--dataset", args.dataset,
        "--out", out_json,
        "--tag", args.tag,
    ]
    print("running pinned official scorer:", " ".join(cmd[1:]))
    return subprocess.call(cmd, cwd=str(REPO))


if __name__ == "__main__":
    sys.exit(main())
