"""Recompute the LoCoMo committer-claims artefact from the frozen context cache.

Why this exists
---------------
The committer is deterministic and LLM-free, so the claimed subset and its
precision can be recomputed exactly from (a) the cached compiled contexts and
(b) the reader run that supplies ground truth / reader_correct, using the same
call the adapter makes in production::

    commit_answer(question, context, category)   # locomo_adapter.py:586

The artefact this script replaced was measured on an earlier committer revision
and an earlier cache snapshot, so it could no longer be re-derived.  Keeping
this recomputation committed turns the published "committer precision" claim
into a reproducible procedure instead of a frozen number.

Usage::

    python scripts/benchmarks/recompute_committer_claims.py \
        --cache benchmark_results/locomo_context_cache_rules.jsonl \
        --run benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json \
        --category 2 \
        --out benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.skills.answer_committer import MIN_TURN_SCORE, commit_answer

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="benchmark_results/locomo_context_cache_rules.jsonl")
    ap.add_argument("--run", default="benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json")
    ap.add_argument("--category", type=int, default=2)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache_path = REPO / args.cache
    run_path = REPO / args.run
    if not cache_path.exists():
        print(f"!! cache not found: {cache_path}")
        return 2
    if not run_path.exists():
        print(f"!! reader run not found: {run_path}")
        return 2

    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_rows = {r["question_id"]: r for r in run["results"]}

    records: list[dict] = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("category") or 0) != args.category:
            continue
        committed = commit_answer(row["question"], row["context"], category=args.category)
        run_row = run_rows.get(row["qid"], {})
        gt = str(run_row.get("ground_truth") or "")
        claimed = bool(committed.used)
        commit_correct = bool(claimed and LoCoMoAdapter.score_binary(args.category, gt, committed.answer))
        reader_correct = bool(run_row.get("is_correct"))
        records.append(
            {
                "question_id": row["qid"],
                "group": f"{args.category} ({CAT.get(args.category, '?')})",
                "claimed": claimed,
                "commit_correct": commit_correct,
                "reader_correct": reader_correct,
                "oracle_recall": run_row.get("oracle_recall"),
            }
        )

    counts = {
        "n": len(records),
        "claimed": sum(1 for r in records if r["claimed"]),
        "commit_correct": sum(1 for r in records if r["commit_correct"]),
    }
    payload = {
        "label": args.label or f"LoCoMo 1,540 - category {args.category} ({CAT.get(args.category, '?')})",
        "suite": "locomo",
        "cache": args.cache,
        "cache_sha256": sha256(cache_path),
        "reader_run": args.run,
        "committer": "LoCoMoAdapter.evaluate_question -> commit_answer (production call path)",
        "matcher": "LoCoMoAdapter.score_binary (matcher-v2)",
        "min_turn_score": MIN_TURN_SCORE,
        "generated_by": "scripts/benchmarks/recompute_committer_claims.py",
        "notes": (
            "Recomputed from the frozen context cache with the current committer "
            "revision, superseding the earlier sweep (109 claimed / 77 correct) "
            "measured on an older committer + cache snapshot."
        ),
        "counts": counts,
        "records": records,
    }
    precision = counts["commit_correct"] / counts["claimed"] * 100 if counts["claimed"] else 0.0
    print(f"category {args.category}: n={counts['n']} claimed={counts['claimed']} "
          f"commit_correct={counts['commit_correct']} ({precision:.2f}% precision)")

    if args.out:
        out = REPO / args.out
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
