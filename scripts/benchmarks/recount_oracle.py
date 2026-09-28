#!/usr/bin/env python3
"""Recompute a run's ``oracle_recall`` from the frozen context cache.

The runner recorded whether the gold evidence turn reached the reader with a
*label* test, and the labels turned out not to mean what they say (the context
compiler numbers the turns it emits in its own group order).  The compiled
context is frozen on disk, so the flag does not need a re-run to be corrected:
for every question of an existing artefact it is recomputed from the cached
context and the dataset, by content and session date.

The original artefact is never touched.  The corrected flags are written to a new
file (``<stem>.oraclefix.json``) so that every downstream tool - the failure
ceiling, the model-sensitivity controls - can be pointed at either version and the
difference is auditable rather than assumed away.

Usage:
  uv run python scripts/benchmarks/recount_oracle.py \\
      --run benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json \\
      --cache benchmark_results/locomo_context_cache_rules.jsonl \\
      --out benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.oraclefix.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

_audit_spec = importlib.util.spec_from_file_location(
    "_oracle_fn_audit_shared", REPO / "scripts" / "benchmarks" / "oracle_fn_audit.py")
_audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(_audit)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--out", default=None, help="default: <run stem>.oraclefix.json")
    args = ap.parse_args()

    run_path = REPO / args.run
    data = json.loads(run_path.read_text(encoding="utf-8"))
    dataset = _audit.load_dataset(str(REPO / args.dataset))
    contexts = _audit.load_contexts(str(REPO / args.cache))

    flipped_to_true = flipped_to_false = unchanged_true = unchanged_false = 0
    skipped = 0
    for row in data.get("results", []):
        qid = str(row.get("question_id") or "")
        item = dataset.get(qid)
        if item is None or not item.get("evidence_ids"):
            skipped += 1
            continue
        here, _ = _audit.evidence_present(contexts.get(qid, ""), item.get("turns") or [],
                                          item["evidence_ids"])
        was = bool(row.get("oracle_recall"))
        row["oracle_recall_by_id"] = was
        row["oracle_recall"] = here
        if here and not was:
            flipped_to_true += 1
        elif was and not here:
            flipped_to_false += 1
        elif here:
            unchanged_true += 1
        else:
            unchanged_false += 1

    out = REPO / f"{run_path.relative_to(REPO).with_suffix('')}.oraclefix.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    total = sum(1 for r in data.get("results", []) if r.get("question_id"))
    print(f"recounted {total} questions ({skipped} without gold evidence ids)")
    print(f"  flag False -> True (the evidence was there) : {flipped_to_true}")
    print(f"  flag True  -> False (it was not)           : {flipped_to_false}")
    print(f"  unchanged True {unchanged_true}, unchanged False {unchanged_false}")
    print(f"  original : {REPO / args.run}")
    print(f"  corrected: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
