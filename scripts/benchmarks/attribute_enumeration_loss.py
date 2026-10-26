"""Attribute the list-item loss to retrieval, selection, or nothing.

The measurement that motivates this file: for LoCoMo multi-hop list questions,
only ~40% of the gold items survive into the compiled context, while ~65% exist
somewhere in the conversation.  That 25-point gap can be caused by two very
different stages, and they need different work:

    CONVERSATION_ONLY  the turn carrying the item never entered the candidate pool
                       -> retrieval / coverage problem
    POOL_ONLY          the turn was pooled but the compiler did not select it
                       -> budget / selection problem
    IN_CONTEXT         already available to the Reader
    ABSENT             not an exact string in the conversation (paraphrase, or a
                       genuine omission); counted separately and never called a
                       defect

The classification uses exact stem matching, so ABSENT is an upper bound on
paraphrase, not proof of absence.  That caveat is why the numbers are reported as
four buckets rather than as a single "recoverable" figure.

Usage
    python scripts/benchmarks/eval_locomo_full.py --cats 1 --freeze-pool \
        --freeze-contexts benchmark_results/frozen_ctx/locomo_r7_cat1_pool
    python scripts/benchmarks/attribute_enumeration_loss.py \
        --contexts benchmark_results/frozen_ctx/locomo_r7_cat1_pool/contexts.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUCKETS = ("IN_CONTEXT", "POOL_ONLY", "CONVERSATION_ONLY", "ABSENT")


def gold_items(gold: str) -> list[str]:
    return [x.strip().strip("\"'") for x in re.split(r",|;|\band\b", gold) if x.strip()]


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "e"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def contains(item: str, text: str) -> bool:
    """Stem-level containment: 'mountains' matches 'mountain', not 'mount'."""
    key = stem(item.lower())
    return re.search(rf"\b{re.escape(key)}[a-z]{{0,2}}\b", text.lower()) is not None


def conversation_text(dataset: list[dict]) -> dict[str, str]:
    out = {}
    for sample in dataset:
        parts = []
        for key, turns in sample["conversation"].items():
            if key.startswith("session_") and isinstance(turns, list):
                parts.extend(str(t.get("text", "")) for t in turns)
        out[sample["sample_id"]] = " ".join(parts)
    return out


def classify(row: dict, conversation: str) -> Counter:
    counts: Counter = Counter()
    context = row.get("context") or ""
    pool_text = " ".join(t.get("t", "") for t in (row.get("pool") or []))
    for item in gold_items(row.get("gold") or ""):
        if contains(item, context):
            counts["IN_CONTEXT"] += 1
        elif contains(item, pool_text):
            counts["POOL_ONLY"] += 1
        elif contains(item, conversation):
            counts["CONVERSATION_ONLY"] += 1
        else:
            counts["ABSENT"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    conversation = conversation_text(dataset)
    rows = [json.loads(line) for line in
            (ROOT / args.contexts).read_text(encoding="utf-8").splitlines() if line.strip()]

    total: Counter = Counter()
    per_question: list[dict] = []
    for row in rows:
        sid = row["question_id"][3:].rsplit("_", 1)[0]
        counts = classify(row, conversation.get(sid, ""))
        total.update(counts)
        if (row.get("gold") or "") and any(counts[b] for b in ("POOL_ONLY", "CONVERSATION_ONLY")):
            per_question.append({
                "question_id": row["question_id"],
                "question": row["question"][:100],
                **{b: counts[b] for b in BUCKETS},
            })

    n_items = sum(total.values())
    print(f"questions: {len(rows)}   gold items: {n_items}")
    for bucket in BUCKETS:
        print(f"  {bucket:<18} {total[bucket]:4d}  ({total[bucket] / max(1, n_items) * 100:5.1f}%)")
    print()
    touched_selection = sum(1 for q in per_question if q["POOL_ONLY"])
    touched_retrieval = sum(1 for q in per_question if q["CONVERSATION_ONLY"])
    print(f"  questions with a pooled-but-dropped item : {touched_selection}")
    print(f"  questions with an unpolled item           : {touched_retrieval}")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "contexts": args.contexts,
            "questions": len(rows),
            "items": n_items,
            "totals": dict(total),
            "questions_with_pool_only": touched_selection,
            "questions_with_conversation_only": touched_retrieval,
            "detail": per_question,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
