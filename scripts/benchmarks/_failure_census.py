"""Census of reader failure modes over a runner artefact.

Groups every wrong answer into mechanically-checkable modes so each fix can be
sized from real counts rather than anecdotes.

Usage:
  uv run python scripts/benchmarks/_failure_census.py --run cap24_coder7b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
NAMES = {1: "MULTI-HOP", 2: "TEMPORAL", 3: "OPEN-DOMAIN", 4: "SINGLE-HOP"}

ABSTAIN = ("no information available", "not mentioned")
BOOL_PRED = re.compile(r"^(likely\s+)?(yes|no)\b\.?$", re.I)
BARE_RELATIVE = re.compile(r"^(last|this|next|recently|yesterday|today)\b", re.I)
ANY_RELATIVE = re.compile(r"\b(last|this|next)\s+(fri|sat|sun|mon|tue|wed|thu|week|weekend|month|year|summer|winter|spring|fall)\b", re.I)
ITEM_SPLIT = re.compile(r"\s*(?:,|;|&|\band\b)\s*", re.I)
YEAR = re.compile(r"\b(19|20)\d{2}\b")


def is_bool(text: str) -> bool:
    return bool(BOOL_PRED.match(text.strip()))


def gt_boolish(gt: str) -> bool:
    return bool(re.match(r"^(likely|yes|no|it is|it's|probably)\b", gt.strip(), re.I))


def items(text: str) -> list[str]:
    return [p for p in ITEM_SPLIT.split(text) if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cap24_coder7b")
    ap.add_argument("--cache", default="benchmark_results/locomo_context_cache.jsonl")
    args = ap.parse_args()

    results = json.load(
        open(REPO / "benchmark_results" / "locomo1540" / f"{args.run}.json", encoding="utf-8")
    )["results"]

    contexts: dict[str, str] = {}
    for line in open(REPO / args.cache, encoding="utf-8"):
        d = json.loads(line)
        if d.get("category") != 5:
            contexts[d["qid"]] = d["context"]

    modes: dict[str, dict[int, int]] = {}
    totals: dict[int, int] = {}
    samples: dict[str, list] = {}

    def bump(mode: str, cat: int, row: dict) -> None:
        modes.setdefault(mode, {}).setdefault(cat, 0)
        modes[mode][cat] += 1
        samples.setdefault(mode, []).append(row)

    for r in results:
        cat = r["category"]
        if r["is_correct"]:
            continue
        totals[cat] = totals.get(cat, 0) + 1
        gt, pred = str(r["ground_truth"]), str(r["prediction"]).strip()
        ctx = contexts.get(r["question_id"], "")
        row = {"qid": r["question_id"], "q": r["question"], "gt": gt, "pred": pred}

        if not r["oracle_recall"]:
            bump("F_retrieval_miss (oracle=False)", cat, row)
        if any(m in pred.lower() for m in ABSTAIN) and r["oracle_recall"]:
            bump("C_spurious_refusal (evidence present)", cat, row)
        if gt_boolish(gt) != is_bool(pred) and (gt_boolish(gt) or is_bool(pred)):
            bump("B_boolean_template_leak", cat, row)
        if ANY_RELATIVE.search(pred) and YEAR.search(gt) and not YEAR.search(pred):
            bump("A_relative_abbrev_not_expanded", cat, row)
        if len(items(gt)) >= 2 and len(items(pred)) < len(items(gt)):
            bump("D_incomplete_enumeration", cat, row)
        if ctx and pred and len(pred) > 8:
            norm = re.sub(r"[^a-z0-9 ]", " ", pred.lower())
            if re.sub(r"[^a-z0-9 ]", " ", norm) in re.sub(r"[^a-z0-9 ]", " ", ctx.lower()):
                bump("E_faithful_copy_but_gt_differs", cat, row)

    cats = sorted({r["category"] for r in results if not r["is_correct"]})
    print(f"run={args.run}")
    print("\nWRONG ANSWERS BY CATEGORY")
    for c in cats:
        n = sum(1 for r in results if r["category"] == c)
        print(f"  {NAMES[c]:<13} {totals.get(c, 0):>4} / {n}  ({100 * totals.get(c, 0) / n:.1f}%)")

    print("\nFAILURE-MODE CENSUS (counts; a row can hit several modes)")
    print(f"{'mode':<45}" + "".join(f"{NAMES[c]:>14}" for c in cats))
    for mode in sorted(modes):
        line = f"{mode:<45}"
        for c in cats:
            line += f"{modes[mode].get(c, 0):>14}"
        print(line)

    print("\nEXAMPLES")
    for mode in sorted(modes):
        print(f"\n--- {mode}")
        for row in samples[mode][:3]:
            print(f"  [{row['qid']}] Q: {row['q']}")
            print(f"      GT : {row['gt']}")
            print(f"      PRED: {row['pred'][:180]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
