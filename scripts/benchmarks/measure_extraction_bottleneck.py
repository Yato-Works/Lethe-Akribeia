"""Is the identity module or the extractor the bottleneck?

measure_identity_union.py reported FIXED 0 / BROKE 31 / WRONG MERGES 0 with only
1 of 242 gold items recovered.  Before concluding anything, this separates the two
possible causes:

  * the IDENTITY module is too conservative, or
  * the EXTRACTOR never surfaces the items in the first place.

For every gold item it asks: is the string even present in the frozen context,
and is it reachable by the crude 'named X' pattern the measurement used?  If the
items are present but the pattern misses them, identity was never the constraint
-- and the fix is extraction, which the Cognee digest identified as the one
irreducible model dependency.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

NAME = re.compile(r"\b(?:named|called)\s+([A-Z][a-z]{2,})\b")


def gold_items(gold: str) -> list[str]:
    return [x.strip() for x in str(gold).split(",") if x.strip()]


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    cohort = [
        r for r in atlas
        if r.get("shape") in ("what_thing", "what_person")
        and r.get("gold_in_context") == "full"
        and r.get("rescored_correct") is False
        and len(gold_items(r["gold"])) >= 2
    ]

    total = present = via_named = 0
    missing_examples: list[tuple] = []

    for row in cohort:
        ctx = cache.get(row.get("cache_qid"))
        if not ctx:
            continue
        context = ctx.get("context") or ""
        lowered = context.lower()
        named = {x.lower() for x in NAME.findall(context)}
        for item in gold_items(row["gold"]):
            total += 1
            key = item.strip().lower()
            if key in lowered:
                present += 1
            if key in named:
                via_named += 1
            elif key not in lowered and len(missing_examples) < 8:
                missing_examples.append((row["question_id"], item,
                                         row["question"][:52]))

    print(f"gold items in the real union target : {total}")
    print(f"  present verbatim in the context   : {present} "
          f"({present / total * 100:.1f}%)")
    print(f"  reachable by the 'named X' pattern: {via_named} "
          f"({via_named / total * 100:.1f}%)")
    print()
    print("=> items present but NOT reachable by the pattern are extraction")
    print("   failures, not identity failures.  The gap between the two lines is")
    print("   exactly how much a better extractor could recover without touching")
    print("   the identity module at all.")
    if missing_examples:
        print("\ngold items not even present verbatim (first 8):")
        for qid, item, q in missing_examples:
            print(f"  {qid:<20} gold={item!r:<24} q={q!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
