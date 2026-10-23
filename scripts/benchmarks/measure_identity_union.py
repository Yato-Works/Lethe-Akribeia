"""Does the identity resolver improve the REAL union target?

The union target is 90 questions (gold with >=2 items), not the 348-question
cluster -- quoting a gain against 348 would inflate the denominator 3.9x, which
measure_union_cluster.py established.

Reported as (n, FIXED, BROKE, WRONG_MERGES).  The last number is the point of the
whole module: recall can always be raised by merging aggressively, so a run whose
wrong-merge count is above zero has not improved anything, it has merely become
confidently wrong.

WRONG MERGE, defined for this measurement: two mentions grouped into one entity
when the evidence says they are distinct -- different owners, or a discontinuity
cue between them.  Eli named both cases explicitly before this was built, and both
are measured rather than assumed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys_path = ROOT / "src"
import sys
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory.skills.autonomous_engines.entity_identity import (  # noqa: E402
    IdentityResolver,
    Mention,
)

ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

#: 'named X' / 'X and Y' -- a crude surface extractor.  Deliberately simple: the
#: question is whether identity-scoped grouping helps at all, not whether this is
#: the best possible name extractor.
_NAME = re.compile(r"\b(?:named|called)\s+([A-Z][a-z]{2,})\b")


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
        and len(gold_items(r["gold"])) >= 2          # the REAL union target
    ]
    print(f"REAL union target (gold >=2 items)  : {len(cohort)}")
    print()

    scored = 0
    fixed = broke = 0
    wrong_merges_total = 0
    recovered_items = 0
    gold_items_total = 0

    for row in cohort:
        ctx = cache.get(row.get("cache_qid") or "")
        if not ctx:
            continue
        turns = parse_turns(ctx.get("context") or "")
        if len(turns) < 2:
            continue
        scored += 1

        owner = ""
        m = re.search(r"\b([A-Z][a-z]+)'s\b", row["question"])
        if m:
            owner = m.group(1)

        resolver = IdentityResolver()
        for t in turns:
            for name in _NAME.findall(t.text):
                resolver.add(Mention(
                    turn_id=t.dia_id, speaker=t.speaker, surface=name,
                    text=t.text, owner=owner or t.speaker, entity_type="pet",
                ))

        findings = resolver.wrong_merge_audit()
        wrong_merges_total += len(findings)

        # Does the identity grouping now surface the gold items with provenance?
        known = {ent.canonical.lower() for ent in resolver.entities.values()}
        items = gold_items(row["gold"])
        gold_items_total += len(items)
        hit = [i for i in items if i.strip().lower().rstrip("s") in
               {k.rstrip("s") for k in known}]
        recovered_items += len(hit)

        was = bool(row.get("engine_claims"))
        now = len(hit) == len(items) and len(items) > 0
        if now and not was:
            fixed += 1
        elif was and not now:
            broke += 1

    print(f"questions scored                     : {scored}")
    print(f"FIXED  (abstained before, full now)  : {fixed}")
    print(f"BROKE  (claimed before, not now)     : {broke}")
    print(f"WRONG MERGES                         : {wrong_merges_total}")
    print()
    if gold_items_total:
        print(f"gold items recovered with provenance : {recovered_items}/"
              f"{gold_items_total} ({recovered_items / gold_items_total * 100:.1f}%)")
    print()
    print("HOW TO READ THIS")
    print(f"  FIXED>0 with WRONG MERGES==0  -> the identity grouping genuinely")
    print(f"    recovered scattered items; the mechanism is sound and worth wiring in.")
    print(f"  WRONG MERGES>0                -> recall was bought by merging things")
    print(f"    that should stay separate.  That is Eli's collision danger, realised.")
    print(f"    Not a success however high the recall is.")
    if wrong_merges_total:
        print(f"\n  --> this run has {wrong_merges_total}.  Kill criterion engaged: the")
        print(f"      hypothesis is NOT supported as implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
