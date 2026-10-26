"""Where does the plural-list cluster actually fail?

The atlas calls 311-348 questions the 'plural-list union' cluster.  Before
building an identity system, this measures what the failure actually is, because
the cluster may be several different diseases sharing a label.

Reported per stratum: n, engine_claims, abstain, and gold cardinality.  A cluster
whose gold is mostly a SINGLE item is not a union problem at all -- it is an
extraction problem, and building a union engine for it would be work spent on the
wrong disease.

Also measures Eli's stated danger directly: how many questions have a subject
whose name could collide with another speaker's (two people both owning a 'Bailey'),
and how many questions state a TIME CHANGE that would make two same-named entities
genuinely distinct (used to own / now owns / adopted another).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"

#: Time-change markers: evidence that two same-named things may be DISTINCT
#: entities rather than one, which is exactly Eli's 'used to own Bailey vs now
#: owns a different Bailey' case.
TIME_CHANGE = re.compile(
    r"\b(used to|no longer|gave away|passed away|rehomed|adopted another|"
    r"new (?:dog|cat|pet)|another (?:dog|cat|pet)|previously|before)\b", re.I)


def gold_items(gold: str) -> int:
    return len([x for x in str(gold).split(",") if x.strip()])


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]

    # The cluster as the roadmap defines it: plural-question shapes with gold
    # fully present in context.  Both partial-enumeration and abstain failures.
    cluster = [
        r for r in atlas
        if r.get("shape") in ("what_thing", "what_person")
        and r.get("gold_in_context") == "full"
        and r.get("rescored_correct") is False
    ]

    print(f"CLUSTER SIZE                        : {len(cluster)}")
    print(f"  engine claims and is wrong        : "
          f"{sum(1 for r in cluster if r.get('engine_claims'))}")
    print(f"  abstains (no engine owns shape)   : "
          f"{sum(1 for r in cluster if not r.get('engine_claims'))}")
    print()

    card = Counter(gold_items(r["gold"]) for r in cluster)
    print("GOLD CARDINALITY -- what a union engine would even be for")
    print(f"  {'items':>6} {'n':>5} {'%':>6}")
    for k in sorted(card):
        print(f"  {k:>6} {card[k]:>5} {card[k] / len(cluster) * 100:>5.1f}%")
    plural = sum(n for k, n in card.items() if k >= 2)
    print(f"  --> gold has >=2 items (a REAL union target): {plural} "
          f"({plural / len(cluster) * 100:.1f}%)")
    print(f"  --> gold has 1 item (NOT a union problem)   : {card.get(1, 0)} "
          f"({card.get(1, 0) / len(cluster) * 100:.1f}%)")
    print()

    # Is the engine's failure on the truly-plural subset different from the rest?
    print("STRATIFIED: does cardinality change whether the engine claims?")
    print(f"  {'items':>6} {'n':>5} {'claims':>7} {'abstain':>8}")
    for k in sorted(card):
        rows = [r for r in cluster if gold_items(r["gold"]) == k]
        claims = sum(1 for r in rows if r.get("engine_claims"))
        print(f"  {k:>6} {len(rows):>5} {claims:>7} {len(rows) - claims:>8}")
    print()

    # Eli's two dangers, measured rather than assumed.
    print("ELI'S COLLISION DANGERS -- are they present in this data?")
    subjects = Counter()
    for r in cluster:
        for m in re.finditer(r"\b([A-Z][a-z]+)'s\b", r["question"]):
            subjects[m.group(1)] += 1
    print(f"  questions naming a possessive subject : "
          f"{sum(subjects.values())} across {len(subjects)} distinct subjects")
    print(f"  most common subjects                 : {subjects.most_common(6)}")
    print()
    print("  (whether two DIFFERENT people own the same name cannot be answered")
    print("   from the question text alone -- it needs the evidence turns.  The")
    print("   identity module therefore must scope keys by OWNER, not name, and")
    print("   the wrong-merge measurement below is what proves it did.)")
    print()

    # Do the frozen contexts for this cluster contain time-change language at all?
    cache = {}
    for line in (ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
                 ).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    with_change = 0
    checked = 0
    for r in cluster:
        ctx = cache.get(r.get("cache_qid") or "")
        if not ctx:
            continue
        checked += 1
        if TIME_CHANGE.search(ctx.get("context") or ""):
            with_change += 1
    if checked:
        print("TIME-CHANGE LANGUAGE IN THE FROZEN CONTEXTS")
        print(f"  contexts checked          : {checked}")
        print(f"  contain a time-change cue : {with_change} "
              f"({with_change / checked * 100:.1f}%)")
        print("  --> these are the questions where supersede-and-keep matters, and")
        print("      where merging two same-named entities would be WRONG.")
    print()

    print("CONCLUSION TO CARRY FORWARD")
    print("  The union target is the >=2-item subset, not the whole cluster.  Size")
    print("  it before promising a gain, and report the gain on THAT subset only --")
    print("  quoting it against the full cluster size would inflate the denominator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
