"""Phase 1 -- trace each over-committing answer to the engine path that produced it.

Six ENGINE_WRONG_EVIDENCE_ABSENT cases committed an answer whose gold is absent from
the frozen context.  Before building any gate, this establishes WHICH code path
produced each one, because a gate placed in the wrong layer is decoration.

The committer's `detail` string names the branch, and the committer already carries a
`source` field naming the subsystem.  This records, per case:
  * the branch label from detail,
  * the source subsystem,
  * the chosen evidence turn (the answer's own provenance),
  * whether the chosen turn even mentions the asked person,
  * whether the gold date appears in ANY turn of the frozen context.

That last check is the one the gate will need: if the gold is nowhere in the context,
then ANY answer is unsupported and the honest action is abstention.  If the gold IS in
the context but a different turn was chosen, the defect is turn SELECTION and the gate
belongs upstream of the committer.

Reported per case with the evidence turn shown verbatim, so the classification can be
checked rather than trusted.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

MAP = ROOT / "benchmark_results/when_date_map/when_date_failure_map.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/overcommit_trace.json"

MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")


def gold_parts(gold: str):
    """(day, month) if the gold names one, else None."""
    s = str(gold).strip()
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)", s)
    if m and m.group(2).lower() in MONTHS:
        return m.group(1), m.group(2).lower()
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})\b", s)
    if m and m.group(1).lower() in MONTHS:
        return m.group(2), m.group(1).lower()
    return None


def date_anywhere(day: str, month: str, context: str) -> list[str]:
    pats = [re.compile(rf"\b{day}\s+{month}\b", re.I),
            re.compile(rf"\b{month}\s+{day}(?:st|nd|rd|th)?\b", re.I)]
    return [ln.strip()[:150] for ln in context.splitlines()
            if any(p.search(ln) for p in pats)]


def main() -> int:
    data = json.loads(MAP.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    cases = [r for r in data["records"]
             if r["engine_class"] == "ENGINE_WRONG_EVIDENCE_ABSENT"]
    print(f"over-committing cases : {len(cases)}")
    print()

    records: list[dict] = []
    branch_counts = Counter()

    for r in cases:
        context = (cache.get(r["cache_qid"]) or {}).get("context") or ""
        turns = parse_turns(context)
        gold = r["gold"]
        gp = gold_parts(gold)

        # Re-run live to capture source + evidence turn, which the map did not store.
        live = DeterministicMemoryEngine.resolve(r["question"], context)
        detail = live.detail or r.get("engine_detail") or ""
        source = getattr(live, "source", "") or ""
        evidence = getattr(live, "evidence_turn", "") or ""

        # Which branch produced it?  The detail string names it.
        branch = "unknown"
        for label in ("extracted relative temporal anchor",
                      "relative event anchor",
                      "extracted explicit date",
                      "month-shape answer",
                      "resolved"):
            if label in detail:
                branch = label
                break
        branch_counts[branch] += 1

        # Is the gold date anywhere in the frozen context at all?
        gold_lines = date_anywhere(*gp, context) if gp else []

        # Does the chosen evidence turn even mention the asked person?
        # 'When' is the first capitalised token in a When-question, which is not a
        # person.  Use the engine's own person extractor instead of a regex.
        from artificial_memory.skills.autonomous_engines.temporal_algebra_engine import (
            TemporalAlgebraEngine,
        )
        person = TemporalAlgebraEngine._extract_person(r["question"], turns)
        ev_turn = None
        for t in turns:
            if evidence and str(evidence).strip()[:40] in t.text:
                ev_turn = t
                break
        person_in_evidence = bool(
            person and ev_turn
            and person.lower() in (ev_turn.speaker + " " + ev_turn.text).lower())

        records.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "gold": gold,
            "engine_answer": live.answer or r["engine_answer"],
            "branch": branch,
            "source": source,
            "detail": detail[:140],
            "evidence_turn_id": (ev_turn.dia_id if ev_turn else None),
            "evidence_turn_text": (re.sub(r"\s+", " ", ev_turn.text)[:190]
                                   if ev_turn else None),
            "person_asked": person,
            "person_in_evidence_turn": person_in_evidence,
            "gold_present_in_context": bool(gold_lines),
            "gold_context_lines": gold_lines[:2],
        })

    print("=" * 74)
    print("WHICH BRANCH COMMITTED EACH UNSUPPORTED ANSWER")
    print("=" * 74)
    for k, n in branch_counts.most_common():
        print(f"  {k:<38} {n}")
    print()

    print("=" * 74)
    print("PER CASE -- is the gold even present, and did the chosen turn have the person?")
    print("=" * 74)
    for x in records:
        print(f"  {x['question_id']}")
        print(f"    Q      : {x['question'][:62]}")
        print(f"    gold   : {x['gold']!r}")
        print(f"    gave   : {x['engine_answer']!r}")
        print(f"    branch : {x['branch']}   source={x['source']!r}")
        print(f"    gold in context?      : {x['gold_present_in_context']}")
        print(f"    person in evidence?   : {x['person_in_evidence_turn']} "
              f"(person={x['person_asked']!r})")
        if x["evidence_turn_id"]:
            print(f"    evidence [{x['evidence_turn_id']}]: "
                  f"{x['evidence_turn_text'][:130]}")
        print()

    n_absent = sum(1 for x in records if not x["gold_present_in_context"])
    n_no_person = sum(1 for x in records if not x["person_in_evidence_turn"])
    print("=" * 74)
    print("WHAT THIS IMPLIES FOR A GATE")
    print("=" * 74)
    print(f"  gold absent from the context entirely      : {n_absent}/{len(records)}")
    print(f"  asked person absent from the chosen turn   : {n_no_person}/{len(records)}")
    print()
    if n_absent == len(records):
        print("  Every one committed an answer whose gold is nowhere in the context.")
        print("  The honest action is abstention, and a single check -- 'does the frozen")
        print("  context contain ANY date consistent with the asked event' -- would have")
        print("  caught all of them.  That check is the Evidence-Commit Gate.")
    else:
        print("  Mixed: some have gold present, so the defect is turn SELECTION for")
        print("  those and the gate must be upstream of the committer.")

    OUT.write_text(json.dumps({
        "label": "phase 1 -- over-committing when_date answers traced to their branch",
        "n": len(records),
        "branches": dict(branch_counts),
        "gold_absent_count": n_absent,
        "person_absent_in_evidence_count": n_no_person,
        "cases": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
