"""Measure the quantity-TYPE confusion across every counting question.

The mission asks for typed cardinality reasoning: distinguish entity count from
occurrence count from duration, and never let one flow where another is asked.

Before building, this measures the population and the confusion, because a type
system that covers 2 questions is a special case wearing a general name.

Types:
  ENTITY      how many <countable noun>   -- people, dogs, pets, tournaments
  OCCURRENCE  how many times / how often  -- a frequency
  DURATION    how many days/weeks/months  -- an elapsed span

The defect being fixed: an ENTITY question routed into the occurrence path, which
counts distinct SESSIONS and answers 'ten times, 10' to 'how many people'.

Reported per type: how many questions, and how many the current engine answers
wrongly.  A type with a real population AND real failures is a justified target.
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
from artificial_memory.skills.autonomous_engines.counting_engine import (  # noqa: E402
    CountingEngine,
)
from artificial_memory.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass

AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/cardinality_type_population.json"

_OCC = re.compile(r"\bhow many times\b|\bhow often\b", re.I)
_DUR = re.compile(
    r"\bhow (?:many|much)\s+(?:days?|weeks?|months?|years?|hours?|minutes?|long)\b",
    re.I)
_ENT = re.compile(r"\bhow many\s+([a-z]+)", re.I)


def qtype(question: str) -> str:
    if _OCC.search(question):
        return "OCCURRENCE"
    if _DUR.search(question):
        return "DURATION"
    if _ENT.search(question):
        return "ENTITY"
    return "OTHER"


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    def cq(qid: str) -> str | None:
        m = re.fullmatch(r"lc_conv-(\d+)_(\d+)", qid)
        return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}" if m else None

    rows: list[dict] = []
    by_type = Counter()
    wrong_by_type = Counter()
    confusion: list[dict] = []

    for r in audit["records"]:
        t = qtype(r["question"])
        if t not in ("ENTITY", "OCCURRENCE", "DURATION"):
            continue
        by_type[t] += 1
        ctx = cache.get(cq(r["question_id"])) or {}
        context = ctx.get("context") or ""
        gold = r["gold"]
        row = {"question_id": r["question_id"], "type": t,
               "question": r["question"], "gold": gold}
        if not context:
            rows.append(row)
            continue
        turns = parse_turns(context)
        ans = DeterministicEngine_resolve = DeterministicMemoryEngine.resolve(
            r["question"], context)
        row["answer"] = ans.answer
        row["detail"] = (getattr(ans, "detail", "") or "")[:110]
        ok = bool(ans.used) and scorer.score(ans.answer, gold)
        row["verdict"] = "correct" if ok else ("wrong" if ans.used else "abstain")
        if not ok:
            wrong_by_type[t] += 1
            # The defect: an ENTITY question answered by the occurrence path.
            if t == "ENTITY" and "distinct activity sessions" in row["detail"]:
                confusion.append(row)
        rows.append(row)

    print("=" * 74)
    print("QUANTITY-TYPE POPULATION (all 1,986 questions)")
    print("=" * 74)
    print(f"  {'type':<12}{'n':>5}{'wrong/abstain':>16}")
    for t in ("ENTITY", "OCCURRENCE", "DURATION"):
        print(f"  {t:<12}{by_type[t]:>5}{wrong_by_type[t]:>16}")
    print()
    print("THE DEFECT -- ENTITY questions answered by the OCCURRENCE path")
    print(f"  n = {len(confusion)}")
    for x in confusion:
        print(f"    {x['question_id']:<17} gold={str(x['gold'])[:14]!r:<16} "
              f"gave={str(x['answer'])[:18]!r}")
        print(f"       {x['question'][:64]}")
    print()
    print("HOW TO READ THIS")
    print("  ENTITY has a real population and real failures, so a type guard there")
    print("  is justified.  OCCURRENCE and DURATION are reported for the contrast")
    print("  cases the mission asks for -- the same rule must not disturb them.")
    print()
    print("  The guard is a ROUTING guard, not a new capability: it stops an ENTITY")
    print("  question reaching the session-counting path.  It cannot invent the right")
    print("  number, so the expected effect is wrong -> abstain, not wrong -> correct.")

    OUT.write_text(json.dumps({
        "label": "cardinality type population and the occurrence-path confusion",
        "by_type": dict(by_type),
        "wrong_by_type": dict(wrong_by_type),
        "confusion": confusion,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
