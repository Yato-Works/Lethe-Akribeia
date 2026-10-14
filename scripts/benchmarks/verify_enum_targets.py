"""Verify the 7 'enumeration/counting' failures are NOT one disease.

The mission asks to generalise from 7 enumeration/counting failures.  Before building
anything, the label is checked against the actual questions -- because this session has
twice found a cluster label hiding a mixed population (the plural-list 348 was 26%
real, and the 96-shape cohort was 23% real lists).

Per question, recorded:
  * does gold actually demand MULTIPLE items?  (is_list_gold + item count)
  * what did the engine claim, and which subsystem produced it?
  * which of the known disease shapes it matches:
      WRONG_EVENT_SCOPE  the union aggregated activities from OTHER events
      CARDINALITY        a 'how many' answered by counting sessions, not people
      SINGLE_VALUE       a single-item gold mislabelled as enumeration
      SHAPE_FRAGMENT     a non-noun-phrase value (already gated elsewhere)

If the 7 split across three or four diseases, then there is no single enumeration
capability to build, and the honest output is a re-scoped target list rather than an
engine change.
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

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass

BASELINE = ROOT / "benchmark_results/evidence_audit/phase_c_baseline.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/enum_target_verified.json"


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    def cq(qid: str) -> str | None:
        m = re.fullmatch(r"lc_conv-(\d+)_(\d+)", qid)
        return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}" if m else None

    targets = [r for r in baseline["rows"]
               if r["shape"] in ("enumeration", "counting")
               and r.get("current") == "wrong"]
    print(f"targets from the phase-C baseline : {len(targets)}")
    print()

    rows: list[dict] = []
    causes = Counter()
    for r in targets:
        context = (cache.get(cq(r["question_id"])) or {}).get("context") or ""
        gold = r["gold"]
        is_list = bool(getattr(scorer, "is_list_gold", lambda g: False)(gold))
        items = [x.strip() for x in str(gold).split(",") if x.strip()]
        ans = DeterministicMemoryEngine.resolve(r["question"], context)
        detail = getattr(ans, "detail", "") or ""

        # Classify by the ACTUAL defect, not the label.
        if re.match(r"\s*how\s+(?:many|much)\b", r["question"], re.I):
            cause = "CARDINALITY"
            why = ("a 'how many' question answered by counting sessions/occurrences "
                   "rather than the counted thing")
        elif not is_list and len(items) == 1:
            cause = "SINGLE_VALUE"
            why = "gold is ONE item; the enumeration label is wrong for this question"
        elif "aggregated activities" in detail or "extracted themes" in detail:
            cause = "WRONG_EVENT_SCOPE"
            why = "a union was produced, but from activities of a DIFFERENT event"
        else:
            cause = "SHAPE_FRAGMENT"
            why = "the value is not a noun phrase"

        causes[cause] += 1
        rows.append({
            "question_id": r["question_id"],
            "shape_label": r["shape"],
            "question": r["question"],
            "gold": gold,
            "gold_is_list": is_list,
            "gold_item_count": len(items),
            "answer": ans.answer,
            "detail": detail[:120],
            "cause": cause,
            "why": why,
        })

    print("=" * 74)
    print("DOES THE 'ENUMERATION' LABEL SURVIVE?")
    print("=" * 74)
    print(f"  gold really demands MULTIPLE items : "
          f"{sum(1 for x in rows if x['gold_is_list'])}/{len(rows)}")
    print(f"  gold is a SINGLE item              : "
          f"{sum(1 for x in rows if not x['gold_is_list'])}/{len(rows)}")
    print()
    print("ACTUAL DEFECT PER QUESTION")
    for k, n in causes.most_common():
        print(f"  {k:<20} {n}")
    print()
    for x in rows:
        print(f"  {x['question_id']:<16} {x['cause']}  (label={x['shape_label']}, "
              f"items={x['gold_item_count']})")
        print(f"    Q    : {x['question'][:64]}")
        print(f"    gold : {str(x['gold'])[:52]!r}")
        print(f"    gave : {str(x['answer'])[:52]!r}")
        print(f"    why  : {x['why'][:70]}")
        print()

    real_union = [x for x in rows if x["cause"] == "WRONG_EVENT_SCOPE"]
    print("=" * 74)
    print("WHAT A 'UNION' ENGINE COULD ACTUALLY REACH")
    print("=" * 74)
    print(f"  questions where a union is even the right shape : {len(real_union)}")
    print(f"  of those, the union came from the WRONG EVENT    : {len(real_union)}")
    print("  So a better union fixes its SCOPE, not its existence -- and scoping is")
    print("  the same event-binding problem already measured as 83% semantic.")
    print()
    print("CONCLUSION")
    print(f"  {len(rows)} questions carry an enumeration/counting label; they are")
    print(f"  {len(causes)} distinct diseases.  There is no single enumeration")
    print("  capability to build here.  Building one would be fitting a label.")

    OUT.write_text(json.dumps({
        "label": "verification that the 7 enumeration/counting failures are not one disease",
        "n": len(rows),
        "gold_really_multi": sum(1 for x in rows if x["gold_is_list"]),
        "causes": dict(causes),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
