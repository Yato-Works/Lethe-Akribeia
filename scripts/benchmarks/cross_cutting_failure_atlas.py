"""CROSS-CUTTING failure analysis -- where is the largest recoverable population?

Every prior audit in this project was scoped to a cluster someone had already named:
when_date, the quantity types, category 4.  This one starts from the whole benchmark and
asks which FAILURE SHAPE dominates, because that is the axis along which a single fix
can reach many questions.

For each of the 1,986 questions the current engine's outcome is combined with whether the
gold string is PRESENT in the frozen context.  That gives the axis the mission asked for:

  NO_EVIDENCE      gold not in context at all          -> nothing to extract
  EVIDENCE_PRESENT gold IS in the context verbatim     -> extraction/binding problem
  (plus the engine verdict, so the two can be crossed)

'gold in context' is a NECESSARY, not sufficient, condition for a deterministic fix, so
the number reported is a CEILING, not a forecast.  Anything that cannot be answered from
a substring of its own context is excluded from every candidate's denominator here.
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

from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory_gen.skills.autonomous_engines.counting_engine import (  # noqa: E402
    CountingEngine,
)
from artificial_memory_gen.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

_spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(_spec)
sys.modules["eval_locomo_full"] = scorer
try:
    _spec.loader.exec_module(scorer)
except SystemExit:
    pass

AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/cross_cutting_failure_atlas.json"

#: Question shapes, used to see whether a failure shape is one phenomenon or several.
_SHAPES = [
    ("WHEN_DATE", re.compile(r"\bwhen\b|\bwhat\s+(?:date|day|year)\b", re.I)),
    ("HOW_MANY", re.compile(r"\bhow\s+(?:many|much|long|often)\b", re.I)),
    ("WHAT_THING", re.compile(r"\bwhat\b", re.I)),
    ("WHO", re.compile(r"\bwho\b", re.I)),
    ("WHY", re.compile(r"\bwhy\b", re.I)),
    ("WHICH", re.compile(r"\bwhich\b", re.I)),
]


def shape_of(q: str) -> str:
    for name, rx in _SHAPES:
        if rx.search(q):
            return name
    return "OTHER"


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cache[json.loads(line)["qid"]] = json.loads(line)

    engine = CountingEngine()
    rows: list[dict] = []
    for r in audit["records"]:
        ctx = cache.get(r["cache_qid"]) or {}
        context = ctx.get("context") or ""
        if not context:
            continue
        gold = r["gold"]
        ans = DeterministicMemoryEngine.resolve(r["question"], context)
        verdict = ("abstain" if not ans.used
                   else ("correct" if scorer.score(ans.answer, gold) else "wrong"))

        # Is the gold recoverable from the context at all?
        g = str(gold).strip()
        low_ctx = context.lower()
        g_low = g.lower()
        verbatim = g_low in low_ctx
        # A gold like 'twice' is trivially present; require the gold's longest content
        # token so '2' or 'the' do not count as evidence.
        toks = [t for t in re.findall(r"[a-z0-9]{3,}", g_low)
                if t not in {"the", "and", "for", "was", "were", "his", "her"}]
        token_present = bool(toks) and all(t in low_ctx for t in toks)

        rows.append({
            "question_id": r["question_id"],
            "cache_qid": r["cache_qid"],
            "conversation": r["cache_qid"].split("-")[1],
            "category": r.get("category"),
            "shape": shape_of(r["question"]),
            "qtype": engine.quantity_type(r["question"]),
            "question": r["question"],
            "gold": gold,
            "verdict": verdict,
            "answer": ans.answer,
            "detail": str(ans.detail)[:130],
            "gold_verbatim_in_context": verbatim,
            "gold_tokens_in_context": token_present,
            "recoverable": bool(verbatim or token_present),
        })

    print("=" * 74)
    print("CROSS-CUTTING FAILURE ATLAS -- all questions with a frozen context")
    print("=" * 74)
    print(f"  scored: {len(rows)}")
    by_v = Counter(x["verdict"] for x in rows)
    print(f"  engine verdict: {dict(by_v)}")
    print()

    # ---- the mission's failure axis -----------------------------------------
    def bucket(x: dict) -> str:
        if x["verdict"] == "correct":
            return "ALREADY_CORRECT"
        if not x["recoverable"]:
            return "NO_EVIDENCE_IN_CONTEXT"
        return "EVIDENCE_PRESENT_NOT_ANSWERED"

    by_b = Counter(bucket(x) for x in rows)
    print("FAILURE AXIS (the mission's classification, measured)")
    for k in ("ALREADY_CORRECT", "EVIDENCE_PRESENT_NOT_ANSWERED",
              "NO_EVIDENCE_IN_CONTEXT"):
        print(f"  {k:<32} {by_b.get(k, 0)}")
    pool = [x for x in rows if bucket(x) == "EVIDENCE_PRESENT_NOT_ANSWERED"]
    print()
    print(f"  RECOVERABLE POOL (gold in context, engine did not answer correctly): "
          f"{len(pool)}")
    print(f"  independent conversations in that pool: "
          f"{len({x['conversation'] for x in pool})}")
    print()

    # ---- decompose the pool ------------------------------------------------
    print("THE POOL, by question shape")
    for k, v in Counter(x["shape"] for x in pool).most_common():
        print(f"  {k:<14} {v}")
    print()
    print("THE POOL, by engine detail (why it failed)")
    for k, v in Counter(x["detail"][:52] for x in pool).most_common(10):
        print(f"  {v:>4}  {k}")
    print()
    print("THE POOL, by verdict")
    for k, v in Counter(x["verdict"] for x in pool).most_common():
        print(f"  {k:<10} {v}")
    print()

    # ---- shape x verdict, the actionable table ------------------------------
    print("SHAPE x VERDICT (only shapes with >= 20 in the pool)")
    tab = Counter((x["shape"], x["verdict"]) for x in pool)
    shapes = [s for s, _ in Counter(x["shape"] for x in pool).most_common()]
    print(f"  {'shape':<14}{'wrong':>7}{'abstain':>9}{'total':>7}")
    for s in shapes:
        w, a = tab.get((s, "wrong"), 0), tab.get((s, "abstain"), 0)
        if w + a >= 20:
            print(f"  {s:<14}{w:>7}{a:>9}{w + a:>7}")
    print()

    # ---- the top sub-population, examined ----------------------------------
    top_shape = shapes[0]
    sub = [x for x in pool if x["shape"] == top_shape]
    print(f"LARGEST SUB-POPULATION: {top_shape}  (n={len(sub)}, "
          f"{len({x['conversation'] for x in sub})} conversations)")
    print("  sample, with the engine's stated reason:")
    for x in sub[:12]:
        print(f"    {x['question_id']:<17} gold={str(x['gold'])[:14]!r:<16} "
              f"{x['verdict']}")
        print(f"       Q: {x['question'][:60]}")
        print(f"       gave: {str(x['answer'])[:34]!r}  why: {x['detail'][:44]!r}")

    OUT.write_text(json.dumps({
        "label": "cross-cutting failure atlas over every question with a frozen context",
        "n_scored": len(rows),
        "by_verdict": dict(by_v),
        "by_bucket": dict(by_b),
        "recoverable_pool_n": len(pool),
        "recoverable_pool_conversations": len({x["conversation"] for x in pool}),
        "pool_by_shape": dict(Counter(x["shape"] for x in pool)),
        "pool_by_detail": dict(Counter(x["detail"][:52] for x in pool)),
        "pool_shape_x_verdict": {f"{a}|{b}": c for (a, b), c in tab.items()},
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
