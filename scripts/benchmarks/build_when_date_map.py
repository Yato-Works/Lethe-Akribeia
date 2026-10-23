"""The honest when_date failure map -- built fresh, with every claim asserted.

Why this exists: the previous decomposition (decompose_when_date.py) classified
163 when_date failures into arithmetic / binding / relative / abstain using the
STORED predictions.  The rejected-branch diagnosis then proved those stored
predictions come from the LLM READER, not the deterministic engine.  So that map
described the reader's failures and called them the engine's -- the same
mis-attribution that made 22 phantom 'parsing errors'.

This map is built differently.  For every when_date failure it records, per path:

    reader  : the stored prediction, and whether the stored scorer says it was right
    engine  : the LIVE deterministic engine's answer, verdict, and abstain reason
    gold    : what the answer must be
    evidence: which frozen-context turns actually carry the gold string

Eli's new rule, enforced mechanically rather than promised:
  * no question_id is counted twice -- asserted,
  * every row names the PATH its prediction came from -- asserted,
  * evidence presence is checked against the frozen context, not inferred.

Output: benchmark_results/when_date_map/when_date_failure_map.json + a printed table.
Nothing here is a fix.  This is the map only.
"""
from __future__ import annotations

import hashlib
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

ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT_DIR = ROOT / "benchmark_results/when_date_map"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def gold_is_verbatim_in_context(gold: str, context: str) -> bool:
    """Is the gold string (or its core) actually written in the frozen context?"""
    g = str(gold).strip().strip('"').strip("'")
    if not g:
        return False
    low = context.lower()
    if g.lower() in low:
        return True
    # Gold may name a day+month while the context writes 'Month D'.
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)(?:,?\s+(20\d\d))?", g)
    if m and m.group(2).lower() in MONTHS:
        d, mon = m.group(1), m.group(2).lower()
        return bool(re.search(rf"\b{d}\s+{mon}\b", low)) or \
            bool(re.search(rf"\b{mon}\s+{d}(?:st|nd|rd|th)?\b", low))
    return False


def classify_engine(ans, gold: str, context: str) -> str:
    """Classify by what the ENGINE did, not by what the reader did."""
    if not ans.used:
        return "ENGINE_ABSTAIN"
    if scorer.score(ans.answer, gold):
        return "ENGINE_CORRECT"
    gold_in_ctx = gold_is_verbatim_in_context(gold, context)
    if gold_in_ctx:
        # The right answer is written in the context and the engine still got it
        # wrong -> a resolution error, not a retrieval one.
        return "ENGINE_WRONG_EVIDENCE_PRESENT"
    return "ENGINE_WRONG_EVIDENCE_ABSENT"


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    rows = [r for r in atlas if r.get("shape") == "when_date"]
    print(f"when_date rows in the atlas : {len(rows)}")

    records: list[dict] = []
    for row in rows:
        ctx_row = cache.get(row.get("cache_qid"))
        context = (ctx_row or {}).get("context") or ""
        rec: dict = {
            "question_id": row["question_id"],
            "cache_qid": row.get("cache_qid"),
            "question": row["question"],
            "gold": row["gold"],
            # ---- READER path (the stored prediction)
            "reader_prediction": row.get("predicted_answer"),
            "reader_claimed": bool(row.get("engine_claims")),
            "reader_stored_verdict": ("correct" if row.get("stored_is_correct")
                                      else "wrong"),
            "reader_rescored_verdict": ("correct" if row.get("rescored_correct")
                                        else "wrong"),
            # ---- ENGINE path (live, on the frozen context)
            "engine_answer": None,
            "engine_used": None,
            "engine_detail": None,
            "engine_class": None,
            "gold_in_context": None,
        }
        if context:
            turns = parse_turns(context)
            ans = DeterministicMemoryEngine.resolve(row["question"], context)
            rec["engine_answer"] = ans.answer
            rec["engine_used"] = bool(ans.used)
            rec["engine_detail"] = (ans.detail or "")[:140]
            rec["engine_class"] = classify_engine(ans, row["gold"], context)
            rec["gold_in_context"] = gold_is_verbatim_in_context(row["gold"], context)
        else:
            rec["engine_class"] = "NO_FROZEN_CONTEXT"
        records.append(rec)

    # ---- Eli's new rule, asserted not promised.
    ids = [r["question_id"] for r in records]
    dupes = [q for q, n in Counter(ids).items() if n > 1]
    assert not dupes, f"question_id counted more than once: {dupes[:5]}"
    assert len(ids) == len(set(ids)) == len(rows), "cohort size != unique ids"
    for r in records:
        assert r["reader_prediction"] is not None or not r["reader_claimed"], (
            f"{r['question_id']}: reader claims an answer but stores none")
        assert r["engine_class"] is not None, f"{r['question_id']}: unclassified"
    print(f"unique question_ids         : {len(set(ids))}  (asserted, no double-count)")

    # ---- The map.
    engine_map = Counter(r["engine_class"] for r in records)
    reader_map = Counter(r["reader_rescored_verdict"] for r in records)

    print()
    print("=" * 74)
    print("READER path -- what the LLM reader did (NOT the engine)")
    print("=" * 74)
    for k, n in reader_map.most_common():
        print(f"  {k:<28} {n}")
    print()
    print("=" * 74)
    print("ENGINE path -- what the deterministic engine does on the SAME questions")
    print("=" * 74)
    for k, n in engine_map.most_common():
        print(f"  {k:<32} {n}")
    print()

    # Where the two paths disagree, that is the real signal.
    agree = disagree = 0
    for r in records:
        reader_ok = r["reader_rescored_verdict"] == "correct"
        engine_ok = r["engine_class"] in ("ENGINE_CORRECT",)
        if reader_ok == engine_ok:
            agree += 1
        else:
            disagree += 1
    print(f"paths AGREE on the verdict   : {agree}")
    print(f"paths DISAGREE               : {disagree}")
    print()

    # The actionable subset: engine abstains AND gold is written in the context.
    recoverable = [r for r in records
                   if r["engine_class"] == "ENGINE_ABSTAIN"
                   and r["gold_in_context"]]
    print("=" * 74)
    print("ACTIONABLE SUBSET -- engine abstains while gold is IN the frozen context")
    print("=" * 74)
    print(f"  n = {len(recoverable)}   <- a fix needs NO retrieval work for these")
    for r in recoverable[:10]:
        print(f"    {r['question_id']:<18} gold={str(r['gold'])[:34]!r}")
    if len(recoverable) > 10:
        print(f"    ... {len(recoverable) - 10} more")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "when_date_failure_map.json").write_text(
        json.dumps({
            "label": "when_date failure map -- reader and engine paths separated",
            "evaluated_code": {
                "note": "the WORKING TREE, not git HEAD. HEAD's engine is 498 "
                        "lines and does not even import; the working copy is 638.",
                "engine": "src/artificial_memory/skills/autonomous_engines/"
                          "temporal_algebra_engine.py",
            },
            "frozen_context": {
                "file": "benchmark_results/locomo_context_cache_rules.jsonl",
                "rows": 1986,
                "sha256": hashlib.sha256(
                    (ROOT / "benchmark_results/locomo_context_cache_rules.jsonl")
                    .read_bytes()).hexdigest(),
            },
            "scorer": "eval_locomo_full.score (repaired: date + numeral guards)",
            "cohort_n": len(records),
            "invariants": {
                "no_double_counted_question_id": not dupes,
                "cohort_eq_unique_ids": len(ids) == len(set(ids)),
                "every_row_classified": all(r["engine_class"] for r in records),
            },
            "reader_verdicts": dict(reader_map),
            "engine_classes": dict(engine_map),
            "paths_agree": agree,
            "paths_disagree": disagree,
            "actionable_engine_abstain_gold_in_context": len(recoverable),
            "records": records,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT_DIR / 'when_date_failure_map.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
