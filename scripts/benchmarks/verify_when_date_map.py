"""Verify the when_date map by spot-checking, and size the 33-case abstain subset.

A map is a claim like any other.  build_when_date_map.py asserted structural
invariants (no double-counted id, every row classified, every reader claim has a
stored answer), but invariants cannot catch a WRONG LABEL.  So this verifies the
labels themselves against the frozen contexts.

Checks, each on real examples rather than on the aggregate:

  1  The 37 ENGINE_CORRECT -- print gold and the engine's answer side by side, and
     re-score independently.  A map that calls something correct when the scorer
     disagrees would inflate the engine's apparent ability.
  2  The 33 actionable abstentions -- for each, show the frozen-context turn that
     actually carries the gold string.  If the gold is genuinely in the context and
     the engine abstains, these are recoverable with no retrieval work.  If the
     'presence' check was too loose, the subset is fictional and the whole
     prioritisation built on it is fiction too.
  3  The abstain reasons -- grouped, so the fix targets the largest real cause.

Reported as counts plus per-case evidence, never as a bare number.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys_path = ROOT / "src"
import sys
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

import importlib.util  # noqa: E402
from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory_gen.skills.deterministic_memory_engine import (  # noqa: E402
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

MAP = ROOT / "benchmark_results/when_date_map/when_date_failure_map.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"


def carrying_turn(gold: str, context: str) -> str | None:
    """The turn text that actually contains the gold string."""
    g = str(gold).strip().strip('"').strip("'")
    if not g:
        return None
    for line in context.splitlines():
        low = line.lower()
        if g.lower() in low:
            return line.strip()[:170]
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)", g)
    if m:
        for line in context.splitlines():
            if re.search(rf"\b{re.escape(m.group(1))}\s+{re.escape(m.group(2))}\b",
                         line, re.I):
                return line.strip()[:170]
    return None


def main() -> int:
    data = json.loads(MAP.read_text(encoding="utf-8"))
    records = data["records"]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    print("=" * 76)
    print("VERIFY 1 -- the 37 ENGINE_CORRECT, re-scored independently")
    print("=" * 76)
    correct = [r for r in records if r["engine_class"] == "ENGINE_CORRECT"]
    mismatch = 0
    for r in correct[:10]:
        ok = scorer.score(r["engine_answer"], r["gold"])
        if not ok:
            mismatch += 1
        print(f"  {'OK ' if ok else 'BAD'} {r['question_id']:<17} "
              f"gold={str(r['gold'])[:26]!r:<28} engine={str(r['engine_answer'])[:26]!r}")
    for r in correct:
        if not scorer.score(r["engine_answer"], r["gold"]):
            mismatch += 1
    print(f"\n  re-scored all {len(correct)}: mismatches = {mismatch}")
    print("  --> 0 means the label is honest. Anything else inflates the engine.")
    print()

    print("=" * 76)
    print("VERIFY 2 -- the 33 actionable abstentions, with their carrying turn")
    print("=" * 76)
    actionable = [r for r in records
                  if r["engine_class"] == "ENGINE_ABSTAIN"
                  and r["gold_in_context"]]
    verified = 0
    for r in actionable:
        context = (cache.get(r["cache_qid"]) or {}).get("context") or ""
        turn = carrying_turn(r["gold"], context)
        if turn:
            verified += 1
    print(f"  claimed actionable : {len(actionable)}")
    print(f"  carrying turn found: {verified}")
    print(f"  --> equal means the gold really is in the frozen context, so these")
    print(f"      need no retrieval work.  A shortfall means the presence check was")
    print(f"      too loose and the subset is fictional.")
    print()
    for r in actionable[:8]:
        context = (cache.get(r["cache_qid"]) or {}).get("context") or ""
        turn = carrying_turn(r["gold"], context)
        print(f"  {r['question_id']}")
        print(f"    Q    : {r['question'][:62]}")
        print(f"    gold : {r['gold']!r}")
        print(f"    turn : {(turn or 'NOT FOUND')[:150]}")
        print(f"    why abstained: {r['engine_detail'][:80]}")
    print()

    print("=" * 76)
    print("VERIFY 3 -- abstain reasons, grouped (this names the fix)")
    print("=" * 76)
    abstains = [r for r in records if r["engine_class"] == "ENGINE_ABSTAIN"]
    reasons = Counter()
    for r in abstains:
        d = (r["engine_detail"] or "").strip()
        key = d[:58] if d else "(no detail)"
        reasons[key] += 1
    for k, n in reasons.most_common(12):
        print(f"  {n:>3}  {k}")
    print()
    print("  Splitting by whether gold is present:")
    with_gold = [r for r in abstains if r["gold_in_context"]]
    without = [r for r in abstains if not r["gold_in_context"]]
    print(f"    abstain, gold IN context     : {len(with_gold)}  <- fixable here")
    print(f"    abstain, gold NOT in context : {len(without)}  <- retrieval problem")
    print()

    print("=" * 76)
    print("PRIORITISATION -- by evidence, not by cluster size")
    print("=" * 76)
    wrong_present = [r for r in records
                     if r["engine_class"] == "ENGINE_WRONG_EVIDENCE_PRESENT"]
    wrong_absent = [r for r in records
                    if r["engine_class"] == "ENGINE_WRONG_EVIDENCE_ABSENT"]
    print(f"  A  abstain + gold in context      : {len(with_gold):>3}  evidence present,"
          f" no wrong-answer risk (abstention is already safe)")
    print(f"  B  wrong + gold in context        : {len(wrong_present):>3}  evidence present,"
          f" actively misleading -- highest harm per case")
    print(f"  C  wrong + gold absent            : {len(wrong_absent):>3}  needs retrieval first")
    print(f"  D  abstain + gold absent          : {len(without):>3}  needs retrieval first")
    print()
    print("  B is small but harmful; A is the biggest safe win.  Both are evidence-")
    print("  present, so neither needs retrieval work.  That is the shortlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
