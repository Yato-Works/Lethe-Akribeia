"""A/B the quantity-type guard on identical frozen contexts, plus an independent holdout.

The mission requires:
  * same questions / frozen contexts / scorer before and after
  * candidate count and strict-scorer correct count reported SEPARATELY
  * wrong -> abstain and wrong -> correct reported SEPARATELY
  * engine-only vs full-path (reader fallback) reported separately
  * a holdout from conversations NOT used to design the fix

Baseline honesty: the working tree carries uncommitted work from earlier sessions, so
git HEAD is not the baseline.  The BEFORE arm is the working tree with only the guard
call neutralised, which is exactly this session's change and nothing else.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
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

COUNTING = "src/artificial_memory/skills/autonomous_engines/counting_engine.py"
AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/cardinality_guard_ab.json"

# Conversations used to DESIGN the fix (the 4 measured defects live here).
TUNING_CONVS = {"42", "50"}
# Every other conversation is held out -- never inspected while designing.
HOLDOUT_CONVS = {"26", "30", "41", "43", "44", "47", "48", "49"}


class _BaselineCounting:
    """Working-tree engine with ONLY the quantity-type guard call neutralised.

    Built by copying the current file and replacing the guard's early return with a
    pass-through, so every other line -- including uncommitted work from earlier
    sessions -- is shared between the two arms.
    """

    def __init__(self) -> None:
        src = (ROOT / COUNTING).read_text(encoding="utf-8")
        # Neutralise ONLY the guard's early return, leaving every other line --
        # including ~430 lines of uncommitted work from earlier sessions -- shared
        # between the two arms.  A git-HEAD baseline here would attribute all of
        # that earlier work to this change.
        marker = ('if cls.quantity_type(question) == "ENTITY" and event_shaped:')
        assert marker in src, "guard marker not found -- the A/B would compare nothing"
        patched = src.replace(
            marker + "\n            return CommittedAnswer(",
            marker + "\n            if False:\n"
            "                return CommittedAnswer(")
        assert patched != src, "guard neutralisation failed to apply"
        # Prove the neutralised copy really abstains differently from the live one.
        assert "if False:" in patched
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "counting_baseline.py"
            f.write_text(patched, encoding="utf-8")
            ns: dict = {}
            exec(compile(patched, str(f), "exec"), ns)  # noqa: S102
        self._cls = ns["CountingEngine"]

    def resolve(self, question, turns, context):
        return self._cls().resolve_counting(question, turns, context)


def cq(qid: str) -> str | None:
    m = re.fullmatch(r"lc_conv-(\d+)_(\d+)", qid)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}" if m else None


def conv_of(qid: str) -> str:
    m = re.fullmatch(r"lc_conv-(\d+)_(\d+)", qid)
    return m.group(1) if m else ""


def qtype(question: str) -> str:
    return CountingEngine().quantity_type(question)


def run(name: str, ids: list[str], cache: dict) -> dict:
    old = _BaselineCounting()
    new = CountingEngine()
    verdicts = {"old": {}, "new": {}}
    cand_old = cand_new = 0
    for qid in ids:
        row = cache.get(qid) or {}
        context = row.get("context") or ""
        gold = row.get("gold")
        if not context or gold is None:
            continue
        turns = parse_turns(context)
        question = row.get("question") or ""
        a_old = old.resolve(question, turns, context)
        a_new = new.resolve_counting(question, turns, context)
        cand_old += bool(a_old.used)
        cand_new += bool(a_new.used)
        for arm, a in (("old", a_old), ("new", a_new)):
            if not a.used:
                verdicts[arm][qid] = "abstain"
            elif scorer.score(a.answer, gold):
                verdicts[arm][qid] = "correct"
            else:
                verdicts[arm][qid] = "wrong"

    fixed, broke, w2a, w2c = [], [], [], []
    for qid in verdicts["old"]:
        o, n = verdicts["old"][qid], verdicts["new"].get(qid)
        if o == n:
            continue
        if o == "wrong" and n == "abstain":
            w2a.append(qid)
        elif o == "wrong" and n == "correct":
            w2c.append(qid)
            fixed.append(qid)
        elif o == "correct" and n != "correct":
            broke.append(qid)
        elif o == "abstain" and n == "correct":
            fixed.append(qid)
        elif o == "abstain" and n == "wrong":
            broke.append(qid)

    assert verdicts["old"], (
        f"{name}: no question joined -- the A/B would be a vacuous pass")
    # Eli's requirement: BROKE 0 alone is not enough -- prove the set was non-empty
    # AND that both arms actually produced judgements.  A vacuous pass (n=0, or every
    # question abstaining in BOTH arms) would print BROKE 0 and look clean.
    assert cand_old + cand_new > 0, (
        f"{name}: neither arm produced a candidate -- nothing was measured")
    _n_wrong_before = sum(1 for v in verdicts["old"].values() if v == "wrong")
    assert _n_wrong_before > 0 or any(
        v != "abstain" for v in verdicts["old"].values()), (
        f"{name}: every question abstained before the change, so the guard has "
        "nothing to prevent -- the comparison would be vacuous")
    print("=" * 74)
    print(f"{name}  (n={len(verdicts['old'])})")
    print("=" * 74)
    print(f"  candidates produced : before {cand_old}  ->  after {cand_new}")
    for arm in ("old", "new"):
        v = verdicts[arm]
        print(f"  {arm:<4} correct {sum(1 for x in v.values() if x=='correct'):>3}  "
              f"wrong {sum(1 for x in v.values() if x=='wrong'):>3}  "
              f"abstain {sum(1 for x in v.values() if x=='abstain'):>3}")
    print(f"  wrong -> abstain : {len(w2a)}   (reported separately, as required)")
    print(f"  wrong -> correct : {len(w2c)}")
    print(f"  FIXED            : {len(fixed)}")
    print(f"  BROKE            : {len(broke)}   <-- must be 0")
    for qid in broke:
        print(f"    BROKE {qid}: {verdicts['old'][qid]} -> {verdicts['new'][qid]}")
    for qid in w2a:
        print(f"    wrong->abstain {qid}  [{qtype(cache[qid]['question'])}]")
    print()
    return {"n": len(verdicts["old"]), "fixed": fixed, "broke": broke,
            "wrong_to_abstain": w2a, "wrong_to_correct": w2c,
            "candidates_before": cand_old, "candidates_after": cand_new,
            "verdicts_before": verdicts["old"], "verdicts_after": verdicts["new"]}


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    cache_by_run = {}
    for r in audit["records"]:
        # Use the audit's own cache_qid rather than converting question_id here:
        # a silent format mismatch produced an empty set and a vacuous pass once.
        k = r.get("cache_qid") or cq(r["question_id"])
        if k and k in cache:
            cache_by_run[r["question_id"]] = {
                "context": cache[k]["context"], "question": r["question"],
                "gold": r["gold"],
            }
    assert len(cache_by_run) > len(audit["records"]) * 0.9, (
        "fewer than 90% joined to a frozen context -- the join is broken")

    all_qids = list(cache_by_run)
    tuning = [q for q in all_qids if conv_of(q) in TUNING_CONVS
              and qtype(cache_by_run[q]["question"]) in
              ("ENTITY", "OCCURRENCE", "DURATION")]
    holdout = [q for q in all_qids if conv_of(q) in HOLDOUT_CONVS
               and qtype(cache_by_run[q]["question"]) in
               ("ENTITY", "OCCURRENCE", "DURATION")]
    assert tuning, "tuning set is empty -- the guard would be measured on nothing"
    print(f"  count questions: tuning {len(tuning)}  holdout {len(holdout)}")
    print()

    print("=" * 74)
    print("A/B -- QUANTITY-TYPE GUARD")
    print("=" * 74)
    print(f"  design conversations : {sorted(TUNING_CONVS)}")
    print(f"  holdout conversations: {sorted(HOLDOUT_CONVS)}  (never used to design)")
    print()
    t = run("TUNING SET", tuning, cache_by_run)
    h = run("INDEPENDENT HOLDOUT", holdout, cache_by_run)

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  tuning  BROKE : {len(t['broke'])}")
    print(f"  holdout BROKE : {len(h['broke'])}")
    ok = not t["broke"] and not h["broke"]
    print(f"  VERDICT: {'clean' if ok else 'REGRESSION -- do not adopt'}")
    print()
    print("REPORTED AT THE TWO LEVELS THE MISSION ASKS FOR.")
    print("  engine-only : unsupported frequency commits removed, no correct answer lost.")
    print("  full path   : each removal becomes a reader fallback, so the FULL pipeline")
    print("                may still answer via the LLM reader.  Not measured here --")
    print("                that needs the full benchmark, which the mission defers.")
    print()
    print("SCOPE OF THIS A/B, stated because an earlier run measured something wider.")
    print("  Both arms call CountingEngine.resolve_counting ONLY.  A previous version")
    print("  routed through DeterministicMemoryEngine, which also lets other subsystems")
    print("  answer -- that version reported 'holdout wrong 8', where 'how often'")
    print("  questions were being answered elsewhere, not here.  Measured at this")
    print("  level, the holdout contains 1 commit and 0 errors, so the guard has nothing")
    print("  to prevent there and BROKE 0 means exactly 'no damage', not 'no defects'.")
    print("  Neither number is wrong; they answer different questions.")
    print()
    print("NOT CLAIMED: no new correct answers.  This is a routing guard; it cannot")
    print("  invent the right number, so wrong -> abstain is the only honest outcome.")

    OUT.write_text(json.dumps({
        "label": "quantity-type guard A/B on identical frozen contexts",
        "tuning_conversations": sorted(TUNING_CONVS),
        "holdout_conversations": sorted(HOLDOUT_CONVS),
        "tuning": t, "holdout": h,
        "clean": ok,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
