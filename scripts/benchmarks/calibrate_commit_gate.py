"""Calibrate the gate's event-overlap threshold against real data, both directions.

The gate ships EVENT_OVERLAP_MIN = 0.34 in a docstring claiming it rejects all six
measured over-commits AND still admits the engine's 37 existing correct date answers.
That claim is unverified until it is measured, and an unverified threshold is exactly
the kind of number that gets tuned to the test set.

This sweeps the threshold over the real frozen data and reports, at each value:

  false_accepts  over-commits the gate still lets through   (must reach 0)
  false_rejects  correct answers the gate now blocks        (the cost)
  net            the trade at that threshold

The claim in the module docstring is then checked against the sweep, and any
disagreement is reported rather than quietly patched.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory.skills.autonomous_engines.temporal_algebra_engine import (  # noqa: E402
    TemporalAlgebraEngine,
)
from artificial_memory.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)
from artificial_memory.skills.evidence_commit_gate import EvidenceCommitGate  # noqa

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


def build_set(label: str, engine_classes: tuple[str, ...],
              gold_in_context: bool | None) -> list[dict]:
    data = json.loads(MAP.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    out = []
    for r in data["records"]:
        if r["engine_class"] not in engine_classes:
            continue
        if gold_in_context is not None and r["gold_in_context"] is not gold_in_context:
            continue
        context = (cache.get(r["cache_qid"]) or {}).get("context") or ""
        if not context:
            continue
        live = DeterministicMemoryEngine.resolve(r["question"], context)
        if not live.used:
            continue
        turns = parse_turns(context)
        out.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "gold": r["gold"],
            "answer": live.answer,
            "evidence_turn": getattr(live, "evidence_turn", "") or "",
            "turns": turns,
            "person": TemporalAlgebraEngine._extract_person(r["question"], turns),
            "label": label,
        })
    return out


def main() -> int:
    # NEGATIVES: the 6 measured over-commits.
    negatives = build_set("over_commit",
                          ("ENGINE_WRONG_EVIDENCE_ABSENT",), None)
    # POSITIVES: answers the engine gets right today and must keep.
    positives = build_set("correct", ("ENGINE_CORRECT",), True)
    print(f"over-commits to reject : {len(negatives)}")
    print(f"correct answers to keep: {len(positives)}")
    print()

    # Find each case's evidence turn id so check 5 can exclude it.
    def ev_turn(case: dict):
        frag = str(case["evidence_turn"]).strip()[:40]
        for t in case["turns"]:
            if frag and frag in t.text:
                return t
        return None

    def ev_id(case: dict) -> str:
        t = ev_turn(case)
        return t.dia_id if t else ""

    def ev_speaker(case: dict) -> str:
        t = ev_turn(case)
        return t.speaker if t else ""

    print(f"{'threshold':>10} {'false_accepts':>15} {'false_rejects':>15} "
          f"{'negatives rejected':>19}")
    print("-" * 64)
    sweep = {}
    for th in (0.0, 0.10, 0.20, 0.25, 0.30, 0.34, 0.40, 0.50, 0.60):
        gate = EvidenceCommitGate(event_overlap_min=th)
        fa = fr = 0
        for c in negatives:
            d = gate.check(c["question"], c["answer"], c["evidence_turn"],
                           evidence_turn_id=ev_id(c), person=c["person"],
                           speaker=ev_speaker(c), all_turns=c["turns"])
            if d.allowed:
                fa += 1
        for c in positives:
            d = gate.check(c["question"], c["answer"], c["evidence_turn"],
                           evidence_turn_id=ev_id(c), person=c["person"],
                           speaker=ev_speaker(c), all_turns=c["turns"])
            if not d.allowed:
                fr += 1
        sweep[th] = (fa, fr)
        print(f"{th:>10.2f} {fa:>15} {fr:>15} {len(negatives) - fa:>19}")

    print()
    claimed = 0.34
    fa_c, fr_c = sweep[claimed]
    print(f"CLAIMED threshold {claimed}:")
    print(f"  over-commits still allowed : {fa_c}   (docstring claims 0)")
    print(f"  correct answers blocked    : {fr_c}   (docstring implies 0)")
    if fa_c == 0 and fr_c == 0:
        print("  --> claim HOLDS")
    elif fa_c == 0:
        print(f"  --> PARTIAL: rejects all over-commits but blocks {fr_c} correct "
              f"answers.  The docstring's implied claim of zero collateral damage "
              f"does NOT hold and must be corrected.")
    else:
        print("  --> claim DOES NOT HOLD; the docstring number must be corrected")

    # Which threshold is actually the lowest that rejects all negatives
    # WITHOUT collateral damage?
    viable = [th for th, (fa, fr) in sweep.items() if fa == 0]
    if viable:
        best = min(viable)
        print(f"\nlowest threshold rejecting ALL over-commits : {best}")
        print(f"  its cost on correct answers               : {sweep[best][1]}")

    gate = EvidenceCommitGate(event_overlap_min=claimed)
    print(f"\nper-negative reason at threshold {claimed}:")
    for c in negatives:
        d = gate.check(c["question"], c["answer"], c["evidence_turn"],
                       evidence_turn_id=ev_id(c), person=c["person"],
                       speaker=ev_speaker(c), all_turns=c["turns"])
        print(f"  {c['question_id']:<17} allowed={d.allowed!s:<5} {d.reason[:70]}")
    print(f"\nper-positive reason at threshold {claimed} (must all be allowed):")
    blocked = 0
    for c in positives:
        d = gate.check(c["question"], c["answer"], c["evidence_turn"],
                       evidence_turn_id=ev_id(c), person=c["person"],
                       speaker=ev_speaker(c), all_turns=c["turns"])
        if not d.allowed:
            blocked += 1
            print(f"  BLOCKED {c['question_id']:<17} {d.reason[:64]}")
    print(f"  correct answers blocked: {blocked}/{len(positives)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
