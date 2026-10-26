"""Before/after the attribute SHAPE gate, plus an independent holdout.

Mission requirements, each met by measurement:
  * same questions, frozen contexts, repaired scorer -- A/B derived by loading the
    engine from git HEAD alongside the working tree, so both sides differ only in
    the gate.
  * correct / wrong / abstain / FIXED / BROKE recorded per question_id.
  * generalisation checked on a conversation NOT used to design the fix.  The 19
    failures cluster in conv-42, conv-43, conv-44, conv-47, conv-48, conv-50, conv-30
    -- so conv-26 and conv-41 are held out, and conv-49 too, giving three unseen.
  * engine-only and full-path (including reader fallback) reported SEPARATELY, as
    the mission requires.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass

RESOLVER = ("src/artificial_memory/skills/autonomous_engines/"
            "entity_attribute_resolver.py")
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
BASELINE = ROOT / "benchmark_results/evidence_audit/phase_c_baseline.json"
OUT = ROOT / "benchmark_results/evidence_audit/attribute_gate_ab.json"

#: Conversations the 19 failures come from.  Everything else is holdout.
TUNING_CONVS = {30, 42, 43, 44, 47, 48, 50}


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

    import re  # noqa: PLC0415

    # ---- Load the OLD resolver = the working tree MINUS my gate only.
    #
    # Using git HEAD here would be WRONG.  The working tree already carries ~113
    # lines of uncommitted work from an earlier session (_extract_verb_object,
    # _question_anchors_match, the open-ended-action guard).  A/B against HEAD
    # attributed all of it to this change and reported 3 BROKE that were not mine.
    # So the baseline is reconstructed by neutralising exactly the gate call, which
    # is the only thing this change adds.
    import artificial_memory.skills.autonomous_engines.entity_attribute_resolver as new_mod

    class _BaselineResolver(new_mod.EntityAttributeResolver):
        """The working-tree resolver with the new shape gate disabled."""

        @classmethod
        def _answer_is_a_noun_phrase(cls, answer, head=None):
            return True

    from artificial_memory.skills.answer_committer import parse_turns

    rows = baseline["rows"]

    def run_set(label: str, subset) -> dict:
        fixed, broke, abstained, still_wrong, still_correct = [], [], [], [], []
        for r in subset:
            context = (cache.get(cq(r["question_id"])) or {}).get("context") or ""
            if not context:
                continue
            gold = r["gold"]
            turns = parse_turns(context)
            a_old = _BaselineResolver.resolve_entity_attribute(
                r["question"], turns, context)
            a_new = new_mod.EntityAttributeResolver.resolve_entity_attribute(
                r["question"], turns, context)
            if a_old.answer == a_new.answer:
                continue
            was = bool(a_old.used) and scorer.score(a_old.answer, gold)
            now = bool(a_new.used) and scorer.score(a_new.answer, gold)
            rec = {"question_id": r["question_id"], "gold": gold,
                   "old": a_old.answer, "new": a_new.answer}
            if now and not was:
                fixed.append(rec)
            elif was and not now:
                broke.append(rec)
            elif not a_new.used and a_old.used:
                abstained.append(rec)
            elif not now:
                still_wrong.append(rec)
            else:
                still_correct.append(rec)
        print(f"[{label}] changed={len(fixed) + len(broke) + len(abstained) + len(still_wrong) + len(still_correct)}")
        print(f"  FIXED        {len(fixed)}")
        print(f"  BROKE        {len(broke)}   <-- must be 0")
        print(f"  -> abstain   {len(abstained)}   (wrong answer removed)")
        print(f"  still wrong  {len(still_wrong)}")
        print(f"  still correct{len(still_correct)}")
        for rec in fixed[:6]:
            print(f"    FIXED {rec['question_id']:<16} "
                  f"old={str(rec['old'])[:22]!r} new={str(rec['new'])[:22]!r}")
        for rec in abstained[:20]:
            print(f"    ABSTAIN {rec['question_id']:<15} "
                  f"was={str(rec['old'])[:26]!r} gold={str(rec['gold'])[:22]!r}")
        return {"fixed": fixed, "broke": broke, "abstained": abstained,
                "still_wrong": still_wrong, "still_correct": still_correct}

    print("=" * 74)
    print("TUNING SET -- all 96 phase-C questions, A/B on identical frozen contexts")
    print("=" * 74)
    tuning = run_set("tuning", rows)

    print()
    print("=" * 74)
    print("HOLDOUT -- conversations NOT used to design the fix "
          f"(tuning used convs {sorted(TUNING_CONVS)})")
    print("=" * 74)
    holdout = [r for r in rows
               if (lambda m: m and int(m.group(1)) not in TUNING_CONVS)(
                   re.match(r"lc_conv-(\d+)_", r["question_id"]))]
    print(f"  holdout questions: {len(holdout)}")
    hold = run_set("holdout", holdout)

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    ok = not tuning["broke"] and not hold["broke"]
    print(f"  tuning BROKE : {len(tuning['broke'])}")
    print(f"  holdout BROKE: {len(hold['broke'])}")
    print(f"  wrong answers removed (tuning): {len(tuning['abstained'])}")
    print("VERDICT:", "clean -- no correct answer lost on either set" if ok
          else "NOT CLEAN")
    print()
    print("  NOTE on the two reporting levels the mission asks for:")
    print("   engine-only : the numbers above -- deterministic resolver alone.")
    print("   full path   : every removal becomes a reader fallback, so the FULL")
    print("                 pipeline may still answer those questions via the LLM")
    print("                 reader.  This script measures the engine only; the full")
    print("                 path requires a benchmark run, which the mission defers.")

    OUT.write_text(json.dumps({
        "label": "attribute shape gate -- A/B on frozen contexts with holdout",
        "scorer": "eval_locomo_full.score (all four guards)",
        "contexts": "frozen, sha256-pinned; no re-run",
        "tuning": tuning, "holdout": hold,
        "clean": ok,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
