"""Regression check done RIGHT: live engine before vs after the fix.

The previous attempt reported 901 regressions and was void.  It compared the LIVE
engine's answer against the run's STORED is_correct flags -- but those flags were
written from the LLM reader's prediction, not the deterministic engine's.  Probe:
lc_conv-26_0 stores pred '7 May 2023' (correct) while the atlas records the
deterministic engine producing '8 May, 2023'.  Comparing the live engine to the
reader's flags measures nothing about the engine.

A valid before/after needs the SAME engine, only the code differing.  This loads
the engine from git HEAD (the fix reverted in-memory) and runs it against the
current engine on identical frozen contexts, scoring both with the same repaired
scorer.

Reported as (n, FIXED, BROKE, spurious) with the question_ids for every change.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ENGINE_FILE = ("src/artificial_memory/skills/autonomous_engines/"
               "temporal_algebra_engine.py")
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"


def main() -> int:
    # Load the CURRENT scorer once; both sides are scored with it.
    spec = importlib.util.spec_from_file_location(
        "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
    scorer = importlib.util.module_from_spec(spec)
    sys.modules["eval_locomo_full"] = scorer
    try:
        spec.loader.exec_module(scorer)
    except SystemExit:
        pass

    # Snapshot the pre-fix engine from git HEAD into a temp package so both
    # versions can be imported side by side in one process.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        head_src = subprocess.run(
            ["git", "show", f"HEAD:{ENGINE_FILE}"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
        print(f"loaded pre-fix engine from HEAD ({len(head_src)} chars)")
        (tmp / "temporal_algebra_engine.py").write_text(head_src, encoding="utf-8")

        import artificial_memory.skills.autonomous_engines.temporal_algebra_engine as cur
        sys.path.insert(0, str(tmp))
        old = importlib.import_module("temporal_algebra_engine")

        atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        cache = {}
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["qid"]] = row

        # Every question that is a date question AND has a frozen context.  This is
        # the honest blast radius, not just the failure cluster.
        from artificial_memory.skills.answer_committer import parse_turns
        cohort = [
            r for r in atlas
            if cache.get(r.get("cache_qid"))
            and (cache[r["cache_qid"]].get("context") or "")
        ]
        print(f"cohort (all atlas questions with a frozen context): {len(cohort)}")

        scored = fixed = broke = 0
        fixed_ids: list[str] = []
        broke_ids: list[str] = []

        for row in cohort:
            context = cache[row["cache_qid"]]["context"]
            turns = parse_turns(context)
            gold = row["gold"]
            if not gold:
                continue
            a_old = old.TemporalAlgebraEngine.resolve_date(
                row["question"], turns, context)
            a_new = cur.TemporalAlgebraEngine.resolve_date(
                row["question"], turns, context)
            # Only questions where the two versions actually differ are of interest.
            if a_old.answer == a_new.answer:
                continue
            scored += 1
            was = bool(a_old.used) and scorer.score(a_old.answer, gold)
            now = bool(a_new.used) and scorer.score(a_new.answer, gold)
            if now and not was:
                fixed += 1
                fixed_ids.append(row["question_id"])
            elif was and not now:
                broke += 1
                broke_ids.append(row["question_id"])

        print()
        print(f"questions where the two versions DIFFER : {scored}")
        print(f"FIXED   (old wrong -> new right)       : {fixed}")
        print(f"BROKE   (old right -> new wrong)       : {broke}")
        if fixed_ids:
            print(f"\nFIXED ids: {fixed_ids[:40]}")
        if broke_ids:
            print(f"\nBROKE ids: {broke_ids[:40]}")
        print()
        print("VERDICT:", "clean -- the fix adds and breaks nothing"
              if broke == 0 else "NOT CLEAN -- see BROKE ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
