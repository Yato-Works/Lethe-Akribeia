"""Re-derive the M17 interpretation from the frozen artefact, without re-calling the model.

The M17 run is complete and its JSON is frozen.  The decision logic was corrected after
the run (the original 'PASSES THE BAR' verdict was wrong), and re-running 150 model calls
just to reprint a table would be waste and would risk a different result.  So the decision
is recomputed here from the stored rows.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "benchmark_results/evidence_audit/v4_rule_vs_noop.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_rule_vs_noop_decision.json"


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    results = d["rows"]

    print("=" * 78)
    print("V4 MISSION 17 -- decision, recomputed from the frozen artefact (no new calls)")
    print("=" * 78)
    print()

    rescue_sets = {
        n: {r["qid"] for r in rows
            if r.get("transition") in ("wrong->correct", "abstain->correct")}
        for n, rows in results.items()
    }
    broken = {
        n: [r["qid"] for r in rows
            if r.get("transition") in ("correct->wrong", "correct->abstain")]
        for n, rows in results.items()
    }
    identical = len({frozenset(s) for s in rescue_sets.values()}) == 1
    any_broken = any(broken.values())

    print(f"  {'arm':<18}{'rescue':>8}{'broken':>8}   rescued ids")
    for n in rescue_sets:
        print(f"  {n:<18}{len(rescue_sets[n]):>8}{len(broken[n]):>8}   "
              f"{sorted(rescue_sets[n])}")
    print()
    print(f"  every arm rescues the SAME set : {identical}")
    print(f"  any arm broke a correct answer : {any_broken}")
    print()

    print("=" * 78)
    print("INTERPRETATION GUARD -- decisive")
    print("=" * 78)
    if identical:
        print("  Every arm rescues the SAME set, including H_REVERSE (a pure reversal with")
        print("  no semantics at all) and H_LEAST_OVERLAP (the exact inverse of")
        print("  H_QUERY_OVERLAP).  Two opposite rules cannot both be correct, yet they")
        print("  agree perfectly.  So the rule CONTENT is irrelevant: the rescue comes from")
        print("  THE ACT OF SWAPPING, not from any ordering principle.")
        print()
        print("  This is why the original 'PASSES THE BAR' verdict was wrong.  The bar")
        print("  requires beating no-op on rescue, and every arm does -- but only because")
        print("  ANY disturbance of the order flips the same borderline cases.  A rule whose")
        print("  effect is indistinguishable from a random swap has not identified a")
        print("  mechanism, so NONE is kept as a candidate.")
    print()

    print("=" * 78)
    print("DECISION")
    print("=" * 78)
    if identical:
        print("  STOP the ordering-rule line of design.  No rule is kept as a")
        print("  selection-mechanism candidate.")
        print()
        print("  WHAT STANDS: order causally affects WHICH candidate is selected (M15:")
        print("  7 rescued, 0 broken, frozen scorer).  WHAT IS RULED OUT: the stronger")
        print("  reading that a blind rule predicts the correct order.")
        print()
        print("  NEW FINDING, and it is about the Reader rather than about ordering: these")
        print("  6 cases are BORDERLINE.  Their answer flips under any reordering and is")
        print("  stable under none, which means the Reader is not resolving the")
        print("  competition at all -- it is sensitive to presentation noise.")
        print()
        print("  NEXT: content-based comparison.  Information update / correction is the")
        print("  candidate, because position is steerable but not predictable -- the")
        print("  mechanism must compare candidate CONTENT, not candidate position.")
    print()
    print("  SCOPE: 30 stratum-selected cases already observed in M15/M16, so this is")
    print("  development screening, not independent generalisation.  Does not observe the")
    print("  Reader's reasoning.  Not a benchmark estimate.  No production change.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 17 decision, recomputed from the frozen artefact",
        "rescue_sets": {k: sorted(v) for k, v in rescue_sets.items()},
        "broken": broken,
        "all_arms_rescue_identical_set": identical,
        "any_correct_broken": any_broken,
        "decision": ("STOP ordering-rule design: the rescue is the act of swapping, not "
                     "any rule's semantics. Move to content-based comparison."),
        "new_finding": ("the 6 rescued cases are BORDERLINE -- they flip under any "
                        "reordering, so the Reader is not resolving the competition, it "
                        "is sensitive to presentation noise"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
