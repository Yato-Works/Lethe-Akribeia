"""0-LLM pricing of a generic conflict-surfacing detector (V4 4.24).

Measured diagnosis (4.23/4.24): contradiction_handling questions reach
gold group 0 (the two planted values) in every compiled context, but
gold group 1 (a conflict marker word) NEVER appears in any context nor
in any player answer, so all seven players fail the category 0/20.
The reader never surfaces a conflict it was never shown.

Before building any conflict annotation into a player, this prices a
generic deterministic detector: two turns asserting DIFFERENT values
for the same (entity, property) pair, extracted with a fixed generic
relational lexicon ("runs|was|is|uses|pinned|set" + preposition).

Boundary: the detector reads ONLY conversation turns.  The scorer's
ground_truth and CONFLICT_MARKERS vocabulary are read nowhere in
player code; this analysis script may read them because it stays on
the measurement side (like the probe scripts).

Output: JSON artifact with detection stats + artifacts dir path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from artificial_memory.research.benchmarks.arena import (  # noqa: E402
    build_dataset,
    CONFLICT_MARKERS,
)

# The player-side detector is the single source of truth: this pricing script
# measures exactly the code the player would run.
from am_current_player import detect_conflicts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=REPO / "benchmark_results" / "am_current"
                    / "conflict_pricing.json")
    args = ap.parse_args()

    dataset = build_dataset()
    all_turns = []
    for scenario in dataset.scenarios:
        for turn in scenario.turns:
            all_turns.append((scenario.scenario_id, str(scenario.day),
                              turn.speaker, turn.content))

    conflicts = detect_conflicts(all_turns)

    # Annotation would fire on a question when its text names the conflict's
    # entity AND property.  Price that per category.
    per_cat_annot: dict[str, int] = defaultdict(int)
    per_cat_n: dict[str, int] = defaultdict(int)
    contradiction_gold_ok = contradiction_total = 0
    samples = []
    for question in dataset.questions:
        cat = question.category.value
        per_cat_n[cat] += 1
        qlow = question.question.lower()
        fired = []
        for (proj, prop) in conflicts:
            if proj != "(none)" and proj in qlow and prop.rstrip(" .") in qlow:
                fired.append(proj)
        if fired:
            per_cat_annot[cat] += 1
            if cat == "contradiction_handling":
                samples.append({"question_id": question.question_id,
                                "keys": fired})
        if cat == "contradiction_handling":
            contradiction_total += 1
            # does the detector capture the planted value pair?
            gold_values = question.ground_truth[0]
            have = set()
            for proj, prop in conflicts:
                if proj in qlow and prop.rstrip(" .") in qlow:
                    have |= {v for v, _ in conflicts[(proj, prop)]}
            vals = {str(v) for v in gold_values}
            if vals and vals.issubset({str(h) for h in have}):
                contradiction_gold_ok += 1

    print(f"assertion keys with >1 distinct value: {len(conflicts)}")
    print(f"contradiction: detector captures planted value pair on "
          f"{contradiction_gold_ok}/{contradiction_total}")
    print("\ntraining-free annotation would fire on (per category):")
    for cat in sorted(per_cat_n):
        print(f"  {cat:<30} {per_cat_annot[cat]:>3}/{per_cat_n[cat]}")

    example = next(((proj, prop, vals) for (proj, prop), vals in conflicts.items()
                    if any(cat == "contradiction_handling" for cat in per_cat_n)
                    and proj != "(none)"), None)
    if example:
        proj, prop, vals = example
        print(f"\nexample annotation (for {proj} / {prop}):")
        print(f"  NOTE: conflicting reports for project {proj} ({prop}): "
              f"{vals[0][0]} vs {vals[-1][0]}; which is correct is unverified.")
    print(f"\n(excluding scorer-side CONFLICT_MARKERS, length {len(CONFLICT_MARKERS)},"
          f" read nowhere in player code)")

    args.output.write_text(json.dumps({
        "detector": "generic relational extractor (fixed lexicon)",
        "keys_with_distinct_values": len(conflicts),
        "contradiction_planted_pair_captured":
            f"{contradiction_gold_ok}/{contradiction_total}",
        "annotation_fires_per_category": {
            cat: {"fired": per_cat_annot[cat], "n": per_cat_n[cat]}
            for cat in sorted(per_cat_n)},
        "sample_fired_contradictions": samples[:5],
        "conflict_keys_sample": [
            {"key": list(k), "values": sorted({v for v, _ in v_})[:4]}
            for k, v_ in sorted(conflicts.items())[:40]],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
