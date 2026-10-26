"""V4 Mission 10, step 0 -- split the holdout BEFORE anything is measured or tuned.

The mission's constraint is explicit: the 989 questions must not be used to design AND
then be called an unused holdout.  So the split happens first, by CONVERSATION (not by
question), so no conversation leaks across the boundary.

M9 touched all 10 conversations, which means every conversation is 'seen' at the case
level -- but the split here is by conversation for the INTERVENTION that comes after, so
tuning and holdout share no dialogue at all.

TUNING  : conv-43, conv-47, conv-49, conv-50   (the conversations carrying most of the
          M9 signal, plus others, so the condition is measurable on both sides)
HOLDOUT : conv-26, conv-30, conv-41, conv-42, conv-44, conv-48   (never used to tune)

Deterministic seed, sorted, recorded with the exact ID lists.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_holdout_split.json"

TUNING_CONVS = {"43", "47", "49", "50"}
HOLDOUT_CONVS = {"26", "30", "41", "42", "44", "48"}


def main() -> int:
    h9 = json.loads(H9.read_text(encoding="utf-8"))
    rows = h9["rows"]

    tuning = sorted(r["question_id"] for r in rows
                    if r["conversation"].split("-")[1] in TUNING_CONVS)
    holdout = sorted(r["question_id"] for r in rows
                     if r["conversation"].split("-")[1] in HOLDOUT_CONVS)

    assert len(tuning) + len(holdout) == len(rows), "split lost questions"
    assert not (set(tuning) & set(holdout)), "a question is in both splits"
    t_convs = {r["conversation"].split("-")[1] for r in rows
               if r["question_id"] in set(tuning)}
    h_convs = {r["conversation"].split("-")[1] for r in rows
               if r["question_id"] in set(holdout)}
    assert not (t_convs & h_convs), "a CONVERSATION appears in both splits"

    print("=" * 74)
    print("V4 MISSION 10 step 0 -- holdout split, fixed BEFORE any measurement")
    print("=" * 74)
    print(f"  total questions : {len(rows)}")
    print(f"  TUNING          : {len(tuning)}  conversations {sorted(t_convs)}")
    print(f"  HOLDOUT         : {len(holdout)}  conversations {sorted(h_convs)}")
    print(f"  conversation overlap between splits : {sorted(t_convs & h_convs)}")
    print()

    from collections import Counter
    print("  per-split outcome mix (recorded now, before the condition is labelled)")
    for label, ids in (("TUNING", set(tuning)), ("HOLDOUT", set(holdout))):
        mix = Counter(r["mutually_exclusive"] for r in rows if r["question_id"] in ids)
        print(f"    {label}: {dict(mix)}")
    print()

    OUT.write_text(json.dumps({
        "label": "V4 mission 10 holdout split -- by conversation, fixed before measurement",
        "tuning_conversations": sorted(t_convs),
        "holdout_conversations": sorted(h_convs),
        "n_tuning": len(tuning), "n_holdout": len(holdout),
        "tuning_qids": tuning, "holdout_qids": holdout,
        "split_unit": "conversation (no dialogue shared across the boundary)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
