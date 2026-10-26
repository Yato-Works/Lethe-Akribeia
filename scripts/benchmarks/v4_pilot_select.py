"""V4 Mission 4 -- stratified pilot selection.  EVALUATION ONLY.

Picks the 90-case pilot BEFORE any annotation exists, so the selection cannot be
influenced by what the annotation later finds:

  30 FAILURE_WRONG     from the 120
  30 FAILURE_ABSTAIN   from the 184
  30 CONTROL_CORRECT   from the 150

Stratification is by CONVERSATION and by QUESTION CATEGORY, proportionally within each
cohort, so no single conversation or category dominates the pilot.  Selection is
deterministic (fixed seed, sorted IDs within a stratum) so it is reproducible and
auditable.

The gold answer is deliberately NOT used as a sampling key, and low-scoring
conversations are not over-sampled -- both are prohibited by the mission.  Only the
cohort label, conversation and category drive the draw.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
RECL = ROOT / "benchmark_results/evidence_audit/v4_reader_h9_reclassified.json"
AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_pilot_selection.json"

PER_COHORT = 30
SEED = 20261015


def main() -> int:
    h9 = json.loads(H9.read_text(encoding="utf-8"))
    recl = json.loads(RECL.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    meta = {r["question_id"]: r for r in audit["records"]}
    arm = {r["question_id"]: r for r in h9["rows"]}

    cohorts = {
        "FAILURE_WRONG": sorted(set(recl["wrong_value_qids"])),
        "FAILURE_ABSTAIN": sorted(set(recl["abstain_qids"])),
        "CONTROL_CORRECT": sorted(
            q for q, r in arm.items() if r["mutually_exclusive"] == "CORRECT"),
    }

    print("=" * 74)
    print("V4 MISSION 4 -- stratified pilot selection (drawn before annotation)")
    print("=" * 74)
    for k, v in cohorts.items():
        print(f"  {k:<18} {len(v)}")
    print(f"  seed                 : {SEED}")
    print()

    rng = random.Random(SEED)
    selected: list[dict] = []
    for cohort, ids in cohorts.items():
        # stratum = (conversation, category)
        strata = defaultdict(list)
        for q in ids:
            r = arm.get(q) or {}
            cat = (meta.get(q) or {}).get("category")
            strata[(r.get("conversation"), cat)].append(q)
        want = PER_COHORT
        total = len(ids)
        # proportional allocation, largest remainder so the strata sum exactly
        alloc = {}
        for key, members in strata.items():
            alloc[key] = want * len(members) / total
        floors = {k: int(v) for k, v in alloc.items()}
        rem = want - sum(floors.values())
        for key in sorted(alloc, key=lambda k: (-(alloc[k] - floors[k]), k)):
            if rem <= 0:
                break
            floors[key] += 1
            rem -= 1

        picked: list[str] = []
        for key in sorted(strata, key=lambda k: (str(k[0]), str(k[1]))):
            members = sorted(strata[key])
            k = min(floors.get(key, 0), len(members))
            if k:
                picked += rng.sample(members, k)
        # If rounding left a shortfall, top up deterministically from the largest
        # strata rather than dropping below the required cohort size.
        if len(picked) < want:
            rest = sorted(set(ids) - set(picked))
            picked += rest[: want - len(picked)]
        picked = sorted(set(picked))[:want]

        for q in picked:
            r = arm.get(q) or {}
            selected.append({
                "question_id": q,
                "cohort": cohort,
                "conversation": r.get("conversation"),
                "category": (meta.get(q) or {}).get("category"),
                "question": (meta.get(q) or {}).get("question", ""),
                "A_verdict": r.get("mutually_exclusive"),
                # gold is recorded for the EVALUATION path only and must never reach
                # the annotator or the Reader prompt.
                "gold_eval_only": (meta.get(q) or {}).get("gold"),
            })

    n = len(selected)
    assert n == 3 * PER_COHORT, f"expected 90, drew {n}"
    ids = [s["question_id"] for s in selected]
    assert len(set(ids)) == n, "duplicate question ID in the pilot"
    for cohort in cohorts:
        c = sum(1 for s in selected if s["cohort"] == cohort)
        assert c == PER_COHORT, f"{cohort}: {c} != {PER_COHORT}"

    print("SELECTED")
    print(f"  total                : {n}")
    print(f"  duplicates           : 0")
    print()
    print("  by cohort x conversation")
    tab = Counter((s["cohort"], s["conversation"]) for s in selected)
    convs = sorted({s["conversation"] for s in selected})
    print(f"    {'cohort':<18}" + "".join(f"{c.replace('conv-', 'c'):>5}"
                                          for c in convs))
    for cohort in cohorts:
        row = "".join(f"{tab.get((cohort, c), 0):>5}" for c in convs)
        print(f"    {cohort:<18}{row}")
    print()
    print("  by cohort x category")
    tab2 = Counter((s["cohort"], s["category"]) for s in selected)
    cats = sorted({s["category"] for s in selected})
    print(f"    {'cohort':<18}" + "".join(f"{str(c):>5}" for c in cats))
    for cohort in cohorts:
        row = "".join(f"{tab2.get((cohort, c), 0):>5}" for c in cats)
        print(f"    {cohort:<18}{row}")
    print()

    # ---- the annotator's view: everything EXCEPT gold and outcome -----------
    blind = [{"question_id": s["question_id"], "question": s["question"],
              "conversation": s["conversation"], "category": s["category"]}
             for s in selected]

    OUT.write_text(json.dumps({
        "label": "V4 mission 4 -- stratified pilot selection, drawn before annotation",
        "seed": SEED,
        "per_cohort": PER_COHORT,
        "n": n,
        "selection_method": "proportional stratification on (conversation, category), "
                            "deterministic seed, sorted within stratum",
        "gold_used_for_sampling": False,
        "by_cohort_conversation": {f"{a}|{b}": c for (a, b), c in tab.items()},
        "by_cohort_category": {f"{a}|{b}": c for (a, b), c in tab2.items()},
        "selected": selected,
        "annotator_view_ids": [b["question_id"] for b in blind],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    blind_out = OUT.with_name("v4_pilot_annotator_view.json")
    blind_out.write_text(json.dumps({
        "label": "what the annotator receives -- NO gold, NO prior prediction, "
                 "NO cohort label",
        "n": len(blind),
        "cases": blind,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  written: {OUT}")
    print(f"  written: {blind_out}  (annotator view: gold and outcomes withheld)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
