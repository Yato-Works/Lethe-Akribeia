"""Engine capability on the frozen contexts, stratified by evidence class.

The mission: measure what the deterministic engine can already answer and how many
more are recoverable, using the FROZEN contexts rather than a fresh benchmark run.
Then split failures four ways:
    ALREADY_CORRECT        the engine answers correctly today
    EVIDENCE_PRESENT_ABSTAIN  gold is in the context, engine abstained  <- recoverable
    WRONG_COMMIT           engine answered and was wrong                  <- harmful
    NO_EVIDENCE            gold absent from the compiled context           <- not here

A denominator correction is recorded first, because Eli caught one: the previous report
called the date-related population '167' while the four evidence classes sum to 192.
167 is DIRECT+DERIVED+UNDETERMINED (three classes); 192 includes INSUFFICIENT (four).
The number is derived from the audit file here rather than typed, so it cannot drift.

Stratified by evidence class so 'recoverable' is never quoted as one aggregate: an
INSUFFICIENT question cannot be recovered by engine work on this context, and lumping
it with DIRECT would overstate the opportunity by 25 questions.

No benchmark re-run: frozen contexts are reused, and both scorer generations are named.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

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

AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
OUT = ROOT / "benchmark_results/evidence_audit/engine_capability.json"

#: Evidence classes that are date-shaped answers, i.e. the temporal engine's remit.
DATE_CLASSES = ("DIRECT", "DERIVED", "UNDETERMINED")
#: The four-way split, so the sum is assertable.
SPLIT = ("ALREADY_CORRECT", "EVIDENCE_PRESENT_ABSTAIN", "WRONG_COMMIT", "NO_EVIDENCE")


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = audit["records"]
    cache = {}
    for line in (ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
                 ).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    # ---- denominator, derived not typed.
    cls = Counter(r["class"] for r in records)
    date_shaped = sum(cls[c] for c in DATE_CLASSES)
    four_class = sum(cls[c] for c in DATE_CLASSES + ("INSUFFICIENT",))
    print("DENOMINATOR, derived from the audit file")
    print(f"  DIRECT+DERIVED+UNDETERMINED (3 classes) : {date_shaped}")
    print(f"  + INSUFFICIENT (4 classes)               : {four_class}")
    print(f"  DIRECT alone                             : {cls['DIRECT']}")
    print("  Eli's correction: the earlier report's '167' was the 3-class figure;")
    print("  the 4-class evidence population is 192.  Recorded, not silently fixed.")
    print()

    # ---- per evidence class, run the engine on the frozen context.
    strata: dict[str, dict] = {}
    all_rows: list[dict] = []

    for ev_class in DATE_CLASSES + ("INSUFFICIENT",):
        subset = [r for r in records if r["class"] == ev_class]
        counts = Counter()
        rows: list[dict] = []
        for r in subset:
            ctx = cache.get(r["cache_qid"]) or {}
            context = ctx.get("context") or ""
            gold = r["gold"]
            row = {"question_id": r["question_id"], "category": r["category"],
                   "question": r["question"], "gold": gold,
                   "evidence_class": ev_class,
                   "source_turn": r.get("source_turn")}
            if not context or not gold:
                counts["NO_EVIDENCE"] += 1
                row["split"] = "NO_EVIDENCE"
                rows.append(row)
                continue

            ans = DeterministicMemoryEngine.resolve(r["question"], context)
            row["engine_answer"] = ans.answer
            row["engine_used"] = bool(ans.used)
            row["engine_detail"] = (ans.detail or "")[:120]

            if not ans.used:
                # Abstain.  Recoverable only if the evidence is really there.
                if ev_class in ("DIRECT", "DERIVED"):
                    row["split"] = "EVIDENCE_PRESENT_ABSTAIN"
                elif ev_class == "UNDETERMINED":
                    row["split"] = "EVIDENCE_PRESENT_ABSTAIN"
                else:
                    row["split"] = "NO_EVIDENCE"
            elif scorer.score(ans.answer, gold):
                row["split"] = "ALREADY_CORRECT"
            else:
                row["split"] = "WRONG_COMMIT"
            counts[row["split"]] += 1
            rows.append(row)

        strata[ev_class] = {"n": len(subset), "split": dict(counts), "rows": rows}
        all_rows.extend(rows)

    print("=" * 76)
    print("ENGINE CAPABILITY, STRATIFIED BY EVIDENCE CLASS")
    print("=" * 76)
    hdr = f"  {'evidence':<14}{'n':>5} " + " ".join(f"{s[:12]:>14}" for s in SPLIT)
    print(hdr)
    print("-" * len(hdr))
    totals = Counter()
    for ev in DATE_CLASSES + ("INSUFFICIENT",):
        s = strata[ev]["split"]
        totals.update({k: v for k, v in s.items()})
        print(f"  {ev:<14}{strata[ev]['n']:>5} "
              + " ".join(f"{s.get(k, 0):>14}" for k in SPLIT))
    print("-" * len(hdr))
    print(f"  {'TOTAL':<14}{sum(strata[e]['n'] for e in strata):>5} "
          + " ".join(f"{totals.get(k, 0):>14}" for k in SPLIT))
    print()

    print("=" * 76)
    print("WHAT IS RECOVERABLE BY ENGINE WORK ON THESE CONTEXTS")
    print("=" * 76)
    rec = totals.get("EVIDENCE_PRESENT_ABSTAIN", 0)
    wrong = totals.get("WRONG_COMMIT", 0)
    correct = totals.get("ALREADY_CORRECT", 0)
    print(f"  already correct                  : {correct}")
    print(f"  EVIDENCE PRESENT but abstained   : {rec}   <- the improvement pool")
    print(f"  wrong commit (harmful)           : {wrong}   <- fixing these LOSES nothing")
    print(f"  no evidence in this context      : {totals.get('NO_EVIDENCE', 0)}")
    print()
    print("  The improvement pool is the EVIDENCE_PRESENT_ABSTAIN count, NOT the whole")
    print("  192: an INSUFFICIENT question needs retrieval or corpus work, not engine")
    print("  work, so counting it would overstate the opportunity.")
    print()

    # ---- the recoverable ones, listed so they can be worked.
    pool = [r for r in all_rows if r["split"] == "EVIDENCE_PRESENT_ABSTAIN"]
    print(f"RECOVERABLE POOL ({len(pool)}), by evidence class:")
    for ev in DATE_CLASSES + ("INSUFFICIENT",):
        n = sum(1 for r in pool if r["evidence_class"] == ev)
        print(f"  {ev:<14} {n}")
    print()
    for r in pool[:12]:
        print(f"  {r['question_id']:<18} [{r['evidence_class']}]")
        print(f"    Q    : {r['question'][:60]}")
        print(f"    gold : {r['gold']!r}   from turn {r.get('source_turn')}")
        print(f"    why  : {r['engine_detail'][:76]}")
    if len(pool) > 12:
        print(f"  ... {len(pool) - 12} more")

    # ---- Category 4, measured rather than assumed.
    print()
    print("=" * 76)
    print("CATEGORY 4 (open-domain) -- measured, not assumed hopeless")
    print("=" * 76)
    cat4 = [r for r in records if r["category"] == 4]
    c4cls = Counter(r["class"] for r in cat4)
    print(f"  n={len(cat4)}   classes={dict(c4cls)}")
    c4_nondate = [r for r in cat4 if r["class"] == "NOT_A_DATE"]
    shapes = Counter()
    for r in c4_nondate:
        q = r["question"].lower()
        if re.search(r"\bhow many\b|\bcount\b", q):
            shapes["counting"] += 1
        elif re.search(r"\bwhat .*(?:like|prefer|hobby|hobbies)\b", q):
            shapes["preference/attribute"] += 1
        elif re.search(r"^(what|which)\b.*\b(things?|items?|names?|books?|movies?|songs?|places?|foods?)\b", q):
            shapes["enumeration"] += 1
        elif re.search(r"\bwhy\b", q):
            shapes["reason"] += 1
        else:
            shapes["other open-domain"] += 1
    print("  NOT_A_DATE question shapes in category 4:")
    for k, n in shapes.most_common():
        print(f"    {k:<24} {n}")
    print()
    print("  These shapes are where a deterministic engine could plausibly reach")
    print("  WITHOUT an LLM: counting and enumeration are set operations, and the")
    print("  identity module already unions them.  Measured here rather than assumed.")

    OUT.write_text(json.dumps({
        "label": "engine capability on frozen contexts, stratified by evidence class",
        "denominator": {
            "three_class_date_shaped": date_shaped,
            "four_class_including_insufficient": four_class,
            "direct": cls["DIRECT"],
            "note": "an earlier report called this population 167; the four-class "
                    "figure is 192.  167 is the three-class subtotal.",
        },
        "scorer": "eval_locomo_full.score (date + day + numeral + month guards)",
        "contexts": "frozen locomo_context_cache_rules.jsonl, reused; no re-run",
        "strata": {k: {"n": v["n"], "split": v["split"]} for k, v in strata.items()},
        "totals": dict(totals),
        "recoverable_pool_n": len(pool),
        "category4_shapes": dict(shapes),
        "rows": all_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
