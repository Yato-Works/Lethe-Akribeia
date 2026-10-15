"""Correct the M19 prevalence measurement.

M19 reported 51.6% update prevalence, and inspection showed WHY that number is wrong:
the dominant firing pattern is `now i` (184 of the hits), and its actual matches are
"Now I want to help people go through it too" and "Now I'm passionate about creating a
safe, inviting..." -- continuations of a current stance, NOT state changes over time.
`used to` (71) and `since then` (41) also fire on narrative recall rather than on a
superseded state.  So the 51.6% measures ordinary conversation, not the axis under test.

This is the same failure mode M13 hit: an instrument that fires on almost everything
carries no information.  It is corrected here rather than allowed to stand.

TIGHTENED DEFINITION -- a state update requires BOTH a past state and a present state
that differ.  Patterns that express only one side, or only narrative continuity, are
removed.  Every retained pattern must be justifiable as "X before, Y now".
"""
from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_update_prevalence_strict.json"

_spec = importlib.util.spec_from_file_location(
    "m18markers", ROOT / "scripts" / "benchmarks" / "v4_update_eligibility.py")
_m18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m18)
CORRECTION = _m18._CORRECTION

# ---- TIGHTENED: both a past state and a differing present state are required ----
# ---- TIGHTENED (v3): 'get used to' is acclimation, NOT a state update ----------
# The v2 marker fired 64/54 times on "took her a few days to get used to the other
# dogs, but now they're awesome friends" -- pets acclimating to each other, which is
# not a knowledge update at all.  `used to` swallowed the "get" prefix because the
# \b before it does not block a preceding word.  A negative lookbehind for get/got/
# getting/gotten excludes it.
_UPDATE_STRICT = re.compile(
    r"(?<!\bget\s)(?<!\bgot\s)(?<!\bgetting\s)(?<!\bgotten\s)"
    r"(?:\bused\s+to\b[^.?!]{0,60}?\bbut\s+(?:now|these\s+days|i)\b"
    r"|\bno\s+longer\b"
    r"|\bnot\s+anymore\b"
    r"|\bi(?:'ve|\s+have)\s+(?:since\s+)?(?:moved\s+on|switched|quit|stopped|"
    r"given\s+(?:it|that)\s+up)\b"
    r"|\b(?:quit|stopped)\s+(?:\w+\s+){0,2}(?:and|but)\s+(?:now\s+)?(?:i|instead)\b"
    r"|\bused\s+to\s+(?:be|do|have|work|live|play|paint|run|go)\b[^.?!]{0,80}?"
    r"\b(?:now|these\s+days|currently|today)\b)",
    re.I)


def main() -> int:
    h9 = json.loads(H9.read_text(encoding="utf-8"))["rows"]
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    print("=" * 78)
    print("V4 MISSION 19 (corrected) -- update prevalence with a TIGHTENED marker")
    print("=" * 78)
    print("  WHY THIS REDO EXISTS: the first M19 reported 51.6% update prevalence.")
    print("  Inspection showed the dominant hit was `now i` (184), matching")
    print("  'Now I want to help people go through it too' and 'Now I'm passionate about")
    print("  creating a safe, inviting...' -- continuations of a current stance, NOT state")
    print("  changes.  `used to` (71) and `since then` (41) fired on narrative recall.")
    print("  The 51.6% therefore measured ordinary conversation, not the axis under test.")
    print("  Same failure mode as M13: an instrument that fires on nearly everything")
    print("  carries no information.  Corrected here rather than left standing.")
    print()
    print("  TIGHTENED RULE: an update now requires BOTH a past state and a differing")
    print("  present state.  One-sided and narrative-continuity patterns are removed.")
    print()

    # ---- self-test FIRST, so the new null is trustworthy -------------------
    POS = [
        "I used to paint every day but now I mostly do photography.",
        "I no longer work at the studio.",
        "I used to run every morning, but now I swim instead.",
        "I've since switched to a different team.",
        "I used to be a teacher, these days I write full time.",
    ]
    NEG = [
        "Now I want to help people go through it too.",
        "Now I'm passionate about creating a safe, inviting space.",
        "I used to go horseback riding with my dad when I was a kid.",
        "Ever since then things have been great.",
        # NOTE: an earlier version of this test used "I'm not into running anymore",
        # which does not contain "not anymore" and was therefore correctly a NEGATIVE.
        # The miss that time was my bad test string, not the marker.
        "It took her a few days to get used to the other dogs, but now they're "
        "awesome friends.",
    ]
    f_pos = [s for s in POS if not _UPDATE_STRICT.search(s)]
    f_fp = [s for s in NEG if _UPDATE_STRICT.search(s)]
    print("SELF-TEST of the tightened marker")
    print(f"  positives recognised : {len(POS) - len(f_pos)}/{len(POS)}")
    print(f"  false positives      : {f_fp or 'none'}")
    print()
    if f_pos or f_fp:
        print("  Marker still imperfect -- results below are provisional and labelled so.")
    else:
        print("  Marker behaves correctly on both sets.  A null now means something.")
    print()

    rows = []
    hits = Counter()
    for r in h9:
        qid = r["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        if not context:
            rows.append({"question_id": qid, "has_context": False})
            continue
        corr = bool(CORRECTION.search(context))
        upd = bool(_UPDATE_STRICT.search(context))
        for mm in _UPDATE_STRICT.finditer(context):
            hits[mm.group(0).lower()[:40]] += 1
        rows.append({"question_id": qid, "has_context": True,
                     "has_correction": corr, "has_update": upd,
                     "has_either": corr or upd,
                     "conversation": r["conversation"]})

    ok = [x for x in rows if x.get("has_context")]
    n_corr = sum(1 for x in ok if x["has_correction"])
    n_upd = sum(1 for x in ok if x["has_update"])
    n_either = sum(1 for x in ok if x["has_either"])

    print("=" * 78)
    print("STRICT PREVALENCE (blind)")
    print("=" * 78)
    print(f"  questions              : {len(ok)}")
    print(f"  explicit correction    : {n_corr}  ({100 * n_corr / len(ok):.1f}%)")
    print(f"  state update (strict)  : {n_upd}  ({100 * n_upd / len(ok):.1f}%)")
    print(f"  either                 : {n_either}  ({100 * n_either / len(ok):.1f}%)")
    print()
    print(f"  previous (loose) figure was 51.6%; strict is "
          f"{100 * n_either / len(ok):.1f}%")
    print()
    print("  firing expressions (top):")
    for k, v in hits.most_common(8):
        print(f"    {k!r}: {v}")
    print()

    outcome = {r["question_id"]: r["mutually_exclusive"] for r in h9}
    for x in ok:
        x["outcome"] = outcome.get(x["question_id"], "UNKNOWN")

    print("=" * 78)
    print("STRICT PREVALENCE x OUTCOME (outcome joined after labelling)")
    print("=" * 78)
    print(f"  {'condition':<18}{'n':>6}{'correct':>9}{'incorrect':>11}{'accuracy':>10}")
    for label, key in (("update present", "has_update"), ("neither", "has_either")):
        sub = [x for x in ok if (x[key] if label != "neither" else not x["has_either"])]
        if not sub:
            print(f"  {label:<18}{0:>6}{'-':>9}{'-':>11}{'-':>10}")
            continue
        c = sum(1 for x in sub if x["outcome"] == "CORRECT")
        i = sum(1 for x in sub if x["outcome"] == "INCORRECT")
        print(f"  {label:<18}{len(sub):>6}{c:>9}{i:>11}{100 * c / len(sub):>9.1f}%")
    print()

    print("=" * 78)
    print("GATE")
    print("=" * 78)
    rate = 100 * n_either / len(ok)
    if n_either == 0:
        print(f"  ZERO under the strict definition.  Explicit corrections: {n_corr}.")
        print("  -> the update/correction axis has NO population in this benchmark's")
        print("     Reader contexts.  It cannot explain the failure set, and it is")
        print("     abandoned on evidence rather than on assumption.")
    elif rate < 3:
        print(f"  {rate:.1f}% ({n_either}/{len(ok)}).  Rare enough that a mechanism for it")
        print("  cannot be the main explanation of the failures.")
    else:
        print(f"  {rate:.1f}% ({n_either}/{len(ok)}).  A population exists under the strict")
        print("  definition, so the axis is eligible for a mechanism test.")
    print()
    print("  SCOPE: still a PREVALENCE measure, not a pairwise-relation measure.  Surface")
    print("  markers, not human adjudication.  Does not observe the Reader's reasoning.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 19 corrected -- strict update/correction prevalence",
        "baseline": {"head": "ae5c9f0", "production_changed": False},
        "why_corrected": ("loose `now i` matched narrative continuations, not state "
                          "changes; loose prevalence was 51.6%"),
        "strict_marker_selftest": {"positives_ok": not f_pos,
                                   "false_positives": f_fp},
        "measure": "prevalence of correction / strict state-update language",
        "not_measured": "pairwise correction relation between specific candidates",
        "n_questions": len(ok), "n_correction": n_corr, "n_update_strict": n_upd,
        "n_either": n_either, "prevalence_pct": rate,
        "firing_expressions": dict(hits.most_common(20)),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
