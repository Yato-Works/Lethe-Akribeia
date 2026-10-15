"""V4 Mission 19 -- does the update/correction axis have ANY population in the benchmark?

M18 stage 1 found ZERO correction/update relations in the 30-case cohort and therefore
declared the axis ineligible THERE.  That null is scoped: the cohort was stratum-selected
for having two competing same-kind events, which is itself a selection that could exclude
corrections.  So the honest follow-up is the mission's own fork -- verify on different
non-holdout data -- before abandoning the axis.

WHAT IS MEASURED, and the honest limit on it.  For the full 989 questions this audit
labels each Reader CONTEXT for the presence of correction language and update language.
This is a PREVALENCE measure, not a pairwise-relation measure: M18 classified a specific
competing pair, which requires per-case candidate identification that does not exist for
all 989.  So "the context contains a correction" is what is measured, and that is what is
reported -- it is NOT "this question's competing candidates stand in a correction
relation".

BLIND: labels come from surface markers on the utterance text only.  The correctness
label is joined afterwards, so no labelling can be tuned to the outcome.

Markers are imported from the M18 script rather than redefined, so the two audits use
identical definitions -- and those markers were self-tested against known positives and
negatives (5/5 fire, 0 false positives).

NO MODEL CALLS.  Frozen contexts and frozen outcomes only.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
SPLIT = ROOT / "benchmark_results/evidence_audit/v4_holdout_split.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_update_prevalence.json"

# reuse the EXACT markers from the M18 audit, so both audits share one definition
_spec = importlib.util.spec_from_file_location(
    "m18markers", ROOT / "scripts" / "benchmarks" / "v4_update_eligibility.py")
_m18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m18)
CORRECTION, UPDATE = _m18._CORRECTION, _m18._UPDATE


def main() -> int:
    h9 = json.loads(H9.read_text(encoding="utf-8"))["rows"]
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    holdout_ids = set(split["holdout_qids"])

    print("=" * 78)
    print("V4 MISSION 19 -- update/correction prevalence across all 989 questions")
    print("=" * 78)
    print("  M18 found 0 relations in its 30-case cohort.  That cohort was stratum-")
    print("  selected for competing same-kind events, a selection that could itself")
    print("  exclude corrections.  So the axis is measured benchmark-wide before it is")
    print("  abandoned.")
    print()
    print("  MEASURED: whether the Reader's context contains correction / update")
    print("  language.  NOT measured: that a specific competing pair stands in a")
    print("  correction relation -- pairwise classification needs per-case candidate")
    print("  identification, which exists only for the M11 cohort.")
    print()
    print("  BLIND: labels from surface markers on utterance text only; the correctness")
    print("  label is joined afterwards.  Markers are imported from the M18 script, so")
    print("  both audits share one definition (self-tested 5/5 positives, 0 false pos).")
    print(f"  NO MODEL CALLS.  questions: {len(h9)}")
    print()

    # ---------------- blind labelling ---------------------------------------
    rows = []
    for r in h9:
        qid = r["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        if not context:
            rows.append({"question_id": qid, "has_context": False})
            continue
        corr = bool(CORRECTION.search(context))
        upd = bool(UPDATE.search(context))
        rows.append({
            "question_id": qid, "has_context": True,
            "has_correction": corr, "has_update": upd,
            "has_either": corr or upd,
            "conversation": r["conversation"],
            "split": "HOLDOUT" if qid in holdout_ids else "TUNING",
        })

    ok = [x for x in rows if x.get("has_context")]
    n_corr = sum(1 for x in ok if x["has_correction"])
    n_upd = sum(1 for x in ok if x["has_update"])
    n_either = sum(1 for x in ok if x["has_either"])

    print("=" * 78)
    print("PREVALENCE (blind)")
    print("=" * 78)
    print(f"  contexts located        : {len(ok)}/{len(h9)}")
    print(f"  correction language     : {n_corr}  ({100 * n_corr / len(ok):.1f}%)")
    print(f"  update language         : {n_upd}  ({100 * n_upd / len(ok):.1f}%)")
    print(f"  either                  : {n_either}  ({100 * n_either / len(ok):.1f}%)")
    print()

    # ---------------- join outcome ONLY NOW ---------------------------------
    outcome = {r["question_id"]: r["mutually_exclusive"] for r in h9}
    for x in ok:
        x["outcome"] = outcome.get(x["question_id"], "UNKNOWN")

    print("=" * 78)
    print("PREVALENCE x OUTCOME  (outcome joined after labelling was frozen)")
    print("=" * 78)
    print(f"  {'condition':<18}{'n':>6}{'correct':>9}{'incorrect':>11}{'accuracy':>10}")
    for label, key in (("correction present", "has_correction"),
                       ("update present", "has_update"),
                       ("either present", "has_either"),
                       ("neither", "has_either")):
        sub = [x for x in ok
               if (x[key] if label != "neither" else not x["has_either"])]
        if not sub:
            print(f"  {label:<18}{0:>6}{'-':>9}{'-':>11}{'-':>10}")
            continue
        c = sum(1 for x in sub if x["outcome"] == "CORRECT")
        i = sum(1 for x in sub if x["outcome"] == "INCORRECT")
        print(f"  {label:<18}{len(sub):>6}{c:>9}{i:>11}{100 * c / len(sub):>9.1f}%")
    print()

    # ---------------- per-conversation, no dialogue may dominate -------------
    print("=" * 78)
    print("PER-CONVERSATION spread of the 'either' cases")
    print("=" * 78)
    convs = Counter(x["conversation"] for x in ok if x["has_either"])
    tot = Counter(x["conversation"] for x in ok)
    print(f"  {'conv':<10}{'either':>8}{'/n':>6}{'rate':>8}")
    for cv in sorted(tot):
        print(f"  {cv:<10}{convs.get(cv, 0):>8}{tot[cv]:>6}"
              f"{100 * convs.get(cv, 0) / tot[cv]:>7.1f}%")
    print()

    # ---------------- tuning / holdout separation ---------------------------
    print("=" * 78)
    print("SPLIT CHECK (holdout stays reserved for final evaluation of an adopted method)")
    print("=" * 78)
    for sp in ("TUNING", "HOLDOUT"):
        sub = [x for x in ok if x["split"] == sp]
        e = sum(1 for x in sub if x["has_either"])
        print(f"  {sp:<9} n={len(sub):<5} either={e}  ({100 * e / len(sub):.1f}%)"
              if sub else f"  {sp}: none")
    print()

    # ---------------- gate --------------------------------------------------
    print("=" * 78)
    print("GATE (pre-declared)")
    print("=" * 78)
    rate = 100 * n_either / len(ok)
    if n_either == 0:
        print("  ZERO prevalence benchmark-wide.  The axis has no population here at all,")
        print("  so it cannot explain the failures and is abandoned with evidence.")
    elif rate < 2:
        print(f"  Prevalence {rate:.1f}% ({n_either}/{len(ok)}).  The axis exists but is")
        print("  vanishingly rare.  A mechanism for it could touch at most that fraction,")
        print("  so it cannot be the main explanation of the failure set.")
    else:
        print(f"  Prevalence {rate:.1f}% ({n_either}/{len(ok)}).  A population exists, so the")
        print("  axis is eligible for a mechanism test on a cohort where it occurs.")
    print()
    print("  SCOPE: a PREVALENCE measure, not a pairwise-relation measure.  Correction")
    print("  language somewhere in a context does not mean the question's competing")
    print("  candidates stand in a correction relation.  Labels are surface markers, not")
    print("  human adjudication.  Nothing here observes the Reader's reasoning.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 19 -- update/correction prevalence across 989 (blind)",
        "baseline": {"head": "ae5c9f0", "production_changed": False},
        "model_calls": 0,
        "markers_shared_with": "v4_update_eligibility.py (self-tested 5/5, 0 FP)",
        "measure": "prevalence of correction/update language in the Reader's context",
        "not_measured": "pairwise correction relation between specific competing candidates",
        "n_questions": len(ok), "n_correction": n_corr, "n_update": n_upd,
        "n_either": n_either, "prevalence_pct": rate,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
