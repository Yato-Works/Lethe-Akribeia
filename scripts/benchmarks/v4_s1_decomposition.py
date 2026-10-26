"""V4 Mission 21 -- decompose s1_source_evidence and make the 610 unattributed visible.

READ-ONLY.  No production change, no model calls, no wiring, no re-derivation of existing
labels.  Frozen artefacts only.

A DISCREPANCY IS RESOLVED FIRST, and it changes the M20 number.  M20's map reported
s1_source_evidence at 493.  Recomputing with the same rule gives 564.  The difference is
an ordering bug in the M20 script: it built the STAGES list with s9 included but applied
the unmeasured exclusion by NAME only for stages it had already detected, so some rows
were attributed to a later stage.  This audit recomputes cleanly, prints both numbers, and
uses 564.  The M20 artefact is left as recorded; the correction is stated rather than
patched silently.

THE SECOND FINDING IS THE ONE THAT MATTERS MOST.  Of the rows whose first blocking stage
is s1, 71 have verdict=correct and 43 have verdict=wrong.  A stage labelled 'blocking'
that co-occurs with correct answers is not blocking those answers at all.  So the label
measures evidence PRESENCE, not causal failure -- and 71 counter-examples prove it.

Decomposition, per the pre-declared hypotheses, using only recorded fields:
  A. no ground in source     : VERBATIM_ABSENT -- the gold string is not in the source
  B. present but not verbatim: TOKEN_PARTIAL -- tokens present, exact string is not
  C. lost after extraction   : s1 has evidence but s2/s3 lost it
  D. misattributed           : s1 partial AND s4/s5 disagree
  E. measurement artifact    : verdict=correct while s1 'blocks', or verdict=wrong with
                               s1=VERBATIM (the evidence was fine and it still failed)

AND THE 610 'no blocking stage recorded' ROWS: are they unmeasured, off-path, or
unclassifiable?  Those are separated explicitly instead of left as the map's largest
unknown.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "benchmark_results/evidence_audit/v4_stage_ledger.json"
H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_s1_decomposition.json"

STAGES = ["s1_source_evidence", "s2_preserved_in_memory", "s3_retrieved_into_context",
          "s4_speaker_attribution", "s5_subject_attribution", "s6_relation_support",
          "s7_temporal_polarity", "s8_commitment", "s9_reader_fallback"]
BLOCK = ("UNKNOWN", "ABSENT", "NOT_FOUND", "NO_TOKENS", "PARTIAL", "NONE")
UNMEAS = ("NO RULE MEASURED", "NOT EXECUTED")


def is_unmeasured(v: str) -> bool:
    return any(m in v.upper() for m in UNMEAS)


def is_block(v: str) -> bool:
    return any(b in v.upper() for b in BLOCK)


def main() -> int:
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    rows = led["rows"]
    h9 = {r["question_id"] for r in
          json.loads(H9.read_text(encoding="utf-8"))["rows"]}

    unmeasured_stages = [s for s in STAGES
                         if all(is_unmeasured(str(r.get(s) or "")) for r in rows)]

    def first_block(r):
        for s in STAGES:
            if s in unmeasured_stages:
                continue
            if is_block(str(r.get(s) or "")):
                return s
        return None

    print("=" * 78)
    print("V4 MISSION 21 -- decompose s1_source_evidence; expose the 610 unattributed")
    print("=" * 78)
    print("  READ-ONLY.  No production change, no model calls, no re-derivation of")
    print("  existing labels.  Frozen artefacts only.")
    print()
    print("UNMEASURED STAGES (excluded everywhere):")
    for s in unmeasured_stages:
        print(f"    {s}  [{str(rows[0].get(s) or '')[:50]}]")
    print()

    # ---------------- resolve the 493 vs 564 discrepancy -------------------
    s1_rows = [r for r in rows if first_block(r) == "s1_source_evidence"]
    print("DISCREPANCY WITH M20, resolved:")
    print(f"    M20 reported s1 at 493; recomputing with the same rule gives {len(s1_rows)}.")
    print("    Cause: M20 applied the unmeasured-stage exclusion by name only for stages")
    print("    it had already detected, so some rows fell through to a later stage.")
    print("    This audit recomputes cleanly and uses "
          f"{len(s1_rows)}.  M20's artefact is left as recorded.")
    print()

    # ---------------- the finding that matters most ------------------------
    v = Counter(r["verdict"] for r in s1_rows)
    print("AND THIS IS THE FINDING THAT MATTERS MOST:")
    print(f"    rows whose first blocking stage is s1, by verdict: {dict(v)}")
    print(f"    {v['correct']} of them are CORRECT and {v['wrong']} are WRONG.")
    print("    A stage labelled 'blocking' that co-occurs with "
          f"{v['correct']} correct answers is")
    print("    not blocking those answers.  The label measures evidence PRESENCE, not")
    print("    causal failure -- and the correct cases are the counter-examples.")
    print()

    # ---------------- decomposition ---------------------------------------
    print("=" * 78)
    print("DECOMPOSITION of the s1 rows (pre-declared hypotheses)")
    print("=" * 78)
    buckets = Counter()
    detail = {"A": [], "B": [], "C": [], "D": [], "E": []}
    # s3 is 'YES (frozen context is the candidate set)' for ALL 1,986 rows, so it can
    # never be a blocking stage and must not be consulted.  BUG FIXED: the first version
    # tested `is_block(s3)` alongside s2, and because s3 is uniformly YES it contributed
    # nothing -- but the s2 test then absorbed everything, mislabelling 282 rows as
    # 'lost at s2/s3' when in fact all 282 have s3=YES.  Retrieval is not the loss point.
    S3_CONSTANT_YES = len({str(r.get("s3_retrieved_into_context") or "") for r in rows}) == 1
    for r in s1_rows:
        s1, s2 = (str(r.get("s1_source_evidence") or ""),
                  str(r.get("s2_preserved_in_memory") or ""))
        s4, s5 = (str(r.get("s4_speaker_attribution") or ""),
                  str(r.get("s5_subject_attribution") or ""))
        if r["verdict"] == "correct" or (r["verdict"] == "wrong" and "VERBATIM" in s1):
            k = "E"      # measurement artifact / not a real s1 failure
        elif "VERBATIM_ABSENT" in s1:
            k = "A"      # no verbatim ground in source
        elif "TOKEN_PARTIAL" in s1:
            if is_block(s2):
                k = "C"  # extracted, then degraded in memory representation
            elif is_block(s4) or is_block(s5):
                k = "D"  # present but attribution failed
            else:
                k = "B"  # present, not verbatim, attribution ok
        else:
            k = "B"
        buckets[k] += 1
        detail[k].append(r["question_id"])

    NAMES = {
        "A": "no verbatim ground in source (evidence may be absent)",
        "B": "ground present but not verbatim; attribution fine",
        "C": "extracted at s1, then degraded at s2 (s3 is constant YES and cannot block)",
        "D": "present but s4/s5 attribution failed",
        "E": "MEASUREMENT ARTIFACT: verdict correct while s1 'blocks', "
             "or wrong despite VERBATIM",
    }
    for k in "ABCDE":
        print(f"  {k}. {NAMES[k]:<58}{buckets[k]:>5}")
        print(f"       verdict mix: "
              f"{dict(Counter(r['verdict'] for r in s1_rows if r['question_id'] in set(detail[k])))}")
    print(f"  {'TOTAL':<64}{len(s1_rows):>5}")
    print()
    print("  Bucket E is the important one: it is where the s1 label disagrees with the")
    print("  outcome, so those rows are NOT s1 failures.  Counting them as s1 losses")
    print("  would overstate the target.")
    print()

    # ---------------- the 610 unattributed ---------------------------------
    print("=" * 78)
    print("THE 610 'NO BLOCKING STAGE RECORDED' -- what are they actually?")
    print("=" * 78)
    none_rows = [r for r in rows if first_block(r) is None]
    print(f"  rows with no blocking stage among the MEASURED stages : {len(none_rows)}")
    nv = Counter(r["verdict"] for r in none_rows)
    print(f"    verdict mix : {dict(nv)}")
    print()
    # sub-classify: were they never even in the engine's decision path?
    in_h9 = sum(1 for r in none_rows if r["question_id"] in h9)
    print(f"    present in the H9 Reader cohort : {in_h9}")
    print()
    print("  Reading: these rows have NO blocking value in any MEASURED stage.  Two")
    print("  possibilities the artefact can distinguish:")
    print("    (i) the deterministic engine genuinely had full evidence and still")
    print("        abstained or erred -- a real failure with no recorded cause, i.e. the")
    print("        s8_commitment / decision stage, which this harness does not label;")
    print("   (ii) the row is outside the harness path, which the H9 membership check")
    print("        above partly indicates.")
    print(f"  Either way {len(none_rows)} rows are UNATTRIBUTED, and they are the map's")
    print("  largest unknown.  They are reported, not absorbed into s1.")
    print()

    print("=" * 78)
    print("WHAT THIS CHANGES")
    print("=" * 78)
    real_e = len(s1_rows) - buckets["E"]
    print(f"  s1 rows reported            : {len(s1_rows)}  (M20 said 493 -- corrected)")
    print(f"  minus measurement artifacts : {real_e}")
    print(f"  of which bucket A (no verbatim ground): {buckets['A']}")
    print()
    print("  So the honest target is not 493 and not 564.  It is at most "
          f"{real_e}, and only")
    print(f"  after bucket A is checked against the source conversation -- a")
    print("  VERBATIM_ABSENT label does not prove the fact is absent, only that the")
    print("  exact gold string is not present, which M7 already showed can be false")
    print("  (inclusive vs inclusivity, paraphrase answers).")
    print()
    print("  NEXT STEP, in order: (1) verify a stratified sample of bucket A against the")
    print("  raw source conversation to split 'truly absent' from 'present but not")
    print("  verbatim'; (2) only then design an intervention, measuring rescued AND")
    print("  newly-broken counts under the same frozen scorer.")
    print()
    print("  SCOPE: read-only decomposition of recorded labels.  No source text was")
    print("  re-inspected here, so bucket A's size is unverified.  No model calls, no")
    print("  production change, and the 553-question holdout is untouched.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 21 -- s1 decomposition + the 610 unattributed exposed",
        "baseline": {"head": "614e5cf", "production_changed": False},
        "model_calls": 0,
        "unmeasured_stages": unmeasured_stages,
        "s1_row_count": len(s1_rows),
        "m20_reported_s1_count": 493,
        "discrepancy_cause": "M20 applied the unmeasured exclusion by name only for "
                             "already-detected stages",
        "s1_verdict_mix": dict(v),
        "buckets": dict(buckets),
        "bucket_definitions": NAMES,
        "bucket_ids": detail,
        "n_unattributed": len(none_rows),
        "unattributed_verdict_mix": dict(nv),
        "unattributed_in_h9": in_h9,
        "honest_target_after_artifact_removal": real_e,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
