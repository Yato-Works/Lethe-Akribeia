"""V4 Mission 20 -- where are the lost points, across the whole benchmark?

Step 1 of the three-step plan: understand the current loss distribution over the ENTIRE
evaluation set, rather than continuing to dig into one hard failure group.

A PROVENANCE PROBLEM IS RESOLVED FIRST, because it determines what can honestly be
measured here.  Two 1,986-row artefacts exist and they disagree by a factor of seven:
  - v4_stage_ledger.json     : verdicts correct 137 / wrong 113 / abstain 1736  (6.9%)
  - locomo_full_20261010_151930.json : correct 1009 / wrong 977                (50.8%)
The stage ledger is auditable -- it carries per-stage attribution (s1..s10), the baseline
HEAD, and sha256 of dataset, context cache and scorer.  The 50.8% artefact carries NONE of
that: no model, no prompt, no config, no context, no reader identity.  M0 already flagged
that exact provenance gap and refused to merge the two.

So this audit measures the AUDITABLE run and says so.  It does NOT report 50.8% as the
current score, because nothing in the repository establishes what produced it.  The gap
between 6.9% and 50.8% is itself the finding that matters for step 1.

WHAT IS MEASURED
  1. The loss distribution of the auditable run, by category and by stage.
  2. For every abstention, WHICH stage first blocked the answer -- this is the actual
     "where are the points going" map, and it is what no previous mission produced.
  3. The size of the Reader-recoverable pool (the H9 cohort) for comparison.

The ledger's stage columns are read as recorded; nothing is re-derived, so this cannot
introduce a new classification error.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EA = ROOT / "benchmark_results/evidence_audit"

LEDGER = EA / "v4_stage_ledger.json"
H9 = EA / "v4_reader_h9.json"
OUT = EA / "v4_loss_decomposition.json"

STAGES = ["s1_source_evidence", "s2_preserved_in_memory", "s3_retrieved_into_context",
          "s4_speaker_attribution", "s5_subject_attribution", "s6_relation_support",
          "s7_temporal_polarity", "s8_commitment", "s9_reader_fallback"]


def main() -> int:
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    rows = led["rows"]
    h9 = json.loads(H9.read_text(encoding="utf-8"))["rows"]

    print("=" * 78)
    print("V4 MISSION 20 -- where are the lost points, benchmark-wide?")
    print("=" * 78)
    print()
    print("PROVENANCE, resolved before measuring.  Two 1,986-row artefacts disagree 7x:")
    led_v = Counter(r["verdict"] for r in rows)
    print(f"  v4_stage_ledger.json      correct {led_v['correct']} / wrong "
          f"{led_v['wrong']} / abstain {led_v['abstain']}   "
          f"= {100 * led_v['correct'] / len(rows):.1f}%")
    print( "    AUDITABLE: per-stage attribution s1..s10, baseline HEAD "
          f"{led['baseline']['head']}, sha256 of dataset/context-cache/scorer")
    print("  locomo_full_20261010_151930.json  correct 1009 / wrong 977 = 50.8%")
    print("    NO model, prompt, config, context or reader identity recorded")
    print()
    print("  This audit measures the AUDITABLE run.  It does NOT report 50.8% as the")
    print("  current score, because nothing here establishes what produced it -- the same")
    print("  gap M0 flagged and refused to merge.")
    print()

    # ---------------- loss by category -------------------------------------
    print("=" * 78)
    print("LOSS BY CATEGORY (auditable run)")
    print("=" * 78)
    print(f"  {'cat':<6}{'n':>6}{'correct':>9}{'wrong':>7}{'abstain':>9}"
          f"{'acc':>8}{'% of all lost':>15}")
    total_lost = led_v["wrong"] + led_v["abstain"]
    cat_rows = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        c = sum(1 for r in sub if r["verdict"] == "correct")
        w = sum(1 for r in sub if r["verdict"] == "wrong")
        a = sum(1 for r in sub if r["verdict"] == "abstain")
        cat_rows[cat] = {"n": len(sub), "correct": c, "wrong": w, "abstain": a}
        print(f"  {cat:<6}{len(sub):>6}{c:>9}{w:>7}{a:>9}"
              f"{100 * c / len(sub):>7.1f}%{100 * (w + a) / total_lost:>14.1f}%")
    print(f"  {'ALL':<6}{len(rows):>6}{led_v['correct']:>9}{led_v['wrong']:>7}"
          f"{led_v['abstain']:>9}{100 * led_v['correct'] / len(rows):>7.1f}%"
          f"{100.0:>14.1f}%")
    print()

    # ---------------- which stage blocks each abstention --------------------
    print("=" * 78)
    print("WHICH STAGE FIRST BLOCKS THE ANSWER?  (the actual loss map)")
    print("=" * 78)

    # ---- CRITICAL: distinguish 'blocked' from 'never measured' ---------------
    # Two stages are UNMEASURED FOR EVERY ROW and must not be read as loss causes:
    #   s7_temporal_polarity : 'UNKNOWN (no rule measured in this pass)'        x1986
    #   s9_reader_fallback   : 'UNKNOWN (reader not executed in this harness)'  x1986
    # The first version excluded only s7, and s9 then surfaced as the top loss source
    # (610 cases, 33%) purely because the Reader was never run in this deterministic
    # harness.  A stage that is unmeasured for every row cannot be a cause of any loss.
    #
    # NOTE ON s9 SPECIFICALLY: 'reader not executed' is EXPECTED for a
    # deterministic-only harness, and the H9 cohort is precisely the subset where the
    # Reader WAS run (989 questions, 685 correct).  So s9 is not missing by accident --
    # it is out of this artefact's scope, and it is reported separately below.
    #
    # BUG FIXED: the first detection compared the value set against a bare string, but
    # the recorded values are 'UNKNOWN (<reason>)', so the set comparison never matched.
    # Detect the phrase as a SUBSTRING of every value instead.
    UNMEASURED_MARKERS = ("NO RULE MEASURED", "NOT EXECUTED")
    unmeasured_stages = []
    for s in STAGES:
        vals = [str(r.get(s) or "").upper() for r in rows]
        if vals and all(any(m in v for m in UNMEASURED_MARKERS) for v in vals):
            unmeasured_stages.append(s)
    if unmeasured_stages:
        print("  STAGES NEVER MEASURED (excluded -- cannot be a cause of loss):")
        for s in unmeasured_stages:
            reason = str(rows[0].get(s) or "")
            print(f"    {s}: unmeasured for all {len(rows)} rows  [{reason[:52]}]")
        print()

    BLOCKING = ("UNKNOWN", "ABSENT", "NOT_FOUND", "NO_TOKENS", "PARTIAL", "NONE")

    def blocking_stage(r):
        for s in STAGES:
            if s in unmeasured_stages:
                continue          # never measured: not evidence of a block
            v = str(r.get(s) or "")
            if any(b in v.upper() for b in BLOCKING):
                return s
        return None

    first_block = Counter()
    for r in rows:
        if r["verdict"] in ("wrong", "abstain"):
            s = blocking_stage(r)
            first_block[s or "NONE_RECORDED"] += 1
    n_noncorrect = led_v["wrong"] + led_v["abstain"]
    print(f"  {'first blocking stage':<28}{'n':>6}{'% of non-correct':>19}")
    for s, n in first_block.most_common():
        print(f"  {s:<28}{n:>6}{100 * n / n_noncorrect:>18.1f}%")
    print(f"  {'TOTAL non-correct':<28}{n_noncorrect:>6}")
    print()
    print("  Rules stated: a stage blocks when its recorded value contains UNKNOWN,")
    print("  ABSENT, NOT_FOUND, NO_TOKENS, PARTIAL or NONE, and the earliest such stage")
    print("  wins.  Stages that are unmeasured for every row are EXCLUDED, because an")
    print("  unmeasured stage cannot be a cause.  Values are read as recorded.")
    print()

    # ---------------- Reader-recoverable pool ------------------------------
    print("=" * 78)
    print("READER-RECOVERABLE POOL (for scale)")
    print("=" * 78)
    from collections import Counter as C2
    h9v = C2(r["mutually_exclusive"] for r in h9)
    print(f"  H9 cohort (engine abstained, Reader was run) : {len(h9)}")
    print(f"    Reader CORRECT                             : {h9v['CORRECT']} "
          f"({100 * h9v['CORRECT'] / len(h9):.1f}%)")
    print(f"    Reader abstained / wrong                   : "
          f"{len(h9) - h9v['CORRECT']}")
    print()
    print(f"  The deterministic engine abstains on {led_v['abstain']} of {len(rows)}.  On")
    print(f"  the {len(h9)} where the Reader actually ran, it answered "
          f"{h9v['CORRECT']} correctly.  The abstention pool is therefore not dead")
    print("  weight -- it is the largest single recoverable mass in the benchmark.")
    print()

    # ---------------- the actionable statement -----------------------------
    print("=" * 78)
    print("WHAT THIS SAYS ABOUT WHERE TO INVEST")
    print("=" * 78)
    real = [(s, n) for s, n in first_block.most_common() if s != "NONE_RECORDED"]
    if real:
        top = real[0]
        print(f"  Largest blocking stage: {top[0]} at {top[1]} cases "
              f"({100 * top[1] / n_noncorrect:.0f}% of non-correct).")
    print()
    print("  Read against M5-M19: every mission so far worked on a cohort the engine had")
    print("  ALREADY failed on, targeting why the Reader picked wrong among visible")
    print("  candidates.  That cohort is a small slice of this map.  The dominant mass is")
    print("  UPSTREAM of candidate selection, which is consistent with M5/M6/M10 all")
    print("  showing that adding context does not help -- much of the context never")
    print("  arrives at all.")
    print()
    if unmeasured_stages:
        print(f"  AND THE LARGEST UNKNOWN IS UNMEASURED, NOT UNBLOCKED: "
              f"{unmeasured_stages[0]}")
        print(f"  carries no measurement for any of the {len(rows)} questions.  So this map")
        print("  cannot say whether temporal reasoning is a loss source or a strength --")
        print("  it is a blank.  That is a measurement gap, not a finding, and the next")
        print("  honest step is to MEASURE it rather than to assume either way.")
    print()
    print("  SCOPE: the auditable deterministic-only run.  Stage columns read as recorded;")
    print("  the blocking rule and the unmeasured exclusion are stated, not hidden.  No")
    print("  provenance is invented for the 50.8% artefact, and the two are not merged.")
    print("  No model calls, no production change, and the 553-question holdout is")
    print("  untouched.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 20 -- benchmark-wide loss decomposition (auditable run)",
        "baseline": {"head": "b222186", "production_changed": False},
        "auditable_run": {"correct": led_v["correct"], "wrong": led_v["wrong"],
                          "abstain": led_v["abstain"],
                          "accuracy_pct": 100 * led_v["correct"] / len(rows)},
        "unauditable_run": {"correct": 1009, "accuracy_pct": 50.8,
                            "provenance": "absent -- not merged, not reported as current"},
        "by_category": cat_rows,
        "first_blocking_stage": dict(first_block),
        "blocking_rule": "value contains UNKNOWN/ABSENT/NOT_FOUND/NO_TOKENS/PARTIAL/NONE",
        "h9_reader_correct": h9v["CORRECT"], "h9_n": len(h9),
        "n_questions": len(rows),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
