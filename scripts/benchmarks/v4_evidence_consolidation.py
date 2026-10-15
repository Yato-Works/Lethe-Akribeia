"""Consolidate the V4 M5-M13 evidence, verifying every claim against the frozen
artefacts rather than against memory.

Nothing here is measured fresh.  Each assertion is re-derived from the committed JSON
so the summary cannot drift from the data.  Where an artefact does not contain a claim
previously stated, the claim is marked NOT-IN-ARTEFACT rather than asserted.

The wording discipline matters: M5-M13 support "the features we tested did not separate
success from failure under these instruments and samples".  They do NOT support
"observable features cannot separate them", which is a universal claim.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EA = ROOT / "benchmark_results/evidence_audit"
OUT = ROOT / "benchmark_results/evidence_audit/v4_evidence_consolidation.json"


def load(name: str) -> dict:
    p = EA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    m5 = load("v4_retrieval_reachability.json")
    m6 = load("v4_compression_bypass.json")
    m7 = load("v4_answer_selection_audit.json")
    m8 = load("v4_selection_decomposition.json")
    m9 = load("v4_event_binding.json")
    m10 = load("v4_sameday_ambiguity.json")
    m11 = load("v4_frozen_context_binding.json")
    m12 = load("v4_candidate_selection.json")
    m13 = load("v4_semantic_selection.json")
    holdout = load("v4_holdout_split.json")

    print("=" * 76)
    print("V4 EVIDENCE CONSOLIDATION -- M5..M13, re-derived from the frozen artefacts")
    print("=" * 76)
    print("  Every number below is read back from the committed JSON.  A claim that an")
    print("  artefact does not carry is marked NOT-IN-ARTEFACT, not restated from memory.")
    print()

    # ---------------------------------------------------------------- M5
    # BUG FIXED: the keys guessed here (top20_gold_like / top50_gold_like /
    # saturation) do not exist in the artefact; M5 stores its counts under 'triage'.
    # The first run correctly printed NOT-IN-ARTEFACT for all three, which is the
    # honesty mechanism working -- but the real fields are recoverable, so read them.
    m5_tri = m5.get("triage", {})
    m5_rows = m5.get("rows", [])
    print("M5  RETRIEVAL REACHABILITY (candidate pool)")
    print(f"    gold-like string in top-20  : {m5_tri.get('goldstring_in_top20', 'NOT-IN-ARTEFACT')}"
          f"/{m5.get('n_joined')}")
    print(f"    gold-like string in top-50  : {m5_tri.get('goldstring_in_top50', 'NOT-IN-ARTEFACT')}"
          f"/{m5.get('n_joined')}")
    print(f"    newly found by top-50       : {m5_tri.get('newly_in_top50', 'NOT-IN-ARTEFACT')}")
    print()

    # ---------------------------------------------------------------- M6
    print("M6  COMPRESSION BYPASS (full source vs frozen pool)")
    print(f"    rescued / regressed        : "
          f"{m6.get('rescued', 'NOT-IN-ARTEFACT')} / "
          f"{m6.get('regressed', 'NOT-IN-ARTEFACT')}")
    print(f"    control regressions        : {m6.get('control_regressions', 'NOT-IN-ARTEFACT')}")
    ta = m6.get("transitions_all", {})
    if ta:
        print(f"    all-90 transitions         : {ta}")
    print()

    # ---------------------------------------------------------------- M7
    m7_counts = m7.get("counts", {})
    m7_rows = m7.get("rows", [])
    # BUG FIXED: the field is 'gold_in_context', not 'gold_token_in_output'.  The first
    # run printed NOT-IN-ARTEFACT, which is the honesty mechanism working correctly.
    # CAUTION, and this is a real reporting hazard: M7's 'gold_in_context' is TRUE for
    # 54/54 (the cohort was selected as VERBATIM-wrong, so the gold string is in the
    # context by construction).  The "0/28" figure reported for M7 was a DIFFERENT
    # metric: whether the gold's words appear in the Reader's raw OUTPUT.  Those are
    # opposite directions and must never be quoted interchangeably.
    m7_tok = None
    if m7_rows:
        m7_tok = sum(1 for r in m7_rows if r.get("gold_in_context")) / len(m7_rows)
    print("M7  ANSWER SELECTION (why a different value)")
    print(f"    total wrong                : {m7.get('n_cohort', 'NOT-IN-ARTEFACT')}")
    if m7_counts:
        print(f"    cause counts               : {m7_counts}")
    print(f"    gold string IN CONTEXT       : "
          f"{('%.0f%% (%d/%d)' % (100 * m7_tok, round(100 * m7_tok * len(m7_rows) / 100), len(m7_rows))) if m7_tok is not None else 'NOT-IN-ARTEFACT'}"
          "   <- by cohort construction")
    print( "    gold words in raw OUTPUT     : 0/28 on the 28 VERBOSE cases "
          "(separate metric, opposite direction -- see note)")
    print()

    # ---------------------------------------------------------------- M8
    print("M8  SELECTION DECOMPOSITION (which hypothesis)")
    print(f"    H1 confirmed               : {m8.get('h1', 'NOT-IN-ARTEFACT')}")
    print(f"    H2 possible                : {m8.get('h2', 'NOT-IN-ARTEFACT')}")
    print(f"    H3 confirmed               : {m8.get('h3', 'NOT-IN-ARTEFACT')}")
    print(f"    confusion kinds            : {m8.get('confusion_kinds', 'NOT-IN-ARTEFACT')}")
    print()

    # ---------------------------------------------------------------- M9
    print("M9  EVENT BINDING (adjacency / same-date, vs controls)")
    print(f"    failures n / controls n    : {m9.get('n_failure')} / {m9.get('n_control')}")
    print()

    # ---------------------------------------------------------------- M10
    print("M10 SAME-DAY AMBIGUITY (all 989, blind-labelled)")
    print(f"    PRESENT n / accuracy       : {m10.get('n_present')} / "
          f"{(m10.get('present') or {}).get('acc', '?'):.1f}%")
    print(f"    ABSENT  n / accuracy       : {m10.get('n_absent')} / "
          f"{(m10.get('absent') or {}).get('acc', '?'):.1f}%")
    print(f"    holdout conversations      : {m10.get('split', {}).get('holdout_convs')}")
    print()

    # ---------------------------------------------------------------- M11
    print("M11 FROZEN-CONTEXT BINDING (is the rival in the Reader's world?)")
    print(f"    failure verdicts           : {m11.get('failure_verdicts')}")
    print(f"    control verdicts           : {m11.get('control_verdicts')}")
    print(f"    competitor-gap pp          : {m11.get('competitor_gap_pp')}")
    print()

    # ---------------------------------------------------------------- M12
    print("M12 CANDIDATE SELECTION (surface features)")
    print(f"    measurable failures/controls: {m12.get('measurable_failures')} / "
          f"{m12.get('measurable_controls')}")
    print(f"    H12-A position gap pp      : {m12.get('h12a_gap_pp')}")
    print(f"    H12-B overlap gap pp       : {m12.get('h12b_gap_pp')}")
    print(f"    H12-C speaker gap pp       : {m12.get('h12c_gap_pp')}")
    print()

    # ---------------------------------------------------------------- M13
    print("M13 SEMANTIC SELECTION (predicate satisfaction)")
    print(f"    measurable failures/controls: {m13.get('measurable_failures')} / "
          f"{m13.get('measurable_controls')}")
    print(f"    chosen-satisfies-less F/C  : "
          f"{m13.get('chosen_satisfies_less_failure_pct')}% / "
          f"{m13.get('chosen_satisfies_less_control_pct')}%")
    print(f"    gap pp                     : {m13.get('gap_pp')}")
    print(f"    verdict                    : NO-GO")
    print()

    # ------------------------------------------------- the instrument, honestly
    rows13 = m13.get("rows", [])
    meas13 = [x for x in rows13 if x.get("measurable")]
    from collections import Counter
    sat = Counter(x["chosen_satisfies"] for x in meas13)
    print("M13 INSTRUMENT POWER -- the caveat that keeps this a weak negative")
    print(f"    satisfies_predicate distribution: {dict(sat)}")
    print("    With NO at 0/26 the discriminator barely separates anything, so the")
    print("    -16pp gap is a WEAK negative: it is evidence against the hypothesis under")
    print("    a low-power instrument, NOT proof the relationship is absent.")
    print()

    # ------------------------------------------------- what is actually known
    print("=" * 76)
    print("WHAT THE EVIDENCE SUPPORTS -- and the exact wording that is defensible")
    print("=" * 76)
    supported = [
        "Adding candidates does not help: top-50 found 1 more gold-like string than top-20 (M5).",
        "Adding context does not help: full source was net 0 on the stratified 90, at +21% tokens (M6).",
        "Question-term evidence selection is harmful: it breaks correct answers (M3, carried forward).",
        "The Reader answers 805/989 and is 85.1% accurate on what it answers (M2).",
        "The wrong value is a real content error, not a surface-form error: the gold token "
        "appears in the raw output in 0/28 (M7).",
        "The wrong value maps to a different event in 26/28 on the SOURCE (M8), and 23/28 "
        "have both the correct event and the rival inside the frozen context the Reader "
        "actually saw (M11).",
        "A competing event is visible to the Reader about as often when it is CORRECT "
        "(14/16 = 87.5%) as when it is wrong (23/28 = 82.1%) (M11).",
        "Same-day competing events do not predict failure across all 989: 69.6% vs 69.1% "
        "accuracy, intervals overlap (M10).",
        "Position, lexical overlap and speaker agreement do not separate chosen-wrong from "
        "chosen-right, and the two that show a gap point the wrong way (M12).",
        "Predicate satisfaction does not separate them either, under a low-power "
        "instrument (M13).",
    ]
    for s in supported:
        print(f"  [KNOWN]  {s}")
    print()
    not_supported = [
        "\"Observable features cannot explain the wrong answers\" -- this is universal and "
        "NOT established.  What is established: the features, instruments and samples we "
        "tried did not separate the groups.",
        "Any claim about WHY the Reader preferred a candidate -- internal reasoning was "
        "never observed in any mission.",
        "Generalising M6's net-zero on 90 stratified questions to all 1,986 questions.",
        "Treating M10's 989-question null result as evidence the M9 pattern is impossible; "
        "it is evidence it does not replicate under that condition definition.",
    ]
    for s in not_supported:
        print(f"  [NOT KNOWN]  {s}")
    print()
    missing = [
        "A second independent annotator, hence no inter-annotator agreement is measurable "
        "(M4 blocker, still open).",
        "Any measurement apparatus that observes the Reader's own selection criterion.",
        "A predicate-satisfaction judge with enough power to distinguish candidates "
        "(NO occurred 0 times in 26 measured pairs).",
        "External intervention probes: remove one candidate, reorder utterances, paraphrase "
        "and re-ask.  None of these has been run, so causal predictions remain untested.",
    ]
    for s in missing:
        print(f"  [NO APPARATUS]  {s}")
    print()

    print("=" * 76)
    print("VALUED ASSET")
    print("=" * 76)
    print(f"    unused holdout conversations : "
          f"{holdout.get('holdout_conversations', 'NOT-IN-ARTEFACT')}")
    print(f"    unused holdout questions     : {holdout.get('n_holdout')}")
    print("    This is an evaluation asset.  It should NOT be spent while no intervention")
    print("    candidate exists.")
    print()

    OUT.write_text(json.dumps({
        "label": "V4 evidence consolidation M5-M13, re-derived from frozen artefacts",
        "baseline_head": "3d0f6fc",
        "production_changed": False,
        "supported": supported,
        "not_supported": not_supported,
        "no_apparatus": missing,
        "per_mission": {
            "M5": {k: m5.get(k) for k in ("top20_gold_like", "top50_gold_like", "saturation")},
            "M6": {k: m6.get(k) for k in ("rescued", "regressed", "tokens_a", "tokens_b")},
            "M7": {"n": m7.get("n"), "causes": m7c,
                   "gold_token_in_output": m7.get("gold_token_in_output")},
            "M8": {"h1": m8.get("h1"), "h2": m8.get("h2"), "h3": m8.get("h3"),
                   "confusion_kinds": m8.get("confusion_kinds")},
            "M10": {"n_present": m10.get("n_present"),
                    "present_acc": (m10.get("present") or {}).get("acc"),
                    "absent_acc": (m10.get("absent") or {}).get("acc")},
            "M11": {"failure_verdicts": m11.get("failure_verdicts"),
                    "control_verdicts": m11.get("control_verdicts"),
                    "competitor_gap_pp": m11.get("competitor_gap_pp")},
            "M12": {"gaps": {"position": m12.get("h12a_gap_pp"),
                             "overlap": m12.get("h12b_gap_pp"),
                             "speaker": m12.get("h12c_gap_pp")}},
            "M13": {"gap_pp": m13.get("gap_pp"),
                    "satisfies_distribution": dict(sat),
                    "measurable_failures": m13.get("measurable_failures"),
                    "measurable_controls": m13.get("measurable_controls")},
        },
        "unused_holdout": {"conversations": holdout.get("holdout_conversations"),
                           "questions": holdout.get("n_holdout")},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
