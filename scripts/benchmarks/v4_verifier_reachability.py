"""V4 Mission 14 -- does the already-proven answer_verifier reach the M11 failures?

A GATE, NOT A PROOF OF EFFECTIVENESS.  The question is reachability and discriminating
power, not "the guard works".

Frozen inputs, nothing re-derived:
  cohort  = M11's 23 FAILURE (both gold and rival visible in the frozen context)
            + M11's 14 CONTROL (correct event + a competitor both visible)
  context = the frozen context the Reader was actually given (sha-locked artefact)
  answer  = the frozen Reader prediction, verbatim

The engine under test is `AnswerVerifier.check_subject_binding(question, answer, context)`.
It is plain string-in / string-out, so NO Turn -> UnifiedProposition adapter is needed --
which is itself a finding, because the type mismatch identified in the stage-1 audit does
not block this entry point.

What is recorded per case, and why:
  fired              : did the guard refuse / rewrite this answer at all
  refusal or rewrite : which of the two, since they have different downstream meaning
  reason             : which guard branch fired (entity-presence vs subject-binding)
  input integrity    : did _parse_turns recover >= 2 speakers from OUR context format,
                       because the guard silently no-ops below 2 turns
  group              : FAILURE or CONTROL -- joined AFTER firing is measured

DECISION RULE, fixed in the document before the run:
  fires on failures, ~never on controls  -> worth proceeding
  fires about equally on both            -> detects but does not discriminate
  does not fire on failures             -> low transferability, only if inputs verified
  inputs lost in adaptation             -> performance verdict withheld, fix the contract
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

M11 = ROOT / "benchmark_results/evidence_audit/v4_frozen_context_binding.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_verifier_reachability.json"


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from artificial_memory_gen.recall.answer_verifier import AnswerVerifier

    m11 = json.loads(M11.read_text(encoding="utf-8"))
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    fails = [r for r in m11["rows"]
             if r["group"] == "FAILURE" and r["verdict"] == "GOLD_AND_PRED_EVENT"]
    ctrls = [r for r in m11["rows"]
             if r["group"] == "CONTROL" and r["verdict"] == "GOLD_AND_COMPETITOR_VISIBLE"]

    verifier = AnswerVerifier(subject_binding=True)

    print("=" * 76)
    print("V4 MISSION 14 -- does check_subject_binding reach the M11 failures?")
    print("=" * 76)
    print("  A GATE, not a proof.  Measures reachability and discriminating power only.")
    print(f"  FAILURE cohort (frozen from M11) : {len(fails)}")
    print(f"  CONTROL cohort (frozen from M11) : {len(ctrls)}")
    print("  engine : AnswerVerifier.check_subject_binding(question, answer, context)")
    print("  NOTE   : string-in / string-out, so NO IR adapter is required for this entry")
    print("           point -- the type mismatch from the stage-1 audit does not block it.")
    print()

    # ---------- input-integrity check FIRST, because the guard no-ops < 2 turns --
    print("INPUT INTEGRITY (must be verified before any verdict about the engine)")
    probe = AnswerVerifier()
    integrity = []
    for r in fails + ctrls:
        cq = (meta.get(r["question_id"]) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        turns = probe._parse_turns(context) if context else []
        speakers = {s.lower() for s, _ in turns}
        integrity.append({
            "question_id": r["question_id"], "group": r["group"],
            "has_context": bool(context),
            "n_turns_parsed": len(turns),
            "n_speakers": len(speakers),
            "speakers": sorted(speakers),
            "guard_would_noop": len(turns) < 2,
        })
    ok_parse = [x for x in integrity if not x["guard_would_noop"]]
    print(f"  contexts located              : {sum(1 for x in integrity if x['has_context'])}"
          f"/{len(integrity)}")
    print(f"  >=2 turns parsed (guard runs) : {len(ok_parse)}/{len(integrity)}"
          f"  <- below 2 the guard silently returns None")
    sp = [x["n_speakers"] for x in ok_parse]
    if sp:
        print(f"  speakers per context: min {min(sp)}  median "
              f"{sorted(sp)[len(sp) // 2]}  max {max(sp)}")
    lost = [x for x in integrity if x["guard_would_noop"]]
    if lost:
        print(f"  INPUT LOST for {len(lost)} cases -> "
              "performance verdict must be WITHHELD per the decision rule")
    print()

    # ---------- measure firing, per group --------------------------------------
    rows = []
    for r in fails + ctrls:
        qid = r["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        # the answer the Reader actually produced, verbatim from the frozen artefact
        answer = (r.get("prediction") or "").strip()
        if not context:
            rows.append({"question_id": qid, "group": r["group"], "measured": False,
                         "why": "no frozen context"})
            continue
        res = verifier.check_subject_binding(r["question"], answer, context)
        fired = res is not None
        rec = {
            "question_id": qid, "group": r["group"], "measured": True,
            "answer": answer[:120],
            "fired": fired,
            "verdict": None, "reason": None,
            "hallucination_detected": None,
        }
        if fired:
            rec["verdict"] = res.verified_answer
            rec["reason"] = res.notes
            rec["hallucination_detected"] = res.hallucination_detected
            # distinguish refusal from rewrite -- different downstream meaning
            low = (res.verified_answer or "").lower()
            rec["outcome_kind"] = ("REFUSAL"
                                   if any(m in low for m in
                                          ("none", "not mentioned", "i don't know"))
                                   else "REWRITE")
        rows.append(rec)

    fm = [x for x in rows if x["measured"] and x["group"] == "FAILURE"]
    cm = [x for x in rows if x["measured"] and x["group"] == "CONTROL"]
    unmeas = [x for x in rows if not x["measured"]]

    def pct(k, n):
        return f"{100 * k / n:.1f}%" if n else "n/a"

    print("=" * 76)
    print("FIRING RATE  (control group joined only now, after firing was measured)")
    print("=" * 76)
    print(f"  {'group':<10}{'fired':>8}{'/n':>6}{'rate':>9}   {'refusal':>8}{'rewrite':>8}")
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        f = sum(1 for x in g if x["fired"])
        ref = sum(1 for x in g if x.get("outcome_kind") == "REFUSAL")
        rew = sum(1 for x in g if x.get("outcome_kind") == "REWRITE")
        print(f"  {label:<10}{f:>8}{len(g):>6}{pct(f, len(g)):>9}   {ref:>8}{rew:>8}")
    if unmeas:
        print(f"  unmeasured (kept, not dropped) : {len(unmeas)}")
    print()

    f_fire = sum(1 for x in fm if x["fired"])
    c_fire = sum(1 for x in cm if x["fired"])
    gap = (100 * f_fire / len(fm) - 100 * c_fire / len(cm)) if (fm and cm) else 0.0
    print(f"  gap (FAILURE - CONTROL) : {gap:+.1f}pp")
    print()

    # ---------- which guard branch fired --------------------------------------
    print("=" * 76)
    print("WHICH BRANCH FIRED (so we know WHICH PART of the +28 is even in play)")
    print("=" * 76)
    kinds = Counter()
    for x in fm + cm:
        if x["fired"]:
            note = (x.get("reason") or "").lower()
            if "entity-presence" in note:
                kinds["entity-presence guard"] += 1
            elif "subject" in note or "binding" in note:
                kinds["subject-binding guard"] += 1
            else:
                kinds["other"] += 1
    for k, v in kinds.most_common():
        print(f"  {k:<32} {v}")
    if not kinds:
        print("  (no firing at all)")
    print()

    # ---------- conversation spread -------------------------------------------
    print("=" * 76)
    print("CONVERSATION SPREAD of the firing failures")
    print("=" * 76)
    conv_of = {r["question_id"]: r["conversation"] for r in m11["rows"]}
    fired_conv = Counter(conv_of.get(x["question_id"]) for x in fm if x["fired"])
    if fired_conv:
        print("  " + ", ".join(f"{k}:{v}" for k, v in sorted(fired_conv.items())))
    else:
        print("  (none)")
    print()

    # ---------- DECISION RULE, pre-declared -----------------------------------
    print("=" * 76)
    print("DECISION RULE (fixed before the run)")
    print("=" * 76)
    if lost:
        print(f"  INPUT LOST for {len(lost)} cases.  Per the pre-declared rule the")
        print("  performance verdict is WITHHELD until the data contract is repaired.")
        print("  Fixing the contract comes before any claim about the engine.")
    else:
        f_rate = 100 * f_fire / len(fm) if fm else 0
        c_rate = 100 * c_fire / len(cm) if cm else 0
        if f_rate >= 30 and gap >= 20:
            print(f"  Fires on {f_rate:.0f}% of failures vs {c_rate:.0f}% of controls "
                  f"({gap:+.0f}pp).")
            print("  -> WORTH PROCEEDING to a measurement path.  Note this is still only")
            print("     reachability + discrimination; it is not evidence of a score gain.")
        elif f_rate < 10:
            print(f"  Fires on only {f_rate:.0f}% of failures, with inputs verified intact.")
            print("  -> LOW TRANSFERABILITY on this path.  The +28 net was measured on a")
            print("     different row set; it does not reach the M11 failure shape.")
            print("     This does NOT reject answer_verifier wholesale -- it says this")
            print("     entry point does not cover these cases.")
        else:
            print(f"  Fires on {f_rate:.0f}% of failures and {c_rate:.0f}% of controls "
                  f"({gap:+.0f}pp).")
            print("  -> DETECTS BUT DOES NOT DISCRIMINATE.  Extension is not urgent; a")
            print("     guard that refuses correct answers as often is not an improvement.")
    print()
    print("  SCOPE, stated once: firing here means the guard refused or rewrote the")
    print("  Reader's OUTPUT.  It says nothing about whether the guard would fix selection")
    print("  between two visible candidates -- M11 showed that gap and this guard does")
    print("  not address it.  And the Reader's internal reasoning is still not observed.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 14 -- answer_verifier reachability on the M11 cohort (a gate)",
        "baseline": {"head": "b328c9a", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "cohort_from": "v4_frozen_context_binding.json (M11), frozen, not regenerated",
        "engine": "AnswerVerifier.check_subject_binding(question, answer, context)",
        "adapter_required": False,
        "n_failure_cohort": len(fails), "n_control_cohort": len(ctrls),
        "input_integrity": integrity,
        "input_lost_cases": len(lost),
        "failures_measured": len(fm), "controls_measured": len(cm),
        "failures_fired": f_fire, "controls_fired": c_fire,
        "gap_pp": gap,
        "branch_counts": dict(kinds),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
