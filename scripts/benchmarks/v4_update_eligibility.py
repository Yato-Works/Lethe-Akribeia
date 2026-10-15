"""V4 Mission 18, Stage 1 -- are update/correction relations even PRESENT in the failures?

An eligibility audit, NOT an engine.  No production change, no wiring, no model calls:
this reads frozen artefacts only and classifies the 30 measurable M15/M17 cases by what
relation actually holds between the two competing utterances.

THE THREE RELATIONS ARE DELIBERATELY KEPT DISTINCT, because collapsing them is exactly the
"plausible principle" trap:
  EXPLICIT_CORRECTION  one utterance retracts/replaces the other ("I was wrong, it's X",
                      "actually it's not X, it's Y")
  STATE_UPDATE         a state change over time ("I used to X, now I Y", "I've moved on")
  PLAIN_CONFLICT       two different facts, but NEITHER supersedes the other
  NO_RELATION          the two utterances do not concern the same fact at all

"Newer therefore supersedes older" is NOT assumed.  That assumption is the thing under
test, so it is never used to do the labelling.

CLASSIFICATION IS BLIND TO THE OUTCOME: the label depends only on the two utterances and
the question, never on whether the answer was right or wrong.  gold, prediction and the
correctness label are joined only after the labels are frozen, so "it failed, so it must
have had a correction" cannot happen.

Every label carries the utterance text it was derived from, so a reviewer can disagree
with the rule rather than trust it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = ROOT / "benchmark_results/evidence_audit/v4_order_probe_scored.json"
M11 = ROOT / "benchmark_results/evidence_audit/v4_frozen_context_binding.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_update_eligibility.json"

# ---- retraction / correction markers (FIXED BEFORE LABELLING) -------------------
_CORRECTION = re.compile(
    r"\b(?:i\s+was\s+wrong|i\s+was\s+mistaken|i\s+made\s+a\s+mistake|correction|"
    r"to\s+be\s+clear|what\s+i\s+meant\s+was|actually\s+it(?:'s|\s+is)\s+not|"
    r"scratch\s+that|never\s*mind|forget\s+what\s+i\s+said|"
    r"i\s+take\s+that\s+back|that\s+was\s+incorrect|i\s+was\s+confused)\b", re.I)
# ---- state-change-over-time markers --------------------------------------------
_UPDATE = re.compile(
    r"\b(?:used\s+to|no\s+longer|not\s+anymore|anymore|these\s+days|now\s+i|"
    r"i(?:'ve|\s+have)\s+(?:since\s+)?(?:moved\s+on|switched|changed|quit|stopped|"
    r"started|taken\s+up|grown\s+out\s+of)|"
    r"ever\s+since|since\s+then|at\s+first\s+.{0,24}\s+but\s+now|"
    r"formerly|previously|in\s+the\s+past\s+i)\b", re.I)
_HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")
_QUOTE = re.compile(r'^(?:\(In reply to[^)]*\)\s*)?(?:"[^"]*"\s*)?([A-Z][a-z]+):')


def main() -> int:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    m11 = json.loads(M11.read_text(encoding="utf-8"))
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    by11 = {r["question_id"]: r for r in m11["rows"]}
    cases = [x for x in probe["rows"] if x["status"] == "MEASURED"]

    print("=" * 78)
    print("V4 MISSION 18 STAGE 1 -- eligibility: do update/correction relations EXIST?")
    print("=" * 78)
    print("  An audit, NOT an engine.  No model calls, no production change.")
    print("  Three relations kept DISTINCT on purpose:")
    print("    EXPLICIT_CORRECTION  one utterance retracts/replaces the other")
    print("    STATE_UPDATE         a state change over time (used to X, now Y)")
    print("    PLAIN_CONFLICT       two different facts, NEITHER superseding the other")
    print("    NO_RELATION          the two do not concern the same fact at all")
    print()
    print("  \"Newer supersedes older\" is NOT assumed -- that is the thing under test, so")
    print("  it is never used to produce the label.")
    print()
    print("  BLIND: the label depends only on the two utterances and the question, never")
    print("  on whether the answer was right.  gold / prediction / correctness are joined")
    print("  only after the labels are frozen.")
    print()

    # ---------------- blind labelling ---------------------------------------
    labelled = []
    for x in cases:
        qid = x["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        a, b = x["pair"]
        lines = [l for l in context.split("\n") if l.strip()]
        body = {}
        for l in lines:
            m = _HDR.match(l)
            if not m:
                continue
            rest = l[m.end():]
            ms = _QUOTE.match(rest)
            body[m.group(1)] = (rest[ms.end():] if ms else rest)
        ta, tb = body.get(a, ""), body.get(b, "")

        a_cor, b_cor = bool(_CORRECTION.search(ta)), bool(_CORRECTION.search(tb))
        a_upd, b_upd = bool(_UPDATE.search(ta)), bool(_UPDATE.search(tb))

        if a_cor or b_cor:
            rel = "EXPLICIT_CORRECTION"
            ev = ta if a_cor else tb
        elif a_upd or b_upd:
            rel = "STATE_UPDATE"
            ev = ta if a_upd else tb
        else:
            # do the two utterances even concern the same fact?  Blind proxy: do they
            # share content words at all?  If not, there is no relation to resolve.
            stop = {"the", "and", "that", "this", "with", "for", "was", "were", "have",
                    "has", "had", "they", "their", "them", "she", "her", "his", "him",
                    "you", "your", "our", "its", "about", "there", "here", "what",
                    "when", "where", "which", "from", "been", "very", "just", "also"}
            wa = {w for w in re.findall(r"[a-z]{4,}", ta.lower()) if w not in stop}
            wb = {w for w in re.findall(r"[a-z]{4,}", tb.lower()) if w not in stop}
            shared = wa & wb
            rel = "PLAIN_CONFLICT" if shared else "NO_RELATION"
            ev = ""
        labelled.append({
            "question_id": qid, "relation": rel,
            "a_turn": a, "b_turn": b,
            "a_text": ta[:150], "b_text": tb[:150],
            "evidence_marker_text": ev[:150],
            "a_correction_marker": a_cor, "b_correction_marker": b_cor,
            "a_update_marker": a_upd, "b_update_marker": b_upd,
        })

    dist = Counter(x["relation"] for x in labelled)
    print("=" * 78)
    print("RELATION DISTRIBUTION (blind, over the 30 measurable cases)")
    print("=" * 78)
    for k in ("EXPLICIT_CORRECTION", "STATE_UPDATE", "PLAIN_CONFLICT", "NO_RELATION"):
        print(f"  {k:<22} {dist.get(k, 0):>3}  ({100 * dist.get(k, 0) / len(labelled):.0f}%)")
    print(f"  {'TOTAL':<22} {len(labelled):>3}")
    print()

    # ---------------- join outcome ONLY NOW ---------------------------------
    outcome = {x["question_id"]: x["transition"] for x in cases}
    group = {x["question_id"]: x["group"] for x in cases}
    for x in labelled:
        x["group"] = group.get(x["question_id"])
        x["transition"] = outcome.get(x["question_id"])

    print("=" * 78)
    print("RELATION x OUTCOME (outcome joined only after labelling was frozen)")
    print("=" * 78)
    print(f"  {'relation':<22}{'FAILURE':>9}{'CONTROL':>9}")
    for rel in ("EXPLICIT_CORRECTION", "STATE_UPDATE", "PLAIN_CONFLICT", "NO_RELATION"):
        f = sum(1 for x in labelled if x["relation"] == rel and x["group"] == "FAILURE")
        c = sum(1 for x in labelled if x["relation"] == rel and x["group"] == "CONTROL")
        print(f"  {rel:<22}{f:>9}{c:>9}")
    print()

    # ---------------- the eligibility gate ----------------------------------
    print("=" * 78)
    print("ELIGIBILITY GATE (pre-declared)")
    print("=" * 78)
    correctable = dist.get("EXPLICIT_CORRECTION", 0) + dist.get("STATE_UPDATE", 0)
    correctable_fail = sum(1 for x in labelled
                           if x["group"] == "FAILURE"
                           and x["relation"] in ("EXPLICIT_CORRECTION", "STATE_UPDATE"))
    print(f"  cases with a correction/update relation : {correctable}/{len(labelled)}")
    print(f"    of which FAILURES                     : {correctable_fail}")
    print()
    if correctable_fail == 0:
        print("  -> NO ELIGIBLE CASES.  An update/correction mechanism cannot be claimed")
        print("     to address the M11 failures, because no failure in this cohort has a")
        print("     correction or update relation to resolve.  Testing one here would be")
        print("     measuring a mechanism against a problem it cannot touch.")
        print("     Next: a different comparison axis, or verify on non-holdout data.")
    elif correctable_fail < 3:
        print(f"  -> ONLY {correctable_fail} eligible failure(s).  Too few to justify a")
        print("     mechanism test on this cohort.  Recorded as a hypothesis without a")
        print("     population here; look for a cohort where the relation actually occurs.")
    else:
        print(f"  -> {correctable_fail} eligible failures.  Enough to justify Stage 2:")
        print("     freeze a generalisable update/correction rule and compare against")
        print("     no-op with the frozen scorer, reporting wrong->correct,")
        print("     wrong->abstain, correct->wrong and abstain->wrong separately.")
        print("     Also required: confirm the update judgment survives REORDERING, which")
        print("     is what distinguishes a content mechanism from a position reaction.")
    print()
    print("  SCOPE: 30 stratum-selected cases.  These labels come from surface markers on")
    print("  the utterances, not from human adjudication -- a reviewer can and should")
    print("  disagree with any individual label by reading the stored utterance text.")
    print("  No claim is made about the Reader's reasoning.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 18 stage 1 -- update/correction eligibility audit (blind)",
        "baseline": {"head": "38b6ff8", "production_changed": False},
        "model_calls": 0,
        "relation_distribution": dict(dist),
        "eligible_failures": correctable_fail,
        "n_cases": len(labelled),
        "marker_sets_fixed_before_labelling": True,
        "newer_supersedes_older_assumed": False,
        "rows": labelled,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
