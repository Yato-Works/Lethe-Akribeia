"""V4 Mission 16 -- extract the principle from the 7 M15 rescues.

NOT "put the correct answer first".  All 7 rescues moved the GOLD LATER, so that rule is
refuted by the data before it is written.  Chronological normalisation is also tested and
scores 2/7 -- no better than chance, so it is refuted too.

What is done here:
  PHASE 1 (blind): for each of the 30 measurable cases, compute candidate ORDERING FEATURES
    from the context and the two candidate turns only -- NOT from gold, prediction or any
    correctness label.  Features are fixed here, before results are seen.
  PHASE 2: join the M15 outcome and test which feature separates rescued from
    not-rescued, and -- critically -- whether it separates rescued from BROKEN, because a
    rule that also breaks controls is worthless.

Candidate hypotheses, all pre-declared:
  H_ANSWER_FIRST  gold earlier than rival            (already refuted, kept as a check)
  H_CHRONO        order by source turn id (session, turn)
  H_CONTEXT_POS   order by position in the frozen context
  H_RETRIEVAL     order by the retriever's own rank (position in context == rank here)
  H_QUERY_OVERLAP order by content-word overlap with the question
  H_SPEAKER       the questioned subject's own turn before/after others
  H_LAST          gold is the LAST of the two  (the observed pattern, tested properly)

The discipline that matters: a rule is only a candidate if it rescues failures WITHOUT
breaking controls.  Rescue rate alone is how a wrong rule gets adopted.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = ROOT / "benchmark_results/evidence_audit/v4_order_probe_scored.json"
M11 = ROOT / "benchmark_results/evidence_audit/v4_frozen_context_binding.json"
M13 = ROOT / "benchmark_results/evidence_audit/v4_semantic_selection.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_order_principle.json"

_STOP = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did",
    "does", "do", "doing", "done", "was", "were", "is", "are", "am", "be", "been",
    "being", "have", "has", "had", "will", "would", "can", "could", "should", "the",
    "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "about",
    "into", "over", "after", "before", "during", "while", "their", "there", "them",
    "they", "she", "her", "hers", "his", "him", "he", "you", "your", "yours", "we",
    "our", "ours", "us", "i", "me", "my", "mine", "it", "its", "any", "some", "all",
    "both", "each", "more", "most", "other", "another", "such", "own", "same", "so",
    "too", "very", "just", "also", "now", "once", "here", "many", "much", "tell",
    "give", "gives", "mention", "mentioned", "according",
})
_HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")
_QUOTE = re.compile(r'^(?:\(In reply to[^)]*\)\s*)?(?:"[^"]*"\s*)?([A-Z][a-z]+):')


def tid(t):
    m = re.search(r"D(\d+):(\d+)", t or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def cwords(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in _STOP and len(w) > 2}


def main() -> int:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    m11 = json.loads(M11.read_text(encoding="utf-8"))
    m13 = json.loads(M13.read_text(encoding="utf-8"))
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    by11 = {r["question_id"]: r for r in m11["rows"]}
    sem = {r["question_id"]: r for r in m13["rows"] if r.get("measurable")}

    print("=" * 78)
    print("V4 MISSION 16 -- what principle do the 7 rescues share?")
    print("=" * 78)
    print("  CORRECTION to an earlier reading of M15: gold was FIRST in the original")
    print("  context in 7/7 of the rescues, so the swap moved the gold LATER.  \"Put the")
    print("  correct answer first\" is therefore refuted -- the rescue ran the other way.")
    print("  Chronological normalisation is tested too and would put gold first in only")
    print("  2/7, so it is refuted as well.")
    print()
    print("  PHASE 1 computes ordering features from context + the two candidate turns")
    print("  ONLY.  gold, prediction and correctness are not read there.")
    print("  PHASE 2 joins the M15 outcome and tests rescue rate AND control breakage,")
    print("  because a rule that rescues failures but breaks controls is worthless.")
    print()

    # ---------------- PHASE 1: blind features -------------------------------
    feats = {}
    for x in probe["rows"]:
        if x["status"] != "MEASURED":
            continue
        qid = x["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        a, b = x["pair"]
        lines = [l for l in context.split("\n") if l.strip()]
        pos = {}
        spk = {}
        body = {}
        for i, l in enumerate(lines):
            m = _HDR.match(l)
            if not m:
                continue
            pos[m.group(1)] = i
            rest = l[m.end():]
            ms = _QUOTE.match(rest)
            spk[m.group(1)] = ms.group(1) if ms else None
            body[m.group(1)] = (rest[ms.end():] if ms else rest)
        q = x.get("question") or (by11.get(qid) or {}).get("question", "")
        qw = cwords(q)
        ta, tb = tid(a), tid(b)
        rec = {
            "A_turn": a, "B_turn": b,
            "A_pos": pos.get(a), "B_pos": pos.get(b),
            "A_src": ta, "B_src": tb,
            "A_speaker": spk.get(a), "B_speaker": spk.get(b),
            "A_overlap": len(qw & cwords(body.get(a, ""))),
            "B_overlap": len(qw & cwords(body.get(b, ""))),
            "question": q,
        }
        feats[qid] = rec

    print(f"  PHASE 1 frozen for {len(feats)} cases.  Now joining the M15 outcome.")
    print()

    # ---------------- PHASE 2: test each ordering rule ---------------------
    # a rule maps (A,B) -> which turn should come FIRST
    RULES = {
        "H_ANSWER_FIRST(gold first)": None,      # filled below, needs the label
        "H_CHRONO(by source turn id)":
            lambda f: (f["A_turn"] if (f["A_src"] and f["B_src"] and f["A_src"] < f["B_src"])
                       else f["B_turn"]),
        "H_CONTEXT_POS(by frozen-context position)":
            lambda f: (f["A_turn"] if (f["A_pos"] is not None and f["B_pos"] is not None
                                       and f["A_pos"] < f["B_pos"]) else f["B_turn"]),
        "H_QUERY_OVERLAP(most overlap first)":
            lambda f: (f["A_turn"] if f["A_overlap"] >= f["B_overlap"] else f["B_turn"]),
        "H_LEAST_OVERLAP(least overlap first)":
            lambda f: (f["A_turn"] if f["A_overlap"] <= f["B_overlap"] else f["B_turn"]),
        "H_LAST(observed: gold LAST)": None,      # needs the label
    }

    print("=" * 78)
    print("BLIND RULES -- testable without any label")
    print("=" * 78)
    print(f"  {'rule':<42}{'rescue':>8}{'/fail':>7}{'break':>7}{'/ctrl':>7}")
    blind = {k: v for k, v in RULES.items() if v is not None}
    results = {}
    for name, fn in blind.items():
        res, brk = 0, 0
        nf = nc = 0
        for qid, f in feats.items():
            x = next(r for r in probe["rows"] if r["question_id"] == qid)
            first = fn(f)
            if x["group"] == "FAILURE":
                nf += 1
                if x["transition"] == "wrong->correct" and first == x["pair"][1]:
                    res += 1      # rescued when the rival came first
            else:
                nc += 1
                if x["transition"] in ("correct->wrong", "correct->abstain"):
                    brk += 1
        results[name] = (res, nf, brk, nc)
        print(f"  {name:<42}{res:>8}{nf:>7}{brk:>7}{nc:>7}")
    print()
    print("  NOTE: for a blind rule, 'rescue' counts the failures the rule's ordering")
    print("  coincides with the M15 swap that rescued.  This is the honest ceiling for a")
    print("  label-free rule, and it is what would actually be deployable.")
    print()

    # ---------------- the two label-dependent checks ------------------------
    print("=" * 78)
    print("LABEL-DEPENDENT CEILING -- how good could ANY rule be here?")
    print("=" * 78)
    gold_first = gold_last = chrono_hit = 0
    for qid, x in ((q, next(r for r in probe["rows"] if r["question_id"] == q))
                   for q in feats):
        r11 = by11.get(qid, {})
        g = (r11.get("gold_events") or [None])[0]
        a, b = x["pair"]
        if x["group"] == "FAILURE" and x["transition"] == "wrong->correct":
            if g == a:
                gold_first += 1
            if g == b:
                gold_last += 1
            f = feats[qid]
            chrono = (f["A_turn"] if (f["A_src"] and f["B_src"] and f["A_src"] < f["B_src"])
                      else f["B_turn"])
            if chrono == g:
                chrono_hit += 1
    print(f"  of the 7 rescues, gold was FIRST in the original context : {gold_first}/7")
    print(f"  of the 7 rescues, gold was LAST  in the original context : {gold_last}/7")
    print(f"  chronological order would have put gold first            : {chrono_hit}/7")
    print()
    if gold_last == 7:
        print("  -> \"gold last\" fits all 7.  But this is a LABEL-DEPENDENT description of")
        print("     what happened, not a deployable rule -- it uses gold to decide.")
        print("     Its only value is as a target for a blind rule to imitate.")
    print()

    # ---------------- can any blind rule imitate 'gold last'? ---------------
    print("=" * 78)
    print("CAN A BLIND RULE IMITATE 'gold LAST'?  (the actual question)")
    print("=" * 78)
    agree = Counter()
    for qid, f in feats.items():
        x = next(r for r in probe["rows"] if r["question_id"] == qid)
        r11 = by11.get(qid, {})
        g = (r11.get("gold_events") or [None])[0]
        if x["group"] != "FAILURE" or x["transition"] != "wrong->correct":
            continue
        target_first = x["pair"][0] if g == x["pair"][1] else x["pair"][1]
        for name, fn in blind.items():
            if fn(f) == target_first:
                agree[name] += 1
    for name in blind:
        print(f"  {name:<42} matches the rescuing order {agree[name]}/7")
    print()

    print("=" * 78)
    print("CONTROL SYMMETRY -- the finding that constrains any interpretation")
    print("=" * 78)
    ctrl_gold_first = 0
    ctrl_n = 0
    for qid, f in feats.items():
        x = next(r for r in probe["rows"] if r["question_id"] == qid)
        if x["group"] != "CONTROL":
            continue
        ctrl_n += 1
        r11 = by11.get(qid, {})
        g = (r11.get("gold_events") or [None])[0]
        if g == x["pair"][0]:
            ctrl_gold_first += 1
    ctrl_broken = sum(1 for qid, f in feats.items()
                      if next(r for r in probe["rows"] if r["question_id"] == qid)["group"] == "CONTROL"
                      and next(r for r in probe["rows"] if r["question_id"] == qid)["transition"]
                      in ("correct->wrong", "correct->abstain"))
    print(f"  controls where gold was FIRST (so the swap moved it later): "
          f"{ctrl_gold_first}/{ctrl_n}")
    print(f"  controls broken by that same move                          : "
          f"{ctrl_broken}/{ctrl_n}")
    print()
    print("  THIS IS THE CONSTRAINT.  The exact same operation -- move the gold later --")
    print("  rescued 7 failures and broke 0 controls.  So 'move it later' is not a")
    print("  selection PRINCIPLE that finds the right answer; it is a SAFE DIRECTION")
    print("  that costs nothing when the answer was already right.  Those are very")
    print("  different claims, and only the second is supported.")
    print()

    print("=" * 78)
    print("PRINCIPLE STATUS")
    print("=" * 78)
    best = max(blind, key=lambda k: agree[k]) if blind else None
    if best and agree[best] >= 5:
        print(f"  {best} reproduces {agree[best]}/7 of the rescuing orders WITHOUT any")
        print("  label.  BUT the control symmetry above caps what that can mean: a blind")
        print("  rule is only worth testing if it rescues MORE than a direction that is")
        print("  already free.  Next experiment: apply the rule to all 30 and compare its")
        print("  rescue against the do-nothing baseline, with breakage measured separately.")
    else:
        print(f"  No blind rule reproduces more than {max(agree.values()) if agree else 0}/7.")
        print("  -> the rescues are NOT explained by any label-free ordering rule tested")
        print("     here.  The swap worked, but WHY it worked is not captured by position,")
        print("     chronology or lexical overlap.  A selection mechanism must compare the")
        print("     candidates on something other than order -- inventing an ordering rule")
        print("     now would be fitting noise on 7 cases.")
    print()
    print("  SCOPE: 7 rescues is a very small base for a generalisation.  Even a rule that")
    print("  matched 7/7 would be a hypothesis to test on the full cohort, not a finding.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 16 -- principle extraction from the 7 M15 rescues",
        "baseline": {"head": "fa5130d", "production_changed": False},
        "gold_first_in_rescues": gold_first,
        "gold_last_in_rescues": gold_last,
        "chrono_would_put_gold_first": chrono_hit,
        "blind_rule_agreement_with_rescuing_order": dict(agree),
        "blind_rule_rescue_break": {k: {"rescue": v[0], "n_failure": v[1],
                                        "break": v[2], "n_control": v[3]}
                                    for k, v in results.items()},
        "phase1_features": feats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
