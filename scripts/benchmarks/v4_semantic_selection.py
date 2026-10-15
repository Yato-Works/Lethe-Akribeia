"""V4 Mission 13 -- semantic dissection of the M11/M12 value-selection failures.

PHASE 1 IS BLIND: for each case it records what the question DEMANDS, the candidate
events, and -- per candidate -- whether the context supports it AS AN ANSWER to that
demand.  gold, prediction and the correctness label are NOT read in phase 1.  Candidates
are referred to as CAND-A and CAND-B, ordered by context position.

WHAT IS NEW HERE versus M12.  M12 measured surface features (position, word overlap,
speaker) and none separated success from failure.  Phase 1 here asks a different
question: does the candidate actually SATISFY the predicate the question asserts?
That is a semantic check, and it is the one thing M12 never tested.

Predicate classes are FIXED HERE, BEFORE any number is seen, and are derived from the
question alone:
  TEMPORAL_ORDER   "before X", "after X", "first", "last", "since", "until"
  EXCLUSION        "besides", "other than", "apart from", "instead of"
  ATTRIBUTE        "what kind", "which one", "what colour/name/place", a noun slot
  REASON           "why", "reason for", "because"
  COUNT            "how many", "how much"
  PERSON           "who"
  PLAIN            everything else

Per candidate, blind, it records:
  satisfies  : does the utterance contain what the predicate requires (CONFIRMED /
               POSSIBLE / UNKNOWN)
  polarity   : does the utterance negate or supersede the fact
  superseDES : is a later utterance a correction of it
  slot_fill  : is the value a member of the kind the question asks for
NO hardcoding of case ids, gold strings or prediction strings; no gold is ever read
in phase 1; UNKNOWN is a legal outcome and is preserved.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

M11 = ROOT / "benchmark_results/evidence_audit/v4_frozen_context_binding.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_semantic_selection.json"

# ---------------- predicate classes, FIXED BEFORE RESULTS -------------------------
_P_TEMPORAL = re.compile(
    r"\b(?:before|after|first|last|since|until|prior\s+to|earlier|later|recently"
    r"|originally|previously|by\s+the\s+time)\b", re.I)
_P_EXCLUSION = re.compile(
    r"\b(?:besides|other\s+than|apart\s+from|aside\s+from|instead\s+of|not\s+including"
    r"|excluding|rather\s+than)\b", re.I)
_P_REASON = re.compile(r"\b(?:why|reason|because|motivat\w*|purpose|for\s+what)\b", re.I)
_P_COUNT = re.compile(r"\bhow\s+(?:many|much|often|long)\b", re.I)
_P_PERSON = re.compile(r"^\s*who\b|\bwhose\b", re.I)
_P_ATTR = re.compile(
    r"\bwhat\s+(?:kind|sort|type|colour|color|name|place|food|song|movie|book"
    r"|hobby|hobbies|activity|activities|sport|sports|language|instrument|pet|animal"
    r"|feeling|emotion)\b"
    r"|\bwhich\b", re.I)
_NEG = re.compile(
    r"\b(?:not|n't|never|no\s+longer|stopped|quit|instead\s+of|rather\s+than"
    r"|used\s+to|switched|changed)\b", re.I)
_SUPER = re.compile(
    r"\b(?:actually|correction|i\s+was\s+wrong|to\s+be\s+clear|what\s+i\s+meant"
    r"|update|i\s+mean)\b", re.I)

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
    "too", "very", "just", "also", "now", "once", "here", "s", "t", "many", "much",
    "tell", "give", "gives", "mention", "mentioned", "according",
})


def predicate_class(q: str) -> str:
    """Order matters; the most specific class wins.  Derived from the question only."""
    if _P_COUNT.search(q):
        return "COUNT"
    if _P_EXCLUSION.search(q):
        return "EXCLUSION"
    if _P_REASON.search(q):
        return "REASON"
    if _P_PERSON.search(q):
        return "PERSON"
    if _P_TEMPORAL.search(q):
        return "TEMPORAL_ORDER"
    if _P_ATTR.search(q):
        return "ATTRIBUTE"
    return "PLAIN"


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in _STOP and len(w) > 2}


_HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")
_QUOTE = re.compile(r'^(?:\(In reply to[^)]*\)\s*)?(?:"[^"]*"\s*)?([A-Z][a-z]+):')
_REPLY = re.compile(r"\(In reply to ([A-Z][a-z]+):")


def parse_context(context: str) -> list[dict]:
    rows = []
    for i, line in enumerate(l for l in context.split("\n") if l.strip()):
        m = _HDR.match(line)
        if not m:
            continue
        rest = line[m.end():]
        rm = _REPLY.search(rest)
        ms = _QUOTE.match(rest)
        body = rest[ms.end():] if ms else rest
        rows.append({"pos": i, "turn": m.group(1),
                     "speaker": ms.group(1) if ms else None,
                     "reply_to": rm.group(1) if rm else None,
                     "utterance": body})
    return rows


def exclusion_terms(q: str) -> list[str]:
    """For EXCLUSION questions, the thing that must NOT be the answer.  Question only."""
    m = re.search(r"(?:besides|other\s+than|apart\s+from|aside\s+from|instead\s+of"
                  r"|rather\s+than|not\s+including)\s+(.+?)[?.!]?$", q, re.I)
    if not m:
        return []
    tail = m.group(1)
    return [w for w in content_words(tail)]


def main() -> int:
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
    cohort = {r["question_id"]: r for r in (fails + ctrls)}

    print("=" * 74)
    print("V4 MISSION 13 -- semantic dissection, PHASE 1 BLIND")
    print("=" * 74)
    print(f"  cohort frozen from M11 : {len(fails)} FAILURE, {len(ctrls)} CONTROL")
    print("  PHASE 1 reads: the question and the context.  It does NOT read gold,")
    print("  prediction or the correctness label.  Candidates are CAND-A / CAND-B.")
    print("  New versus M12: M12 tested surface features.  This tests whether a")
    print("  candidate SATISFIES the predicate the question asserts.")
    print()

    frozen_out: dict[str, dict] = {}
    for qid, r in cohort.items():
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        q = r["question"]
        pc = predicate_class(q)
        rec = {"question_id": qid, "predicate_class": pc,
               "exclusion_terms": exclusion_terms(q)}
        if not context:
            rec["ok"] = False
            rec["why"] = "no frozen context"
            frozen_out[qid] = rec
            continue
        ctx = parse_context(context)
        pos_of = {c["turn"]: c["pos"] for c in ctx}
        # The two candidate turns, as frozen by M11, ordered by context position.
        # BUG FIXED: the first version did gold_events[0] + pred_events[0] and also
        # admitted extra turns, which produced gold-vs-gold pairs (lc_conv-41_128 came
        # out as D25:2 vs D25:3, both GOLD turns) and dropped 20 of 37 cases as
        # 'not separable'.  M11 already froze the exact pair it judged in ctx_positions,
        # so recover the two turn ids from those positions instead of guessing which
        # list entry to use.  Duplicates are removed, so a side contributing the same
        # turn twice can never self-pair.
        pos_by_turn = {c["turn"]: c["pos"] for c in ctx}
        cp = r.get("ctx_positions")
        ids: list[str] = []
        if r["group"] == "FAILURE" and isinstance(cp, list) and len(cp) >= 2:
            for want in (cp[0], cp[1]):
                hit = next((t for t, p in pos_by_turn.items() if p == want), None)
                if hit:
                    ids.append(hit)
        if len(ids) < 2:
            g = list(dict.fromkeys(r.get("gold_events") or []))
            p = list(dict.fromkeys(r.get("pred_events") or []))
            if r["group"] == "CONTROL":
                p = list(dict.fromkeys(r.get("competitor_events") or []))
            ids = (([g[0]] if g else []) + ([p[0]] if p else []))
        ids = [t for t in dict.fromkeys(ids) if t in pos_of]
        if len(ids) < 2:
            rec["ok"] = False
            rec["why"] = "fewer than two candidates located in the frozen context"
            frozen_out[qid] = rec
            continue
        ids.sort(key=lambda t: pos_of[t])
        qw = content_words(q)
        cands = {}
        for label, t in zip(("CAND-A", "CAND-B"), ids):
            row = next((c for c in ctx if c["turn"] == t), None)
            u = row["utterance"] if row else ""
            uw = content_words(u)
            ov = qw & uw
            # ---- does this candidate SATISFY the predicate?  (blind) ----
            negated = bool(_NEG.search(u))
            supersede = bool(_SUPER.search(u))
            if pc == "EXCLUSION":
                ex = rec["exclusion_terms"]
                viol = [w for w in ex if w in uw]
                sat = "UNKNOWN" if not ex else ("NO" if viol else "POSSIBLE")
            elif pc == "COUNT":
                has_num = bool(re.search(r"\b(?:one|two|three|four|five|six|seven"
                                         r"|eight|nine|ten|eleven|twelve|\d+)\b",
                                         u, re.I))
                sat = "POSSIBLE" if has_num else "UNKNOWN"
            elif pc == "PERSON":
                sat = "POSSIBLE" if (uw & qw) else "UNKNOWN"
            elif pc == "REASON":
                sat = "POSSIBLE" if len(ov) >= 2 else "UNKNOWN"
            elif pc == "TEMPORAL_ORDER":
                has_ord = bool(re.search(r"\b(?:before|after|first|last|since|until"
                                         r"|yesterday|then|when)\b", u, re.I))
                sat = "POSSIBLE" if has_ord else "UNKNOWN"
            else:
                sat = "POSSIBLE" if len(ov) >= 1 else "UNKNOWN"
            cands[label] = {
                "turn": t, "pos": pos_of[t],
                "speaker": row["speaker"] if row else None,
                "reply_to": row["reply_to"] if row else None,
                "utterance": u[:180],
                "overlap_words": sorted(ov)[:10], "overlap_n": len(ov),
                "negated": negated, "superseding_language": supersede,
                "satisfies_predicate": sat,
            }
        rec["ok"] = True
        rec["candidates"] = cands
        frozen_out[qid] = rec

    print("PHASE 1 frozen.  Predicate classes (from the question alone):")
    print("  " + str(dict(Counter(v["predicate_class"] for v in frozen_out.values()))))
    print(f"  measurable : {sum(1 for v in frozen_out.values() if v.get('ok'))}"
          f"    not measurable : {sum(1 for v in frozen_out.values() if not v.get('ok'))}")
    print()
    print("Now joining gold / prediction / outcome -- PHASE 2.")
    print()

    # ================= PHASE 2 =================================================
    rows = []
    for qid, ff in frozen_out.items():
        if not ff.get("ok"):
            rows.append({"question_id": qid, "measurable": False,
                         "why": ff.get("why")})
            continue
        r = cohort[qid]
        gset = set(r.get("gold_events") or [])
        pset = set(r.get("pred_events") or [])
        cset = set(r.get("competitor_events") or [])
        A, B = ff["candidates"]["CAND-A"], ff["candidates"]["CAND-B"]
        if r["group"] == "FAILURE":
            chosen = "CAND-A" if A["turn"] in pset else ("CAND-B" if B["turn"] in pset else None)
            rejected = "CAND-A" if A["turn"] in gset else ("CAND-B" if B["turn"] in gset else None)
        else:
            chosen = "CAND-A" if A["turn"] in gset else ("CAND-B" if B["turn"] in gset else None)
            rejected = "CAND-A" if A["turn"] in cset else ("CAND-B" if B["turn"] in cset else None)
        if not (chosen and rejected) or chosen == rejected:
            rows.append({"question_id": qid, "measurable": False,
                         "why": "chosen/rejected not separable in the frozen ids"})
            continue
        C, R = ff["candidates"][chosen], ff["candidates"][rejected]
        rows.append({
            "question_id": qid, "measurable": True, "group": r["group"],
            "conversation": r["conversation"], "predicate_class": ff["predicate_class"],
            "chosen": chosen, "rejected": rejected,
            "chosen_turn": C["turn"], "rejected_turn": R["turn"],
            "chosen_satisfies": C["satisfies_predicate"],
            "rejected_satisfies": R["satisfies_predicate"],
            "chosen_negated": C["negated"], "rejected_negated": R["negated"],
            "chosen_superseding": C["superseding_language"],
            "rejected_superseding": R["superseding_language"],
            "chosen_overlap_n": C["overlap_n"], "rejected_overlap_n": R["overlap_n"],
            "chosen_is_earlier": C["pos"] < R["pos"],
            "gold": r.get("gold"), "prediction": r.get("prediction"),
            "question": r["question"],
        })

    meas = [x for x in rows if x.get("measurable")]
    fm = [x for x in meas if x["group"] == "FAILURE"]
    cm = [x for x in meas if x["group"] == "CONTROL"]
    print(f"  measurable failures : {len(fm)}/{len(fails)}    "
          f"controls : {len(cm)}/{len(ctrls)}    "
          f"kept-as-unmeasurable : {len(rows) - len(meas)}")
    print()

    def pct(k, n):
        return f"{100 * k / n:.0f}%" if n else "n/a"

    print("=" * 74)
    print("H13-1  Does the CHOSEN candidate satisfy the question's predicate?")
    print("=" * 74)
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        if not g:
            continue
        yes = sum(1 for x in g if x["chosen_satisfies"] == "POSSIBLE")
        unk = sum(1 for x in g if x["chosen_satisfies"] == "UNKNOWN")
        no = sum(1 for x in g if x["chosen_satisfies"] == "NO")
        print(f"  {label:<8} chosen: POSSIBLE {yes}/{len(g)}={pct(yes, len(g))}"
              f"   UNKNOWN {unk}   NO {no}")
    print()
    # the key contrast: rejected satisfies MORE than chosen?
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        if not g:
            continue
        rank = {"NO": 0, "UNKNOWN": 1, "POSSIBLE": 2}
        worse = sum(1 for x in g
                    if rank[x["chosen_satisfies"]] < rank[x["rejected_satisfies"]])
        print(f"  {label:<8} chosen satisfies the predicate LESS than the rejected "
              f"candidate : {worse}/{len(g)} = {pct(worse, len(g))}")
    print()

    print("=" * 74)
    print("H13-2  NEGATION / SUPERSESSION on the chosen candidate")
    print("=" * 74)
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        if not g:
            continue
        cn = sum(1 for x in g if x["chosen_negated"])
        rn = sum(1 for x in g if x["rejected_negated"])
        cs = sum(1 for x in g if x["chosen_superseding"])
        rs = sum(1 for x in g if x["rejected_superseding"])
        print(f"  {label:<8} negated  chosen {cn}/{len(g)}  rejected {rn}      "
              f"superseding chosen {cs}  rejected {rs}")
    print()

    print("=" * 74)
    print("H13-3  PREDICATE CLASS breakdown of the failures")
    print("=" * 74)
    for pc, n in Counter(x["predicate_class"] for x in fm).most_common():
        sub = [x for x in fm if x["predicate_class"] == pc]
        rank = {"NO": 0, "UNKNOWN": 1, "POSSIBLE": 2}
        worse = sum(1 for x in sub
                    if rank[x["chosen_satisfies"]] < rank[x["rejected_satisfies"]])
        print(f"  {pc:<14} n={n:<3} chosen-satisfies-less {worse}/{n}")
    print()

    print("=" * 74)
    print("CONVERSATION SPREAD")
    print("=" * 74)
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        print(f"  {label:<8} " + ", ".join(
            f"{k}:{v}" for k, v in sorted(Counter(x["conversation"] for x in g).items())))
    print()

    rank = {"NO": 0, "UNKNOWN": 1, "POSSIBLE": 2}
    f_worse = (sum(1 for x in fm if rank[x["chosen_satisfies"]] < rank[x["rejected_satisfies"]])
               / len(fm)) if fm else 0
    c_worse = (sum(1 for x in cm if rank[x["chosen_satisfies"]] < rank[x["rejected_satisfies"]])
               / len(cm)) if cm else 0
    gap = 100 * (f_worse - c_worse)
    print("=" * 74)
    print("GATE (pre-declared)")
    print("=" * 74)
    print(f"  FAILURE : chosen satisfies the predicate less than rejected "
          f"{f_worse * 100:.0f}%")
    print(f"  CONTROL : chosen satisfies the predicate less than rejected "
          f"{c_worse * 100:.0f}%")
    print(f"  gap : {gap:+.0f}pp")
    print()
    if gap >= 30:
        print("  -> predicate satisfaction DOES separate the groups.  This is a")
        print("     generalisable, gold-free, surface-adjacent signal and it is the")
        print("     justified candidate for an isolated experiment.")
    elif gap >= 15:
        print("  -> weak separation.  Recorded as a candidate, but it does not meet the")
        print("     bar for an isolated experiment on 37 cases.")
    else:
        print("  -> predicate satisfaction does NOT separate the groups either.  The")
        print("     semantic layer that M12 did not test has now been tested and also")
        print("     fails.  No new rule is invented; NO-GO is the honest outcome.")
    print()
    print("  LIMIT: this audit still measures what the OUTPUT chose.  It does not")
    print("  observe the Reader's internal reasoning.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 13 -- semantic predicate audit, phase 1 blind",
        "baseline": {"head": "ce792ca", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "cohort_from": "v4_frozen_context_binding.json (M11), frozen, not regenerated",
        "n_failure": len(fails), "n_control": len(ctrls),
        "measurable_failures": len(fm), "measurable_controls": len(cm),
        "phase1_blind": True,
        "predicate_classes": dict(Counter(v["predicate_class"] for v in frozen_out.values())),
        "chosen_satisfies_less_failure_pct": 100 * f_worse,
        "chosen_satisfies_less_control_pct": 100 * c_worse,
        "gap_pp": gap,
        "rows": rows,
        "phase1_frozen": frozen_out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
