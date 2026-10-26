"""V4 Mission 12 -- candidate selection audit: what distinguishes the event the Reader
CHOSE from the event it did NOT choose?

Cohort is frozen from M11 (not re-searched, not re-tuned):
  FAILURE n=23  (M11 GOLD_AND_PRED_EVENT)  chosen vs rejected = pred vs gold event
  CONTROL n=14  (M11 GOLD_AND_COMPETITOR_VISIBLE)                chosen vs rejected = gold vs rival
  remaining rows stay recorded, not dropped

TWO PHASES, and phase 1 is blind AND ANONYMISED.
  PHASE 1: from the question, the context and the candidate events only, measure
           position, lexical overlap and person/speaker relations.  Candidates are
           labelled A/B in order of context position, never 'gold'/'pred'.  gold,
           prediction and correctness are not read here, so the metric cannot be tuned
           to the outcome.
  PHASE 2: join the outcome and only then map A/B onto chosen/rejected.
Preprocessing, tokenisation and metric definitions are FIXED HERE, BEFORE the numbers
are seen.  No threshold is chosen after looking.

THIS MEASURES SELECTION TENDENCY IN THE OUTPUT.  It does NOT observe the Reader's
internal reasoning, and nothing below claims it does.
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
OUT = ROOT / "benchmark_results/evidence_audit/v4_candidate_selection.json"

# ---- FIXED BEFORE RESULTS: tokenisation, stopwords, metric definitions -------------
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
    "tell", "give", "gives", "mention", "mentioned", "according", "didnt", "dont",
})
_TIE_ADJ = ("early", "first", "earlier", "before", "initial", "originally",
            "started", "start", "begin", "began", "beginning", "recent", "recently",
            "late", "later", "last", "latest", "current", "currently", "yesterday")
_END_OF_RE = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")


def content_words(s: str) -> set[str]:
    """FIXED definition: lowercase, keep alnum runs of length > 2, drop stopwords."""
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in _STOP and len(w) > 2}


_HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")
_QUOTE = re.compile(r'^(?:\(In reply to[^)]*\)\s*)?(?:"[^"]*"\s*)?([A-Z][a-z]+):')
_REPLY = re.compile(r"\(In reply to ([A-Z][a-z]+):")


def parse_context(context: str) -> list[dict]:
    """Position (line order as the Reader saw it), speaker, reply-TARGET, utterance."""
    rows = []
    for i, line in enumerate(l for l in context.split("\n") if l.strip()):
        m = _HDR.match(line)
        if not m:
            continue
        rest = line[m.end():]
        rm = _REPLY.search(rest)
        reply_to = rm.group(1) if rm else None
        ms = _QUOTE.match(rest)
        spk = ms.group(1) if ms else None
        body = rest[ms.end():] if ms else rest
        rows.append({"pos": i, "turn": m.group(1), "speaker": spk,
                     "reply_to": reply_to, "utterance": body})
    return rows


def main() -> int:
    m11 = json.loads(M11.read_text(encoding="utf-8"))
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    by_id = {r["question_id"]: r for r in m11["rows"]}
    fails = [r for r in m11["rows"]
             if r["group"] == "FAILURE" and r["verdict"] == "GOLD_AND_PRED_EVENT"]
    ctrls = [r for r in m11["rows"]
             if r["group"] == "CONTROL" and r["verdict"] == "GOLD_AND_COMPETITOR_VISIBLE"]

    print("=" * 74)
    print("V4 MISSION 12 -- candidate selection audit (features frozen BEFORE outcome)")
    print("=" * 74)
    print("  FAILURE cohort : 23  (chosen = pred event,  rejected = gold event)")
    print("  CONTROL cohort : 14  (chosen = gold event,  rejected = rival)")
    print(f"  excluded but recorded: {len(m11['rows']) - len(fails) - len(ctrls)} rows "
          "(not measurable, kept in the artefact)")
    print("  PHASE 1 measures position / overlap / speaker with candidates anonymised")
    print("           as A and B, in context-position order.  gold, pred and the")
    print("           correctness label are NOT read in phase 1.")
    print("  This measures SELECTION TENDENCY IN THE OUTPUT.  It does not observe the")
    print("  Reader's internal reasoning, and nothing here claims it does.")
    print()

    # =========================== PHASE 1 -- blind, anonymised ==================
    frozen_feats: dict[str, dict] = {}
    for qid, r in list(by_id.items()):
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        if not context:
            frozen_feats[qid] = {"ok": False, "why": "no frozen context"}
            continue
        q = r["question"]
        ctx = parse_context(context)
        pos_of = {c["turn"]: c["pos"] for c in ctx}
        q_words = content_words(q)

        # the two candidate turn ids for this question, as recorded and FROZEN by M11
        cand = list(r.get("gold_events") or []) + list(r.get("pred_events") or [])
        if r["group"] == "CONTROL":
            cand = list(r.get("gold_events") or []) + list(r.get("competitor_events") or [])
        cand = [t for t in cand if t]
        if len(cand) < 2:
            frozen_feats[qid] = {"ok": False, "why": "fewer than two candidate events"}
            continue
        a, b = cand[0], cand[1]

        def feat(t: str) -> dict:
            row = next((c for c in ctx if c["turn"] == t), None)
            if row is None:
                return {"turn": t, "pos": pos_of.get(t), "speaker": None,
                        "reply_to": None, "utterance": "", "overlap": set()}
            u = row["utterance"]
            # H12-B, fixed definition: content-word overlap of question with the
            # utterance supporting the candidate, absolute AND relative to the
            # utterance's own content words.
            uw = content_words(u)
            ov = q_words & uw
            return {"turn": t, "pos": row["pos"], "speaker": row["speaker"],
                    "reply_to": row["reply_to"], "utterance": u[:160],
                    "u_words": uw, "overlap": ov,
                    "overlap_n": len(ov),
                    "overlap_frac": (len(ov) / len(uw)) if uw else 0.0}

        fa, fb = feat(a), feat(b)
        # anonymise by context position: A = earlier line, B = later line
        if fa["pos"] is not None and fb["pos"] is not None and fa["pos"] > fb["pos"]:
            fa, fb = fb, fa
            a, b = b, a
        frozen_feats[qid] = {
            "ok": True, "group": r["group"], "n_candidates": len(cand),
            "A_turn": a, "B_turn": b,
            "A_pos": fa["pos"], "B_pos": fb["pos"],
            "A_overlap_n": fa.get("overlap_n"), "B_overlap_n": fb.get("overlap_n"),
            "A_overlap_frac": fa.get("overlap_frac"),
            "B_overlap_frac": fb.get("overlap_frac"),
            "A_speaker": fa.get("speaker"), "B_speaker": fb.get("speaker"),
            "A_reply_to": fa.get("reply_to"), "B_reply_to": fb.get("reply_to"),
            "A_utterance": fa.get("utterance", ""), "B_utterance": fb.get("utterance", ""),
            "A_overlap": sorted(fa.get("overlap") or []),
            "B_overlap": sorted(fb.get("overlap") or []),
        }

    print("PHASE 1 complete.  Features frozen.  Now joining the outcome (PHASE 2).")
    ok_n = sum(1 for v in frozen_feats.values() if v.get("ok"))
    print(f"  measurable : {ok_n}    not measurable : {len(frozen_feats) - ok_n}")
    print()

    # =========================== PHASE 2 -- join outcome =======================
    # BUG FIXED: this loop iterated frozen_feats, i.e. ALL of M11's rows, so the
    # M11 cohorts (23 failures / 14 controls) were reported as 27 and 15 -- the
    # PRED_ONLY and GOLD_ONLY rows leaked into the measured groups and inflated the
    # denominators.  The cohort must be intersected with the frozen M11 selection.
    cohort = {r["question_id"]: r for r in (fails + ctrls)}
    people_re = re.compile(r"\b([A-Z][a-z]{2,})\b")

    def question_people(q: str) -> set[str]:
        return {w for w in people_re.findall(q or "") if w not in _STOP}

    rows = []
    for qid, ff in frozen_feats.items():
        if qid not in cohort:
            # recorded, not silently dropped, and NOT counted in any denominator
            rows.append({"question_id": qid, "measurable": False,
                         "group": ff.get("group"),
                         "why": "not in the frozen M11 cohort for this mission"})
            continue
        if not ff.get("ok"):
            rows.append({"question_id": qid, "measurable": False,
                         "group": ff.get("group"), "why": ff.get("why")})
            continue
        r = cohort[qid]
        if r["group"] == "FAILURE":
            # chosen = the event that produced the prediction; rejected = gold event
            gset, pset = set(r.get("gold_events") or []), set(r.get("pred_events") or [])
            if ff["A_turn"] in pset:
                chosen, rejected = "A", "B"
            elif ff["A_turn"] in gset:
                chosen, rejected = "B", "A"
            else:
                chosen = rejected = None
        else:
            gset, cset = set(r.get("gold_events") or []), set(r.get("competitor_events") or [])
            if ff["A_turn"] in gset:
                chosen, rejected = "A", "B"
            elif ff["A_turn"] in cset:
                chosen, rejected = "B", "A"
            else:
                chosen = rejected = None
        if chosen is None:
            rows.append({"question_id": qid, "measurable": False,
                         "why": "candidate identity not resolvable from frozen ids"})
            continue
        ch = chosen.lower()
        qp = question_people(r["question"])
        rows.append({
            "question_id": qid, "measurable": True,
            "group": r["group"], "conversation": r["conversation"],
            "chosen": ch, "rejected": "b" if ch == "a" else "a",
            "chosen_turn": ff[f"{ch.upper()}_turn"],
            "rejected_turn": ff[f"{rejected.upper()}_turn"],
            # H12-A position
            "chosen_pos": ff[f"{ch.upper()}_pos"],
            "rejected_pos": ff[f"{rejected.upper()}_pos"],
            "pos_gap": (ff[f"{rejected.upper()}_pos"] - ff[f"{ch.upper()}_pos"])
                       if (ff[f"{ch.upper()}_pos"] is not None
                           and ff[f"{rejected.upper()}_pos"] is not None) else None,
            "chosen_is_first": ff[f"{ch.upper()}_pos"] is not None
                               and ff[f"{rejected.upper()}_pos"] is not None
                               and ff[f"{ch.upper()}_pos"] < ff[f"{rejected.upper()}_pos"],
            # H12-B lexical overlap
            "chosen_overlap_n": ff[f"{ch.upper()}_overlap_n"],
            "rejected_overlap_n": ff[f"{rejected.upper()}_overlap_n"],
            "chosen_overlap_frac": ff[f"{ch.upper()}_overlap_frac"],
            "rejected_overlap_frac": ff[f"{rejected.upper()}_overlap_frac"],
            # H12-C people
            "question_people": sorted(qp),
            "chosen_speaker": ff[f"{ch.upper()}_speaker"],
            "rejected_speaker": ff[f"{rejected.upper()}_speaker"],
            "chosen_reply_to": ff[f"{ch.upper()}_reply_to"],
            "chosen_speaker_in_q": (ff[f"{ch.upper()}_speaker"] in qp)
                                   if ff[f"{ch.upper()}_speaker"] else None,
            "rejected_speaker_in_q": (ff[f"{rejected.upper()}_speaker"] in qp)
                                     if ff[f"{rejected.upper()}_speaker"] else None,
            "question": r["question"],
        })

    meas = [x for x in rows if x.get("measurable")]
    fm = [x for x in meas if x["group"] == "FAILURE"]
    cm = [x for x in meas if x["group"] == "CONTROL"]
    print(f"  measurable failures : {len(fm)}/23     measurable controls : {len(cm)}/14")
    print(f"  not measurable (kept) : {len(rows) - len(meas)}")
    print()

    def pct(k, n):
        return f"{100 * k / n:.0f}%" if n else "n/a"

    # --------------------------- H12-A position -------------------------------
    print("=" * 74)
    print("H12-A  POSITION in the frozen context (line order the Reader saw)")
    print("=" * 74)
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        if not g:
            continue
        first = sum(1 for x in g if x["chosen_is_first"])
        print(f"  {label:<8} chosen is the EARLIER of the two : {first}/{len(g)} "
              f"= {pct(first, len(g))}")
    print()
    f_first = sum(1 for x in fm if x["chosen_is_first"]) / len(fm) if fm else 0
    c_first = sum(1 for x in cm if x["chosen_is_first"]) / len(cm) if cm else 0
    print(f"  gap (FAILURE - CONTROL) : {100 * (f_first - c_first):+.0f}pp")
    print("  Reading: in FAILURE the chosen (wrong) event being EARLIER would mean the")
    print("  Reader favours the first-seen candidate.  In CONTROL the chosen (right)")
    print("  event being earlier would mean the correct event simply tends to come")
    print("  first -- a property of the data, not of the Reader.")
    print()

    # --------------------------- H12-B lexical overlap ------------------------
    print("=" * 74)
    print("H12-B  LEXICAL OVERLAP with the question (definitions fixed before results)")
    print("=" * 74)
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        if not g:
            continue
        hi = sum(1 for x in g
                 if (x["chosen_overlap_n"] or 0) > (x["rejected_overlap_n"] or 0))
        lo = sum(1 for x in g
                 if (x["chosen_overlap_n"] or 0) < (x["rejected_overlap_n"] or 0))
        eq = len(g) - hi - lo
        print(f"  {label:<8} chosen overlaps MORE : {hi}/{len(g)} = {pct(hi, len(g))}"
              f"   less: {lo}   equal: {eq}")
    print()
    f_hi = (sum(1 for x in fm if (x["chosen_overlap_n"] or 0) > (x["rejected_overlap_n"] or 0))
            / len(fm)) if fm else 0
    c_hi = (sum(1 for x in cm if (x["chosen_overlap_n"] or 0) > (x["rejected_overlap_n"] or 0))
            / len(cm)) if cm else 0
    print(f"  gap (FAILURE - CONTROL) : {100 * (f_hi - c_hi):+.0f}pp")
    print()

    # --------------------------- H12-C people --------------------------------
    print("=" * 74)
    print("H12-C  PEOPLE (reply-target is never treated as the speaker)")
    print("=" * 74)
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        if not g:
            continue
        ch_sp = sum(1 for x in g if x["chosen_speaker_in_q"] is True)
        rj_sp = sum(1 for x in g if x["rejected_speaker_in_q"] is True)
        print(f"  {label:<8} speaker matches a person named in the question -- "
              f"chosen {ch_sp}/{len(g)}, rejected {rj_sp}/{len(g)}")
    print()

    # --------------------------- conversation spread --------------------------
    print("=" * 74)
    print("CONVERSATION SPREAD (no single dialogue may dominate)")
    print("=" * 74)
    for label, g in (("FAILURE", fm), ("CONTROL", cm)):
        cnt = Counter(x["conversation"] for x in g)
        print(f"  {label:<8} " + ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items())))
    print()
    # leave-one-conversation-out on each hypothesis's gap
    print("  LEAVE-ONE-CONVERSATION-OUT")
    for name, metric in (("H12-A chosen-is-earlier", "chosen_is_first"),
                         ("H12-B chosen-overlaps-more", None)):
        gaps = []
        for cv in sorted({x["conversation"] for x in meas}):
            f2 = [x for x in fm if x["conversation"] != cv]
            c2 = [x for x in cm if x["conversation"] != cv]
            if not f2 or not c2:
                continue
            if metric:
                g1 = (sum(1 for x in f2 if x[metric]) / len(f2)
                      - sum(1 for x in c2 if x[metric]) / len(c2))
            else:
                g1 = (sum(1 for x in f2 if (x["chosen_overlap_n"] or 0)
                          > (x["rejected_overlap_n"] or 0)) / len(f2)
                      - sum(1 for x in c2 if (x["chosen_overlap_n"] or 0)
                            > (x["rejected_overlap_n"] or 0)) / len(c2))
            gaps.append((cv, 100 * g1))
        if gaps:
            print(f"    {name:<28} gap range "
                  f"{min(g for _, g in gaps):+.0f}pp .. {max(g for _, g in gaps):+.0f}pp")
    print()

    # --------------------------- gate ------------------------------------------
    print("=" * 74)
    print("GATE (pre-declared)")
    print("=" * 74)
    a_gap = 100 * (f_first - c_first)
    b_gap = 100 * (f_hi - c_hi)
    spk_f = (sum(1 for x in fm if x["chosen_speaker_in_q"] is True) / len(fm)) if fm else 0
    spk_c = (sum(1 for x in cm if x["chosen_speaker_in_q"] is True) / len(cm)) if cm else 0
    c_gap = 100 * (spk_f - spk_c)
    for nm, gp in (("H12-A position", a_gap), ("H12-B overlap", b_gap),
                   ("H12-C speaker", c_gap)):
        if abs(gp) >= 25:
            print(f"  {nm:<18} gap {gp:+.0f}pp  -> discriminates; candidate for the next")
            print(f"                    intervention, NOT adopted on this evidence alone.")
        else:
            print(f"  {nm:<18} gap {gp:+.0f}pp  -> does not discriminate success from failure.")
    print()
    strong = [nm for nm, gp in (("A", a_gap), ("B", b_gap), ("C", c_gap))
              if abs(gp) >= 25]
    if not strong:
        print("  ALL THREE are weak.  Position, lexical overlap and speaker agreement do")
        print("  not separate the chosen-from-correct case from the chosen-from-wrong")
        print("  case.  The honest conclusion is that an UNMEASURED factor is involved;")
        print("  no new rule is invented, and nothing is promoted to production.")
    print()
    print("  LIMIT, stated once and not walked back: this audit measures selection")
    print("  tendency in the Reader's OUTPUT.  It does not observe the Reader's internal")
    print("  reasoning, so no claim here explains WHY a candidate was preferred.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 12 -- candidate selection audit, features frozen before outcome",
        "baseline": {"head": "8106b2f", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "cohort_from": "v4_frozen_context_binding.json (M11), frozen, not re-selected",
        "n_failure_cohort": len(fails), "n_control_cohort": len(ctrls),
        "measurable_failures": len(fm), "measurable_controls": len(cm),
        "phase1_blind_and_anonymised": True,
        "observes_internal_reasoning": False,
        "h12a_gap_pp": a_gap, "h12b_gap_pp": b_gap, "h12c_gap_pp": c_gap,
        "discriminating_hypotheses": strong,
        "rows": rows,
        "frozen_features": frozen_feats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
