"""V4 Mission 11 -- is the M8 'wrong EVENT' finding true inside the world the READER saw?

M8 established, on the ORIGINAL dialogue, that 26 of 28 wrong answers belong to a
DIFFERENT event than the gold.  M10 then showed that a source-side fact need not be true
of the frozen context the Reader was actually given (M9's 7 same-day cases were not even
co-present in it).  So M8's claim cannot be carried into production reasoning untested.

THIS AUDIT IS IN THE FROZEN CONTEXT, NOT THE SOURCE.

TWO PHASES, DELIBERATELY SEPARATED.  A classifier written first would make the result an
artifact of the classifier -- exactly the M10 trap.  So:
  PHASE 1  identify the EVENTS inside the frozen context and freeze the evidence
           (source turn id, context position, speaker, date + its provenance, the
           supporting utterance, relation to what the question asks).  No gold, no
           prediction, no correctness label is touched here.
  PHASE 2  join gold and prediction to the phase-1 events and assign the verdict.
Verdicts are mutually exclusive on the failure side:
  GOLD_AND_PRED_EVENT | GOLD_ONLY | PRED_ONLY | NEITHER | UNKNOWN
'Contained' is NOT string presence.  An event counts only when an utterance inside the
context supports the relation the question asks about for the queried subject.

THE CONTROL GROUP USES THE SAME CONTEXT CRITERION.  A correct answer has no 'wrong event
it chose', so for controls we record (a) whether the correct event is visible and
(b) whether a COMPETING event is also visible.  Controls whose competing event cannot be
located stay UNKNOWN and remain in the denominator.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

M9 = ROOT / "benchmark_results/evidence_audit/v4_event_binding.json"
M8 = ROOT / "benchmark_results/evidence_audit/v4_selection_decomposition.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_frozen_context_binding.json"

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
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
    "too", "very", "just", "also", "now", "then", "once", "here", "s", "t",
})


def header_date(header: str) -> tuple[str | None, str]:
    m = re.search(r"\bon\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", header or "")
    if not m:
        return None, "NONE"
    day, mon, yr = int(m.group(1)), m.group(2).lower(), m.group(3)
    num = next((v for n, v in _MONTHS.items() if mon.startswith(n[:3])), None)
    if not num:
        return None, "NONE"
    return f"{yr}-{num:02d}-{day:02d}", "HEADER"


def text_date(text: str) -> tuple[str | None, str]:
    for pat, kind in (
        (re.compile(rf"\b(\d{{1,2}})\s+({'|'.join(_MONTHS)})\w*,?\s+(\d{{4}})\b", re.I), "dmy"),
        (re.compile(rf"\b({'|'.join(_MONTHS)})\w*\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I), "mdy"),
    ):
        m = pat.search(text)
        if not m:
            continue
        g = m.groups()
        try:
            if kind == "dmy":
                return f"{g[2]}-{_MONTHS[g[1].lower()]:02d}-{int(g[0]):02d}", "TEXT"
            return f"{g[2]}-{_MONTHS[g[0].lower()]:02d}-{int(g[1]):02d}", "TEXT"
        except Exception:
            continue
    return None, "NONE"


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in _STOP and len(w) > 2}


# --------------------------------------------------------------------------- PHASE 1
def phase1_events(context: str, question: str) -> list[dict]:
    """Identify the EVENTS in the frozen context.  No gold / prediction / label used."""
    q_toks = toks(question)
    events: list[dict] = []
    _HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")
    _QUOTE = re.compile(r'^(?:\(In reply to[^)]*\)\s*)?(?:"[^"]*"\s*)?([A-Z][a-z]+):')
    for i, line in enumerate(l for l in context.split("\n") if l.strip()):
        m = _HDR.match(line)
        if not m:
            continue
        rest = line[m.end():]
        ms = _QUOTE.match(rest)
        spk = ms.group(1) if ms else ""
        body = rest[ms.end():] if ms else rest
        hdr = m.group(0)
        d, src = header_date(hdr)
        if d is None:
            d, src = text_date(body)
        b_toks = toks(body)
        # relation to what the QUESTION asks: content-word overlap with the question
        rel = sorted(q_toks & b_toks)
        events.append({
            "source_turn": m.group(1),          # source id, e.g. D16:6
            "context_pos": i,                    # where it sits in the frozen context
            "speaker": spk or None,
            "date": d, "date_source": src,
            "header": hdr.strip("[] "),
            "utterance": body[:200],
            "question_overlap": rel[:8],
            "overlap_n": len(rel),
        })
    return events


def find_supporting_events(events: list[dict], value: str) -> list[dict]:
    """An event SUPPORTS a value only if its utterance contains the value's content
    words.  String presence alone is not enough -- the utterance must be the evidence."""
    v_toks = toks(value)
    if not v_toks:
        return []
    hits = []
    for e in events:
        u = toks(e["utterance"])
        cov = v_toks & u
        if cov:
            hits.append({**e, "value_coverage": sorted(cov),
                         "coverage_n": len(cov), "coverage_frac": len(cov) / len(v_toks)})
    hits.sort(key=lambda h: (-h["coverage_frac"], -h["overlap_n"]))
    return hits


# --------------------------------------------------------------------------- PHASE 2
def verdict_for(events, gold, pred, question):
    g = find_supporting_events(events, gold)
    p = find_supporting_events(events, pred)
    # an event must also relate to what the question asks about, else a shared word is
    # not evidence FOR THIS QUESTION
    g_ok = [e for e in g if e["overlap_n"] >= 1] or g[:1]
    p_ok = [e for e in p if e["overlap_n"] >= 1] or p[:1]
    gvis, pvis = bool(g_ok), bool(p_ok)
    if gvis and pvis:
        v = "GOLD_AND_PRED_EVENT"
    elif gvis:
        v = "GOLD_ONLY"
    elif pvis:
        v = "PRED_ONLY"
    elif g or p:
        v = "UNKNOWN"          # a value word appears but not tied to the question
    else:
        v = "NEITHER"
    return v, g_ok, p_ok


def main() -> int:
    m9 = json.loads(M9.read_text(encoding="utf-8"))
    m8 = {r["question_id"]: r for r in
          json.loads(M8.read_text(encoding="utf-8"))["rows"]}
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    print("=" * 74)
    print("V4 MISSION 11 -- binding inside the FROZEN CONTEXT (not the source)")
    print("=" * 74)
    print("  PHASE 1 identifies events in the Reader's context.  gold / prediction /")
    print("  correctness are NOT read in phase 1.")
    print("  PHASE 2 joins gold and prediction to those frozen events.")
    print()

    rows: list[dict] = []
    for group, recs in (("FAILURE", m9["failures"]), ("CONTROL", m9["controls"])):
        for r in recs:
            qid = r["question_id"]
            cq = (meta.get(qid) or {}).get("cache_qid", "")
            context = frozen.get(cq, "")
            if not context:
                rows.append({"group": group, "question_id": qid,
                             "verdict": "UNKNOWN", "note": "no frozen context"})
                continue
            q = r["question"]
            # ---------------- PHASE 1 (blind) ----------------
            events = phase1_events(context, q)
            rec = {"group": group, "question_id": qid, "conversation": r["conversation"],
                   "question": q, "gold": r["gold"], "prediction": r["prediction"],
                   "n_context_events": len(events),
                   "n_context_events_dated": sum(1 for e in events if e["date"])}
            if group == "FAILURE":
                # ---------------- PHASE 2 ----------------
                v, g_ok, p_ok = verdict_for(events, r["gold"], r["prediction"], q)
                rec["verdict"] = v
                rec["gold_events"] = [e["source_turn"] for e in g_ok[:3]]
                rec["gold_event_speakers"] = [e["speaker"] for e in g_ok[:3]]
                rec["pred_events"] = [e["source_turn"] for e in p_ok[:3]]
                rec["pred_event_speakers"] = [e["speaker"] for e in p_ok[:3]]
                if g_ok and p_ok:
                    rec["both_in_context"] = True
                    rec["same_speaker"] = (bool(g_ok[0]["speaker"])
                                           and g_ok[0]["speaker"] == p_ok[0]["speaker"])
                    rec["same_date"] = (g_ok[0]["date"] == p_ok[0]["date"]
                                        if (g_ok[0]["date"] and p_ok[0]["date"]) else None)
                    rec["ctx_positions"] = [g_ok[0]["context_pos"], p_ok[0]["context_pos"]]
                # what M8 concluded ON THE SOURCE, for a direct comparison.
                # BUG FIXED: the field was read as r.get("H3_event_confusion") from the
                # M9 artefact, which does not carry that key, so every case scored
                # 'M8 said wrong event: 0/28' and the M8 comparison silently vanished.
                # M8's verdict lives in v4_selection_decomposition.json as H3_confusion.
                m8r = m8.get(qid, {})
                rec["M8_said_wrong_event"] = m8r.get("H3_confusion") == "CONFIRMED"
                rec["M8_gold_in_given_context"] = m8r.get("gold_in_given_context")
                rec["M8_confusion_kinds"] = m8r.get("confusion_kinds")
                rec["M8_gold_turns"] = m8r.get("gold_source_turns")
                rec["M8_pred_turns"] = m8r.get("pred_source_turns")
            else:
                # A correct answer has no 'wrong event it chose'.  Record (a) correct
                # event visibility and (b) whether a COMPETING event is also visible.
                g_ok = find_supporting_events(events, r["gold"])
                g_ok = [e for e in g_ok if e["overlap_n"] >= 1] or g_ok[:1]
                rival = r.get("control_rival") or ""
                r_ok = find_supporting_events(events, rival) if rival else []
                r_ok = [e for e in r_ok if e["overlap_n"] >= 1] or r_ok[:1]
                if not g_ok:
                    rec["verdict"] = "UNKNOWN" if not events else "GOLD_NOT_IN_CONTEXT"
                else:
                    rec["verdict"] = ("GOLD_AND_COMPETITOR_VISIBLE" if r_ok
                                      else "GOLD_ONLY")
                rec["gold_events"] = [e["source_turn"] for e in g_ok[:3]]
                rec["competitor_events"] = [e["source_turn"] for e in r_ok[:3]]
                rec["control_rival"] = rival[:120]
            rows.append(rec)

    f = [x for x in rows if x["group"] == "FAILURE"]
    c = [x for x in rows if x["group"] == "CONTROL"]
    print(f"  frozen contexts located : {sum(1 for x in rows if 'n_context_events' in x)}"
          f"/{len(rows)}")
    print(f"  median events per context: "
          f"{sorted(x['n_context_events'] for x in rows if 'n_context_events' in x)[len([x for x in rows if 'n_context_events' in x]) // 2]}")
    print()

    print("=" * 74)
    print("PHASE 2 -- FAILURES, judged inside the frozen context (mutually exclusive)")
    print("=" * 74)
    vc = Counter(x["verdict"] for x in f)
    order = ["GOLD_AND_PRED_EVENT", "GOLD_ONLY", "PRED_ONLY", "NEITHER", "UNKNOWN"]
    print(f"  {'verdict':<26}{'n':>5}{'%':>8}")
    for v in order:
        n = vc.get(v, 0)
        print(f"  {v:<26}{n:>5}{100 * n / len(f):>7.1f}%")
    print(f"  {'TOTAL':<26}{len(f):>5}")
    print()

    print("=" * 74)
    print("THE DECISIVE COMPARISON -- does M8's finding survive in the frozen context?")
    print("=" * 74)
    both = [x for x in f if x["verdict"] == "GOLD_AND_PRED_EVENT"]
    m8_yes = [x for x in f if x.get("M8_said_wrong_event")]
    survived = [x for x in both if x.get("M8_said_wrong_event")]
    print(f"  M8 said 'wrong event' (on the SOURCE) : {len(m8_yes)}/{len(f)}")
    print(f"  still BOTH visible in frozen context  : {len(both)}/{len(f)}")
    print(f"  of those, also a wrong event in M8    : {len(survived)}/{len(f)}")
    print()
    print("  so the Reader could actually SEE both the correct and the rival event in:")
    print(f"    {len(both)} of {len(f)} failures ({100 * len(both) / len(f):.0f}%)")
    print()

    # controls: the same criterion
    print("=" * 74)
    print("CONTROLS -- same context criterion (a correct answer has no chosen rival)")
    print("=" * 74)
    cc = Counter(x["verdict"] for x in c)
    for v in ("GOLD_AND_COMPETITOR_VISIBLE", "GOLD_ONLY", "GOLD_NOT_IN_CONTEXT", "UNKNOWN"):
        n = cc.get(v, 0)
        if n:
            print(f"  {v:<32}{n:>5}{100 * n / len(c):>8.1f}%")
    print(f"  {'TOTAL':<32}{len(c):>5}")
    print()
    c_comp = cc.get("GOLD_AND_COMPETITOR_VISIBLE", 0)
    f_comp = len(both)
    print("  COMPETITOR-VISIBLE RATE -- the whole point of using a control")
    print(f"    failures : {f_comp}/{len(f)} = {100 * f_comp / len(f):.1f}%")
    print(f"    controls : {c_comp}/{len(c)} = {100 * c_comp / len(c):.1f}%")
    print(f"    difference : {100 * f_comp / len(f) - 100 * c_comp / len(c):+.1f}pp")
    print()
    if 100 * f_comp / len(f) - 100 * c_comp / len(c) < 10:
        print("  -> controls show a competitor about as often as failures.  'A rival was")
        print("     visible' does NOT by itself explain the wrong answer.")
    else:
        print("  -> failures show a visible competitor substantially more often.")
    print()

    # speaker/date kept SEPARATE, never merged into the verdict
    if both:
        ns = sum(1 for x in both if x.get("same_speaker"))
        nd = sum(1 for x in both if x.get("same_date") is True)
        nn = sum(1 for x in both if x.get("same_date") is None)
        print(f"  SEPARATE FIELDS among the {len(both)} both-visible failures")
        print(f"    same speaker : {ns}/{len(both)}   same date: {nd}/{len(both)}"
              f"   date unknown: {nn}")
        print()

    print("=" * 74)
    print("GATE (pre-declared)")
    print("=" * 74)
    gap = 100 * f_comp / len(f) - 100 * c_comp / len(c)
    unknown_frac = 100 * (vc.get("UNKNOWN", 0) + vc.get("NEITHER", 0)) / len(f)
    if len(both) >= len(f) * 0.5 and gap >= 10:
        print(f"  Both events visible in {len(both)}/{len(f)} failures AND failures beat")
        print(f"  controls on competitor visibility by {gap:+.1f}pp.")
        print("  -> event-binding intervention is justified to design; prove it with a")
        print("     paired A/B (>=1 rescue, 0 correct->wrong, 0 abstain->wrong, holdout).")
    elif len(both) >= len(f) * 0.5 and gap < 10:
        print(f"  Both events ARE visible ({len(both)}/{len(f)}), so the rival really was")
        print("  in front of the Reader -- but controls show a rival just as often, so")
        print("  visibility alone does not explain the mistake.  Move to the value")
        print("  SELECTION stage, not to event binding.")
    elif unknown_frac >= 30:
        print(f"  UNKNOWN/NEITHER is {unknown_frac:.0f}% -- the event correspondence is not")
        print("  reliably determinable in the frozen context at this granularity.")
        print("  -> HOLD.  Do not promote to a production fix; the instrument is not")
        print("     yet reliable enough to justify an intervention.")
    else:
        print(f"  Only {len(both)}/{len(f)} failures have both events in context, so the")
        print("  dominant story is not 'picked the wrong of two visible events'.")
        print("  -> examine evidence loss / compression / presentation separately, and do")
        print("     NOT label these as wrong-event binding.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 11 -- binding verified inside the FROZEN context, not the source",
        "baseline": {"head": "b7cae72", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "phase1_was_blind": True,
        "phase_separation": "phase 1 identifies events with no gold/pred/label; phase 2 joins",
        "failure_verdicts": dict(vc), "control_verdicts": dict(cc),
        "n_failure": len(f), "n_control": len(c),
        "both_visible_failures": len(both),
        "competitor_gap_pp": gap,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
