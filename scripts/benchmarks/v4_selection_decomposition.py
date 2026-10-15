"""V4 Mission 8 -- decompose WHY the 28 answer-selection failures picked a wrong value.

ANATOMY, NOT A FIX.  No production code changes, no correction engine, and no attempt to
build a classifier that explains all 28.  The goal is the BOUNDARY: under what condition
does the Reader pick the right value, and under what condition does it jump to a
different one.

The 28 cases are FROZEN (from v4_answer_selection_audit.json, cause
VERBOSE_INSTEAD_OF_SLOT).  For each one the ORIGINAL SOURCE DIALOGUE is inspected
directly, not just the gold and prediction strings.

THREE HYPOTHESES, TESTED INDEPENDENTLY -- a case may satisfy several, and that is
recorded rather than collapsed:

  H1  the correct value lives in a DIFFERENT turn than the one the Reader used
  H2  the SAME turn (or a nearby one) contains competing values of the same kind
  H3  the wrong answer corresponds to a real but DIFFERENT event/person/time

Each hypothesis is graded CONFIRMED / POSSIBLE / UNKNOWN.  UNKNOWN is a real result and
is never forced into the nearest bucket, because the mission's point is to find where
the evidence runs out.

Per case the table records: gold source turn, wrong-answer source turn/event, candidate
competition (same turn / other turn / not determinable), confusion kind, and the verdict.

GOLD IS USED ONLY TO LOCATE EVIDENCE FOR THIS AUDIT.  Nothing here alters a runtime
path, and no classification rule is fed back into answering or retrieval.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

AUDIT = ROOT / "benchmark_results/evidence_audit/v4_answer_selection_audit.json"
AUDITFULL = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
RAW = ROOT / "datasets/external/locomo10.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_selection_decomposition.json"

_STOP = frozenset({
    "the", "and", "for", "was", "were", "his", "her", "its", "their", "our", "your",
    "him", "she", "they", "them", "not", "but", "all", "any", "out", "about", "into",
    "over", "than", "then", "when", "what", "who", "why", "how", "which", "there",
    "here", "have", "has", "had", "been", "being", "are", "will", "would", "could",
    "should", "did", "does", "with", "that", "this", "from", "many", "much", "long",
    "often", "ago", "now", "get", "got", "one", "two", "did", "describe", "describes",
    "jon", "john", "melanie", "caroline", "nolan", "andrew", "audrey", "james",
    "joanna", "nate", "maria", "gina", "tim", "evan", "calvin", "dave", "sam",
    "jolene", "recently", "started", "learning",
})


def content_tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in _STOP}


def load_source() -> dict[str, list[dict]]:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for s in data:
        sid = s.get("sample_id")
        conv = s.get("conversation") or {}
        turns = []
        for k in sorted(conv.keys()):
            m = re.fullmatch(r"session_(\d+)", k)
            if not m:
                continue
            date = conv.get(f"session_{m.group(1)}_date_time", "")
            for t in conv[k] or []:
                turns.append({"dia_id": t.get("dia_id", ""),
                              "speaker": t.get("speaker", ""),
                              "text": t.get("text", ""),
                              "date": date})
        if sid:
            out[sid] = turns
    return out


def find_turns(haystack: list[dict], needle: str) -> list[dict]:
    """Source turns whose text contains the needle (case-insensitive, normalised)."""
    n = re.sub(r"\s+", " ", str(needle).strip().lower().strip("\"'.,!?"))
    if not n:
        return []
    hits = []
    for t in haystack:
        if n in re.sub(r"\s+", " ", t["text"].lower()):
            hits.append(t)
    return hits


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    full = json.loads(AUDITFULL.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in full["records"]}
    cohort = [x for x in audit["rows"] if x["cause"] == "VERBOSE_INSTEAD_OF_SLOT"]

    src = load_source()

    frozen_ctx = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            frozen_ctx[r["qid"]] = r["context"]

    print("=" * 74)
    print("V4 MISSION 8 -- answer-selection failure decomposition (ANATOMY, not a fix)")
    print("=" * 74)
    print(f"  frozen cohort : {len(cohort)} cases "
          f"({len({x['conversation'] for x in cohort})} conversations)")
    print("  hypotheses are tested INDEPENDENTLY; a case may satisfy several.")
    print("  Every case is graded against the ORIGINAL SOURCE dialogue, not just the")
    print("  gold/prediction strings.  UNKNOWN is kept as a real result.")
    print()

    results: list[dict] = []
    for x in cohort:
        qid = x["question_id"]
        conv = x["conversation"]
        gold = str(x["gold"])
        pred = str(x["prediction"])
        raw = str(x.get("raw_output") or pred)
        turns = src.get(conv, [])

        # Where does the gold live in the ORIGINAL conversation?
        gold_hits = find_turns(turns, gold)
        gold_turn_ids = [t["dia_id"] for t in gold_hits]
        gold_speakers = sorted({t["speaker"] for t in gold_hits})

        # Which turn did the Reader's answer come from?  Locate the prediction's most
        # distinctive content token in the source, so the wrong value is traced to a
        # real turn rather than assumed.
        pred_toks = content_tokens(raw) | content_tokens(pred)
        pred_turns = []
        for t in turns:
            low = t["text"].lower()
            hit = sum(1 for w in pred_toks if w in low)
            if hit >= max(2, len(pred_toks) // 2):
                pred_turns.append((hit, t))
        pred_turns.sort(key=lambda p: -p[0])
        pred_turn_ids = [t["dia_id"] for _, t in pred_turns[:3]]
        pred_speakers = sorted({t["speaker"] for _, t in pred_turns[:3]})

        # Is the gold turn inside the frozen context the Reader actually saw?
        cache_lines = [l for l in frozen_ctx.get(
            (meta.get(qid) or {}).get("cache_qid", ""), "").split("\n") if l.strip()]
        cache_ids = []
        for l in cache_lines:
            m = re.match(r"\[([^\]]+)\]", l)
            if m:
                cache_ids.append(m.group(1).split(" on ")[0].strip())
        gold_in_given_context = any(g in cache_ids for g in gold_turn_ids)

        # ---- H1: the right value is in a DIFFERENT turn than the one used -------
        if not gold_hits:
            h1, h1_why = "UNKNOWN", "gold string not located in the source dialogue"
        elif gold_in_given_context and gold_turn_ids and pred_turn_ids \
                and not (set(gold_turn_ids) & set(pred_turn_ids)):
            h1, h1_why = ("CONFIRMED",
                          f"gold in {gold_turn_ids[:3]} (given to the Reader) while the "
                          f"answer came from {pred_turn_ids[:3]}")
        elif gold_turn_ids and pred_turn_ids and \
                not (set(gold_turn_ids) & set(pred_turn_ids)):
            h1, h1_why = ("POSSIBLE",
                          f"gold in {gold_turn_ids[:3]} but that turn is NOT in the "
                          f"frozen context; answer from {pred_turn_ids[:3]}")
        elif set(gold_turn_ids) & set(pred_turn_ids):
            h1, h1_why = ("NOT_H1",
                          f"both gold and answer trace to the same turn "
                          f"{sorted(set(gold_turn_ids) & set(pred_turn_ids))[:3]}")
        else:
            h1, h1_why = "UNKNOWN", "could not locate either side in the source"

        # ---- H2: competing values of the same kind in the same/nearby turn ------
        # The first version matched r'\b[A-Z][a-z]{3,}\b' as a 'name', which also
        # catches any sentence-initial word -- 'Thanks', 'Nice', 'What', 'Last' -- so
        # H2 fired POSSIBLE on 26 of 28 and carried no information.  A proper noun must
        # be capitalised AND not sentence-initial AND not a known speaker name (a
        # speaker is attribution metadata, not a competing value).
        _SPEAKERS = {t["speaker"].lower() for t in turns}
        competitors = []
        for t in (gold_hits[:2] + [t for _, t in pred_turns[:2]]):
            names = set()
            for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", t["text"]):
                w = m.group(1).lower()
                if w in _SPEAKERS or w in _STOP:
                    continue
                names.add(w)
            dates = {d.lower() for d in re.findall(
                r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
                r"\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
                r"\s+\d{1,2}\b", t["text"], re.I)}
            counts = {c.lower() for c in re.findall(
                r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b|\b\d+\b",
                t["text"])}
            gold_c = content_tokens(gold)
            for kind, pool in (("name", names), ("date", dates), ("count", counts)):
                others = pool - gold_c
                if len(others) >= 2:
                    competitors.append({"turn": t["dia_id"], "kind": kind,
                                        "others": sorted(others)[:5]})
        if competitors:
            kinds = sorted({c["kind"] for c in competitors})
            h2, h2_why = ("POSSIBLE",
                          f"{len(competitors)} turn(s) carry other same-kind values "
                          f"({', '.join(kinds)}), e.g. "
                          f"{competitors[0]['others'][:3]}")
        else:
            h2, h2_why = ("NOT_H2", "no other same-kind value found near either side")

        # ---- H3: the wrong answer is a real but different event/person ----------
        confusion = []
        if gold_speakers and pred_speakers and \
                not (set(gold_speakers) & set(pred_speakers)):
            confusion.append("person")
        g_dates = {t["date"] for t in gold_hits}
        p_dates = {t["date"] for _, t in pred_turns[:3]}
        if g_dates and p_dates and not (g_dates & p_dates):
            confusion.append("time")
        gt, pt = content_tokens(gold), content_tokens(raw)
        if gt and pt and not (gt & pt):
            confusion.append("event")
        if gold_hits and pred_turns:
            h3 = ("CONFIRMED" if confusion else "NOT_H3")
            h3_why = (f"wrong answer maps to a different "
                      f"{'+'.join(confusion)}" if confusion else
                      "wrong answer maps to the same speaker/time as the gold")
        else:
            h3, h3_why = "UNKNOWN", "could not trace both sides to source turns"

        results.append({
            "question_id": qid,
            "conversation": conv,
            "question": x["question"],
            "gold": gold,
            "prediction": pred,
            "raw_output": raw[:200],
            "gold_source_turns": gold_turn_ids[:4],
            "gold_speakers": gold_speakers,
            "gold_in_given_context": gold_in_given_context,
            "pred_source_turns": pred_turn_ids,
            "pred_speakers": pred_speakers,
            "H1_different_turn": h1, "H1_why": h1_why,
            "H2_competing_values": h2, "H2_why": h2_why,
            "H3_confusion": h3, "H3_why": h3_why,
            "confusion_kinds": confusion,
            "competitors": competitors[:4],
        })

    # ---- aggregation, and hypotheses are reported SEPARATELY (they overlap) -----
    def tally(field: str) -> Counter:
        return Counter(r[field] for r in results)

    print("=" * 74)
    print("HYPOTHESIS TALLIES  (independent, overlapping -- NOT a partition)")
    print("=" * 74)
    for f, label in (("H1_different_turn", "H1 correct value in a DIFFERENT turn"),
                     ("H2_competing_values", "H2 competing values nearby"),
                     ("H3_confusion", "H3 wrong value = a different event/person/time")):
        counts_f = tally(f)
        print(f"  {label}")
        for k, v in counts_f.most_common():
            print(f"      {k:<10} {v}")
    print()

    print("CONFUSION KINDS (H3 detail, overlapping)")
    ck = Counter(k for r in results for k in r["confusion_kinds"])
    for k, v in ck.most_common():
        print(f"  {k:<10} {v}")
    print()

    print("WAS THE GOLD EVEN IN THE CONTEXT THE READER SAW?")
    print(f"  gold turn present in the frozen context : "
          f"{sum(1 for r in results if r['gold_in_given_context'])}/{len(results)}")
    print()

    print("PER CASE")
    print("-" * 74)
    for r in results:
        print(f"  {r['question_id']}  gold={r['gold'][:20]!r}")
        print(f"     Q    : {r['question'][:60]}")
        print(f"     gold turns {r['gold_source_turns']} spk={r['gold_speakers']} "
              f"in_given_ctx={r['gold_in_given_context']}")
        print(f"     pred turns {r['pred_source_turns']} spk={r['pred_speakers']}")
        print(f"     H1={r['H1_different_turn']}  H2={r['H2_competing_values']}  "
              f"H3={r['H3_confusion']}  confusion={r['confusion_kinds']}")
        print(f"     H1: {r['H1_why'][:76]}")
        if r["H2_competing_values"] != "NOT_H2":
            print(f"     H2: {r['H2_why'][:76]}")
        if r["H3_confusion"] != "NOT_H3":
            print(f"     H3: {r['H3_why'][:76]}")
    print()

    # ---- which hypothesis, if any, leads somewhere ---------------------------
    print("=" * 74)
    print("WHAT THIS POINTS AT (per the mission's pre-declared routing)")
    print("=" * 74)
    h1c = tally("H1_different_turn").get("CONFIRMED", 0)
    h1p = tally("H1_different_turn").get("POSSIBLE", 0)
    h2p = tally("H2_competing_values").get("POSSIBLE", 0)
    h3c = tally("H3_confusion").get("CONFIRMED", 0)
    h3u = tally("H3_confusion").get("UNKNOWN", 0)
    unk = (tally("H1_different_turn").get("UNKNOWN", 0)
           + tally("H3_confusion").get("UNKNOWN", 0))
    print(f"  H1 confirmed {h1c} / possible {h1p}")
    print(f"  H2 possible  {h2p}")
    print(f"  H3 confirmed {h3c} / unknown {h3u}")
    print(f"  cases with any UNKNOWN grade: {unk}")
    print()
    n = len(results)
    if h1c >= n * 0.4 and h1c > h3c:
        print("  -> H1 leads.  Investigate how the QUESTION is bound to its evidence")
        print("     turn, not where the evidence lives.")
    elif h3c >= n * 0.4 and h3c > h1c:
        print("  -> H3 leads.  Investigate event / person / time correspondence.")
    elif h2p >= n * 0.4:
        print("  -> H2 leads.  Investigate the relational information needed to")
        print("     discriminate between competing values of the same kind.")
    elif unk >= n * 0.3:
        print(f"  -> EVIDENCE-INSUFFICIENT to route further: {unk} cases cannot be")
        print("     graded from source inspection.  Stop here and report what is")
        print("     missing rather than building on an unstable classification.")
    else:
        print("  -> MIXED.  Per the mission, do NOT try to rescue all 28 with a single")
        print("     correction engine; the conditions differ across cases.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 8 -- answer-selection failure decomposition (anatomy only)",
        "n": len(results),
        "conversations": len({r["conversation"] for r in results}),
        "h1": dict(tally("H1_different_turn")),
        "h2": dict(tally("H2_competing_values")),
        "h3": dict(tally("H3_confusion")),
        "confusion_kinds": dict(ck),
        "gold_in_given_context": sum(1 for r in results
                                     if r["gold_in_given_context"]),
        "rows": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
