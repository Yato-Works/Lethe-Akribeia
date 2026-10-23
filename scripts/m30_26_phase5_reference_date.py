#!/usr/bin/env python3
"""M30-26 Phase 5: Reference-date reconciliation (isolated diagnostic).

For each of the 28 Bucket A cases:
  1. SESSION_DATES[conv]  (what the scorer was given)
  2. the session datetime of the LoCoMo annotated evidence turn (what the turn says)
  3. whether they agree
  4. for relative-date turns: does resolving against the EVIDENCE turn's own
     session datetime reach Gold, where the scorer's reference_date did not?

Read-only. No production code, Gold, benchmark data or scorer is modified.
553 holdout untouched.
"""
import json, re
from pathlib import Path

LOCOMO = Path(r"C:\Users\smily\AppData\Local\Temp\locomo10.json")
BUCKET = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
M25 = Path("benchmark_results/evidence_audit/v4_intervention_temporal.json")
M30_3 = Path("results/m30_3_evaluation_results.json")

SESSION_DATES = {
    "conv-26": "2023-08-28", "conv-30": "2023-11-30",
    "conv-41": "2023-12-04", "conv-42": "2023-12-04",
    "conv-44": "2023-12-04", "conv-47": "2022-09-30",
    "conv-48": "2023-01-31", "conv-49": "2023-12-25",
    "conv-50": "2023-08-28",
}
MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}


def parse_session_datetime(s):
    """'1:32 pm on 6 January, 2024' -> (2024, 1, 6); '2023-08-28' -> (2023, 8, 28)"""
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s)
    if m:
        mon = MONTHS.get(m.group(2).lower())
        if mon:
            return (int(m.group(3)), mon, int(m.group(1)))
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def resolve(expr, base):
    """Resolve one relative expression against a base (Y,M,D). Returns (Y,M,D|None)."""
    y, mo, d = base
    if "next month" in expr:
        return (y + (1 if mo == 12 else 0), 1 if mo == 12 else mo + 1, None)
    if "last month" in expr:
        return (y - (1 if mo == 1 else 0), 12 if mo == 1 else mo - 1, None)
    if "this month" in expr:
        return (y, mo, None)
    if "next year" in expr:
        return (y + 1, None, None)
    if "last year" in expr or "one year ago" in expr or re.search(r"\ba year ago\b", expr):
        return (y - 1, None, None)
    m = re.search(r"(\d+)\s+years?\s+ago", expr)
    if m:
        return (y - int(m.group(1)), None, None)
    if "yesterday" in expr:
        return (y, mo, d - 1) if d > 1 else ((y, mo - 1, 28) if mo > 1 else (y - 1, 12, 31))
    if "tomorrow" in expr:
        return (y, mo, d + 1) if d < 28 else ((y, mo + 1, 1) if mo < 12 else (y + 1, 1, 1))
    return None


def gold_date(gold):
    g = str(gold).lower().replace(".", " ").replace(",", " ")
    m = re.search(r"\b(\d{1,2})\s+([a-z]+)\s+(\d{4})", g)
    if m and m.group(2) in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)), "day")
    m = re.search(r"\b([a-z]+)\s+(\d{1,2})\s+(\d{4})", g)
    if m and m.group(1) in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)), "day")
    m = re.search(r"\b([a-z]+)\s+(\d{4})", g)
    if m and m.group(1) in MONTHS:
        return (int(m.group(2)), MONTHS[m.group(1)], None, "month")
    m = re.search(r"\b((?:19|20)\d{2})\b", g)
    if m:
        return (int(m.group(1)), None, None, "year")
    return None


def reaches(resolved, gold):
    if not resolved or not gold:
        return False
    ry, rmo, rd = resolved
    gy, gmo, gd, gran = gold
    if ry != gy:
        return False
    if gran == "year":
        return True
    if gmo is not None and rmo is not None and rmo != gmo:
        return False
    if gran == "day" and gd is not None and rd is not None and rd != gd:
        return False
    return True


def main():
    locomo = json.loads(LOCOMO.read_text(encoding="utf-8"))
    bucket = json.loads(BUCKET.read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucket["cases"]}
    rows = json.loads(M25.read_text(encoding="utf-8"))["rows"]
    targets = [r["question_id"] for r in rows if r.get("group") == "TARGET"]
    m30_3 = {r["question_id"]: r for r in json.loads(M30_3.read_text(encoding="utf-8"))["results"]}

    agree, disagree, no_evidence = [], [], []
    detail = []

    for qid in targets:
        c = cases[qid]
        conv_id = c["source_conversation"]
        question = c["question"]
        gold = str(c["gold_answer"])
        scorer_ref = SESSION_DATES.get(conv_id)

        conv = next((x for x in locomo if x.get("sample_id") == conv_id), None)
        qa = None
        if conv:
            for cand in conv.get("qa", []):
                if cand.get("question", "").lower().strip() == question.lower().strip():
                    qa = cand
                    break
        ev_ids = qa.get("evidence", []) if qa else []
        conv_data = conv.get("conversation", {}) if conv else {}

        ev_turns = []
        for eid in ev_ids:
            m = re.match(r"D(\d+):(\d+)", eid)
            if not m:
                continue
            skey = f"session_{m.group(1)}"
            sdt = conv_data.get(f"{skey}_date_time", "")
            turns = conv_data.get(skey, [])
            idx = int(m.group(2)) - 1
            if isinstance(turns, list) and 0 <= idx < len(turns) and isinstance(turns[idx], dict):
                ev_turns.append({"id": eid, "session_datetime": sdt, "text": turns[idx].get("text", "")})

        ev_dates = {parse_session_datetime(t["session_datetime"]) for t in ev_turns}
        scorer_date = parse_session_datetime(scorer_ref) if scorer_ref else None

        record = {
            "question_id": qid,
            "gold": gold,
            "scorer_reference_date": scorer_ref,
            "evidence_ids": ev_ids,
            "evidence_session_datetimes": [t["session_datetime"] for t in ev_turns],
            "reader_output": m30_3.get(qid, {}).get("reader_text", ""),
            "normal_correct": m30_3.get(qid, {}).get("scorer_correct", False),
        }

        if not ev_turns:
            no_evidence.append(qid)
            record["status"] = "NO_EVIDENCE_TURN"
        else:
            # does the scorer's ref agree with at least one evidence turn's date?
            if scorer_date in ev_dates:
                agree.append(qid)
                record["status"] = "AGREE"
            else:
                disagree.append(qid)
                record["status"] = "DISAGREE"

            # For relative expressions: can we reach Gold using the evidence turn's
            # own session datetime, where the scorer's reference date failed?
            reader = m30_3.get(qid, {}).get("reader_text", "")
            rel = reader.lower()
            has_rel = any(k in rel for k in [
                "next month", "last month", "this month", "next year",
                "last year", "one year ago", "years ago", "yesterday", "tomorrow"])
            if has_rel and not record["normal_correct"]:
                gd = gold_date(gold)
                for t in ev_turns:
                    base = parse_session_datetime(t["session_datetime"])
                    if not base:
                        continue
                    res = resolve(rel, base)
                    if res and reaches(res, gd):
                        record["recoverable_with_evidence_date"] = {
                            "evidence_id": t["id"],
                            "evidence_session_datetime": t["session_datetime"],
                            "resolved": res,
                        }
                        break

        detail.append(record)

    print("=" * 78)
    print("M30-26 Phase 5: reference-date reconciliation")
    print("=" * 78)
    print(f"targets                     : {len(targets)}")
    print(f"scorer ref == evidence date : {len(agree)}")
    print(f"scorer ref != evidence date : {len(disagree)}")
    print(f"no evidence turn            : {len(no_evidence)}")
    print()
    print("DISAGREE:")
    for qid in disagree:
        r = next(x for x in detail if x["question_id"] == qid)
        print(f"  {qid}: scorer={r['scorer_reference_date']}  evidence={r['evidence_session_datetimes']}")
    print()
    recovered = [r for r in detail if "recoverable_with_evidence_date" in r]
    print(f"relative-answer cases recoverable with the EVIDENCE turn's own date: {len(recovered)}")
    for r in recovered:
        rec = r["recoverable_with_evidence_date"]
        print(f"  {r['question_id']}: reader='{r['reader_output']}' gold='{r['gold']}'")
        print(f"      via {rec['evidence_id']} @ {rec['evidence_session_datetime']} -> {rec['resolved']}")

    out = Path("results/m30_26/phase5_reference_date_reconciliation.json")
    out.write_text(json.dumps({
        "agree": agree, "disagree": disagree, "no_evidence": no_evidence,
        "recoverable_with_evidence_date": [r["question_id"] for r in recovered],
        "detail": detail,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
