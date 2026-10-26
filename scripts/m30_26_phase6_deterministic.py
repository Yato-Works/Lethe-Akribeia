#!/usr/bin/env python3
"""M30-26 Phase 6: Deterministic temporal resolution — isolated diagnostic.

Baseline  : resolve the Reader's relative expression against SESSION_DATES[conv]
            (what the production scorer actually used).
Treatment : resolve the SAME Reader output against the session datetime of the
            LoCoMo annotated evidence turn (oracle; diagnostic only).

No LLM call. No production code / Gold / scorer / benchmark change.
553 holdout untouched.  Treatment numbers are NOT production performance.
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


def parse(s):
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def resolve(expr, base):
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
    return None


def gold_date(g):
    g = str(g).lower().replace(".", " ").replace(",", " ")
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


def reaches(res, gold):
    if not res or not gold:
        return False
    ry, rmo, rd = res
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
    targets = [r["question_id"] for r in json.loads(M25.read_text(encoding="utf-8"))["rows"]
               if r.get("group") == "TARGET"]
    m30_3 = {r["question_id"]: r for r in json.loads(M30_3.read_text(encoding="utf-8"))["results"]}

    rows = []
    for qid in targets:
        c = cases[qid]
        conv_id = c["source_conversation"]
        gold = str(c["gold_answer"])
        reader = m30_3.get(qid, {}).get("reader_text", "")
        rel = reader.lower()
        is_relative = any(k in rel for k in ["next month", "last month", "this month",
                                             "next year", "last year", "year ago"])
        if not is_relative:
            continue

        conv = next((x for x in locomo if x.get("sample_id") == conv_id), None)
        qa = next((x for x in conv.get("qa", [])
                   if x.get("question", "").lower().strip() == c["question"].lower().strip()), None) if conv else None
        ev_base = None
        if qa:
            for eid in qa.get("evidence", []):
                m = re.match(r"D(\d+):(\d+)", eid)
                if not m:
                    continue
                skey = f"session_{m.group(1)}"
                sdt = conv["conversation"].get(f"{skey}_date_time", "")
                b = parse(sdt)
                if b:
                    ev_base = (b, eid, sdt)
                    break

        scorer_base = parse(SESSION_DATES.get(conv_id, ""))
        gd = gold_date(gold)

        base_res = resolve(rel, scorer_base) if scorer_base else None
        trt_res = resolve(rel, ev_base[0]) if ev_base else None

        rows.append({
            "question_id": qid,
            "reader_output": reader,
            "gold": gold,
            "scorer_reference_date": SESSION_DATES.get(conv_id),
            "evidence_turn": ev_base[1] if ev_base else None,
            "evidence_session_datetime": ev_base[2] if ev_base else None,
            "baseline_resolved": base_res,
            "baseline_reaches_gold": reaches(base_res, gd),
            "treatment_resolved": trt_res,
            "treatment_reaches_gold": reaches(trt_res, gd),
        })

    print("=" * 78)
    print("M30-26 Phase 6: deterministic temporal resolution (diagnostic)")
    print("=" * 78)
    print(f"relative-expression cases: {len(rows)}")
    b = sum(1 for r in rows if r["baseline_reaches_gold"])
    t = sum(1 for r in rows if r["treatment_reaches_gold"])
    print(f"baseline  (SESSION_DATES)      reaches gold: {b}/{len(rows)}")
    print(f"treatment (evidence-turn date) reaches gold: {t}/{len(rows)}")
    print()
    for r in rows:
        mark = ""
        if r["treatment_reaches_gold"] and not r["baseline_reaches_gold"]:
            mark = "  <== RECOVERED"
        print(f"  {r['question_id']}: '{r['reader_output']}' gold='{r['gold']}'{mark}")
        print(f"      baseline : ref={r['scorer_reference_date']} -> {r['baseline_resolved']}")
        print(f"      treatment: {r['evidence_turn']} @ {r['evidence_session_datetime']} -> {r['treatment_resolved']}")

    out = Path("results/m30_26/phase6_deterministic_temporal.json")
    out.write_text(json.dumps({
        "note": "treatment uses oracle evidence-turn date; NOT production performance",
        "baseline_reaches_gold": b, "treatment_reaches_gold": t, "total": len(rows),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
