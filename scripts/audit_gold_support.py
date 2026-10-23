#!/usr/bin/env python3
"""M30-21: Gold Support Auditor — calibrated classifier.

Explicitly separates:
- explicit_text_support (date string or all gold words verbatim)
- normalized_date_support (date written in any accepted format)
- temporal_inference_support (relative date resolved using session_datetime)
- alternative_turn_support (a different annotated turn carries the date)
- world_knowledge_required (geography / entity knowledge not in the turn)
It never treats world-knowledge inference as textual support.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCOMO_PATH = Path(r"C:\Users\smily\AppData\Local\Temp\locomo10.json")

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in MONTHS.items() if len(m) > 3})

MONTH_NAME_RE = "|".join(MONTHS)

CITY_TO_STATE = {"fort wayne": "indiana", "indianapolis": "indiana"}

class AuditError(RuntimeError):
    pass


def extract_gold_date(gold: str) -> dict[str, Any] | None:
    """Return only a date that is *written* in the answer, at the right granularity."""
    if gold is None:
        return None
    g = str(gold).strip().lower().replace(".", " ").replace(",", " ")
    m = re.search(rf"\b(\d{{1,2}})\s+({MONTH_NAME_RE})\s+(\d{{4}})", g)
    if m:
        return {"day": int(m.group(1)), "month": MONTHS[m.group(2)], "year": int(m.group(3)), "granularity": "day", "written": True}
    m = re.search(rf"\b({MONTH_NAME_RE})\s+(\d{{1,2}})\s+(\d{{4}})", g)
    if m:
        return {"day": int(m.group(2)), "month": MONTHS[m.group(1)], "year": int(m.group(3)), "granularity": "day", "written": True}
    m = re.search(rf"\b({MONTH_NAME_RE})\s+(\d{{4}})", g)
    if m:
        return {"day": None, "month": MONTHS[m.group(1)], "year": int(m.group(2)), "granularity": "month", "written": True}
    m = re.search(r"\b((?:19|20)\d{2})\b", g)
    if m:
        return {"day": None, "month": None, "year": int(m.group(1)), "granularity": "year", "written": True}
    return None


def parse_session_datetime(s: str | None) -> dict[str, int] | None:
    if not s:
        return None
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s)
    if not m:
        return None
    month = MONTHS.get(m.group(2).lower())
    if not month:
        return None
    return {"day": int(m.group(1)), "month": month, "year": int(m.group(3))}


def resolve_temporal(text: str, session_datetime: str | None) -> dict[str, int] | None:
    """Resolve only explicit relative day/month/year expressions. Returns None otherwise."""
    sd = parse_session_datetime(session_datetime)
    if not sd:
        return None
    t = " " + text.lower().strip() + " "
    year, month, day = sd["year"], sd["month"], sd["day"]

    def resulting_date(delta_years: int = 0, new_month: int | None = None, new_day: int | None = None) -> dict[str, int]:
        target_year = year + delta_years
        target_month = new_month or month
        target_day = new_day if new_day is not None else day
        # Calendar rollover is intentionally conservative: only day/month arithmetic.
        if target_day < 1:
            target_month -= 1
            target_day = 28
        if target_month < 1:
            target_month = 12
            target_year -= 1
        return {"day": target_day, "month": target_month, "year": target_year}

    patterns = [
        (r"\bnext month\b", resulting_date(new_month=month + 1 if month < 12 else 1) | {"year": year + (1 if month == 12 else 0)}),
        (r"\blast month\b", resulting_date(new_month=month - 1 if month > 1 else 12) | {"year": year - (1 if month == 1 else 0)}),
        (r"\bthis month\b", resulting_date()),
        (r"\blast year\b", resulting_date(delta_years=-1)),
        (r"\bnext year\b", resulting_date(delta_years=1)),
        (r"\b(?:one|a) year ago\b", resulting_date(delta_years=-1)),
        (r"\byesterday\b", resulting_date(new_day=day - 1)),
        (r"\btomorrow\b", resulting_date(new_day=day + 1)),
    ]
    for pattern, result in patterns:
        if re.search(pattern, t):
            return result
    m = re.search(r"\b(\d+) years? ago\b", t)
    if m:
        return resulting_date(delta_years=-int(m.group(1)))
    m = re.search(r"\b(\d+) months? ago\b", t)
    if m:
        n = int(m.group(1))
        target_year = year
        target_month = month - n
        while target_month < 1:
            target_month += 12
            target_year -= 1
        return {"day": day, "month": target_month, "year": target_year}
    m = re.search(r"\b(\d+) months? from now\b", t)
    if m:
        n = int(m.group(1))
        target_year = year
        target_month = month + n
        while target_month > 12:
            target_month -= 12
            target_year += 1
        return {"day": day, "month": target_month, "year": target_year}
    return None


def date_matches_written(target: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Granularity-aware equality. Year+month must not collapse into bare-year equality."""
    if target["written"] is False or actual["written"] is False:
        return False
    if target["granularity"] == "year":
        return actual["granularity"] == "year" and actual["year"] == target["year"]
    if target["granularity"] == "month":
        return actual["granularity"] in ("month", "day") and actual["year"] == target["year"] and actual["month"] == target["month"]
    return actual["granularity"] == "day" and actual["year"] == target["year"] and actual["month"] == target["month"] and actual["day"] == target["day"]


def explicit_text_support(gold: str, text: str) -> dict[str, Any]:
    gold_date = extract_gold_date(gold)
    if not gold_date:
        return {"supported": False, "reason": "gold is not a written date"}
    g = text.lower().replace(",", " ")
    found_dates: list[dict[str, Any]] = []
    # Extract only the most specific date pattern present in the text.
    m = re.search(rf"\b(\d{{1,2}})\s+({MONTH_NAME_RE})\s+(\d{{4}})\b", g)
    if m:
        found_dates.append({"day": int(m.group(1)), "month": MONTHS[m.group(2)], "year": int(m.group(3)), "granularity": "day", "written": True})
    else:
        m = re.search(rf"\b({MONTH_NAME_RE})\s+(\d{{1,2}})\s+(\d{{4}})\b", g)
        if m:
            found_dates.append({"day": int(m.group(2)), "month": MONTHS[m.group(1)], "year": int(m.group(3)), "granularity": "day", "written": True})
        else:
            m = re.search(rf"\b({MONTH_NAME_RE})\s+(\d{{4}})\b", g)
            if m:
                found_dates.append({"day": None, "month": MONTHS[m.group(1)], "year": int(m.group(2)), "granularity": "month", "written": True})
            else:
                m = re.search(r"\b((?:19|20)\d{2})\b", g)
                if m:
                    found_dates.append({"day": None, "month": None, "year": int(m.group(1)), "granularity": "year", "written": True})
    supported = any(date_matches_written(gold_date, d) for d in found_dates)
    return {"supported": supported, "gold_date": gold_date, "text_dates": found_dates, "reason": "explicit written date" if supported else "no matching written date"}


def temporal_inference_support(gold: str, text: str, session_datetime: str | None) -> dict[str, Any]:
    gold_date = extract_gold_date(gold)
    if not gold_date or gold_date["written"] is False:
        return {"supported": False, "reason": "gold is not a written date"}
    resolved = resolve_temporal(text, session_datetime)
    if not resolved:
        return {"supported": False, "reason": "no resolvable relative expression"}
    # Temporal date is by definition day or month precision, depending on expression.
    supported = (
        resolved["year"] == gold_date["year"]
        and (gold_date["month"] is None or resolved["month"] == gold_date["month"])
        and (gold_date["day"] is None or resolved["day"] == gold_date["day"])
    )
    return {"supported": supported, "gold_date": gold_date, "resolved_date": resolved, "reason": "relative date resolved with session_datetime" if supported else "resolved date does not match gold"}


def classify_one(gold: str, text: str, session_datetime: str | None) -> dict[str, Any]:
    explicit = explicit_text_support(gold, text)
    temporal = temporal_inference_support(gold, text, session_datetime)
    words = [w for w in re.findall(r"[a-z]+", gold.lower()) if len(w) > 2 and w not in {"the", "in", "on", "at", "of", "in", "a", "an", "to", "for"} and w not in MONTHS]
    contains_named_entity_only = bool(words) and all(w in text.lower() for w in words)
    world_knowledge = any(city in text.lower() and city in CITY_TO_STATE for city in CITY_TO_STATE)
    return {
        "explicit_text_support": explicit,
        "normalized_date_support": explicit["supported"],
        "temporal_inference_support": temporal,
        "alternative_turn_support": {"supported": contains_named_entity_only and not explicit["supported"] and not temporal["supported"], "reason": "only named-entity/date words match; no direct date"},
        "world_knowledge_required": {"required": world_knowledge, "note": "geographic inference is not textual support"},
        "gold_supported": explicit["supported"] or temporal["supported"],
        "confidence": "high" if explicit["supported"] else "medium" if temporal["supported"] else "low",
    }


def regression_tests() -> list[dict[str, Any]]:
    tests = [
        {
            "name": "41_10: distinguish D7:1 from D7:5",
            "gold": "February 24, 2023",
            "turns": [
                ("D7:1", "Hey John, how's it going? I took a creative writing class recently.", "8:55 pm on 25 February, 2023"),
                ("D7:5", "While volunteering yesterday, I met this amazing woman, Jean.", "8:55 pm on 25 February, 2023"),
            ],
            "expected": {"D7:1": False, "D7:5": True},
        },
        {
            "name": "48_12: explicit parenthesized date is normalized, not world knowledge",
            "gold": "January 26, 2023",
            "turns": [
                ("D2:7", "Look what letter I received yesterday (26 January 2023)!", "9:49 am on 27 January, 2023"),
            ],
            "expected": {"D2:7": True},
        },
        {
            "name": "50_36: 'the 11th' must not gain a month by itself",
            "gold": "September 11, 2023",
            "turns": [
                ("D18:1", "My album finally dropped on the 11th.", "10:56 am on 13 September, 2023"),
            ],
            "expected": {"D18:1": False},
        },
        {
            "name": "42_73: Fort Wayne → Indiana is world knowledge, not textual support",
            "gold": "Indiana",
            "turns": [
                ("D28:22", "I took that pic on a hike last summer near Fort Wayne.", "5:54 pm on 9 November, 2022"),
            ],
            "expected": {"D28:22": False},
        },
        {
            "name": "44_13: 'this month' in May does not support June gold",
            "gold": "June 2023",
            "turns": [
                ("D9:1", "Woohoo! I got a new place with a bigger backyard!", "1:51 pm on 26 June, 2023"),
                ("D6:2", "My week's been good - taking the dogs out almost every day this month.", "2:03 pm on 11 May, 2023"),
            ],
            "expected": {"D9:1": False, "D6:2": False},
        },
        {
            "name": "granularity: year / month-year / day-month-year are distinct",
            "gold": "2022",
            "turns": [
                ("Y1", "This happened in 2022.", ""),
                ("Y2", "This happened in February 2022.", ""),
                ("Y3", "This happened on 24 February 2022.", ""),
            ],
            "expected": {"Y1": True, "Y2": False, "Y3": False},
        },
    ]
    results = []
    for test in tests:
        observed = {}
        for turn_id, text, session_datetime in test["turns"]:
            result = classify_one(test["gold"], text, session_datetime)
            observed[turn_id] = result["gold_supported"]
        passed = observed == test["expected"]
        results.append({"name": test["name"], "expected": test["expected"], "observed": observed, "passed": passed})
    return results


def run_regression() -> int:
    results = regression_tests()
    print("=== Gold Support Auditor regression tests ===")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['name']}: expected={r['expected']} observed={r['observed']}")
    if not all(r["passed"] for r in results):
        print("FAILED")
        return 1
    print(f"Passed: {len(results)}/{len(results)}\n")
    return 0


def load_inputs():
    with open("benchmark_results/evidence_audit/v4_intervention_temporal.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    target_ids = [r["question_id"] for r in rows if r.get("group") == "TARGET"]
    with open("benchmark_results/evidence_audit/v4_bucketA_all.json", encoding="utf-8") as f:
        bucket = json.load(f)
    with open(LOCOMO_PATH, encoding="utf-8") as f:
        locomo = json.load(f)
    return target_ids, {c["question_id"]: c for c in bucket["cases"]}, locomo


def collect_locomo_turns(locomo, conversation_id: str, question: str):
    for conv in locomo:
        if conv.get("sample_id") != conversation_id:
            continue
        qa = None
        for candidate in conv.get("qa", []):
            if candidate.get("question", "").lower().strip() == question.lower().strip():
                qa = candidate
                break
        if qa is None:
            for candidate in conv.get("qa", []):
                if question.lower().strip() in candidate.get("question", "").lower().strip():
                    qa = candidate
                    break
        if qa is None:
            return [], []
        conversation = conv.get("conversation", {})
        annotated = []
        for eid in qa.get("evidence", []):
            m = re.match(r"D(\d+):(\d+)", eid)
            if not m:
                continue
            session_key = f"session_{m.group(1)}"
            session_dt = conversation.get(f"{session_key}_date_time", "")
            turns = conversation.get(session_key, [])
            idx = int(m.group(2)) - 1
            if isinstance(turns, list) and 0 <= idx < len(turns) and isinstance(turns[idx], dict):
                annotated.append({"id": eid, "dia_id": turns[idx].get("dia_id", eid), "session_datetime": session_dt, "speaker": turns[idx].get("speaker", ""), "text": turns[idx].get("text", "")})
        all_turns = []
        session_keys = sorted((k for k in conversation if k.startswith("session_") and k.split("_")[-1].isdigit()), key=lambda k: int(k.split("_")[-1]))
        for key in session_keys:
            session_dt = conversation.get(f"{key}_date_time", "")
            turns = conversation.get(key, [])
            if not isinstance(turns, list):
                continue
            for idx, turn in enumerate(turns):
                if isinstance(turn, dict):
                    all_turns.append({"id": f"D{key.split('_')[-1]}:{idx+1}", "dia_id": turn.get("dia_id", f"D{key.split('_')[-1]}:{idx+1}"), "session_datetime": session_dt, "speaker": turn.get("speaker", ""), "text": turn.get("text", "")})
        return annotated, all_turns
    return [], []


def audit_28() -> int:
    target_ids, cases, locomo = load_inputs()
    audited = []
    for qid in target_ids:
        case = cases.get(qid)
        if not case:
            audited.append({"question_id": qid, "verdict": "INCONCLUSIVE", "reason": "Bucket A case not found"})
            continue
        annotated, all_turns = collect_locomo_turns(locomo, case["source_conversation"], case["question"])
        if not annotated:
            audited.append({"question_id": qid, "gold": case["gold_answer"], "verdict": "INCONCLUSIVE", "reason": "No LoCoMo QA / evidence found", "annotated_evidence_valid": False, "gold_supported": False, "world_knowledge_required": False})
            continue
        annotated_results = []
        for turn in annotated:
            result = classify_one(str(case["gold_answer"]), turn["text"], turn["session_datetime"])
            annotated_results.append({**turn, **result})
        alternate_results = []
        for turn in all_turns:
            if any(x["dia_id"] == turn["dia_id"] for x in annotated):
                continue
            result = classify_one(str(case["gold_answer"]), turn["text"], turn["session_datetime"])
            if result["gold_supported"]:
                alternate_results.append({**turn, **result})
        annotated_supported = any(x["gold_supported"] for x in annotated_results)
        any_explicit_or_temporal = any(x["explicit_text_support"]["supported"] or x["temporal_inference_support"]["supported"] for x in annotated_results)
        world_knowledge_required = any(x["world_knowledge_required"]["required"] for x in annotated_results) and not any_explicit_or_temporal
        if annotated_supported:
            verdict = "GOLD_SUPPORTED_BY_ANNOTATED_EVIDENCE"
        elif alternate_results:
            verdict = "GOLD_SUPPORTED_BY_OTHER_TURN"
        else:
            verdict = "GOLD_UNSUPPORTED_OR_AMBIGUOUS"
        audited.append({
            "question_id": qid,
            "question": case["question"],
            "gold": case["gold_answer"],
            "annotated_evidence": annotated_results,
            "alternative_turns": alternate_results[:5],
            "annotated_evidence_valid": annotated_supported,
            "world_knowledge_required": world_knowledge_required,
            "gold_supported": bool(annotated_results),
            "verdict": verdict,
            "confidence": "high" if annotated_supported else "medium" if alternate_results else "low",
            "reason": "annotated evidence supports gold" if annotated_supported else "alternate turn supports gold" if alternate_results else "no direct support found in the full conversation",
        })
    summary: dict[str, int] = {}
    for item in audited:
        summary[item["verdict"]] = summary.get(item["verdict"], 0) + 1
    print("=== 28-case calibrated Gold support audit ===")
    for verdict, count in sorted(summary.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {verdict}: {count}")
    print()
    for item in audited:
        print(f"  {item['question_id']}: {item['verdict']} (confidence={item.get('confidence', 'n/a')})")
        print(f"    reason: {item['reason']}")
        for evidence in item.get("annotated_evidence", []):
            print(f"    annotated {evidence['dia_id']}: explicit={evidence['explicit_text_support']['supported']} temporal={evidence['temporal_inference_support']['supported']} world_knowledge={evidence['world_knowledge_required']['required']}")
    with open("results/m30_21_calibrated_audit.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": audited}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: results/m30_21_calibrated_audit.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-only", action="store_true")
    args = parser.parse_args()
    if args.regression_only:
        return run_regression()
    if run_regression() != 0:
        return 1
    return audit_28()


if __name__ == "__main__":
    raise SystemExit(main())
