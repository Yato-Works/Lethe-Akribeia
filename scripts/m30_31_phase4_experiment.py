#!/usr/bin/env python3
"""M30-31 Phase 4: Minimal controlled experiment on lc_conv-41_10.

Baseline: M30-3 Reader output ("I don't know.")
Treatment: Treatment A's Reader output + M30-29 Temporal Resolution Engine

Expected: "Maria met Jean yesterday while volunteering." → "Maria met Jean February 24, 2023 while volunteering."
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}


def parse_session_datetime(s):
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def resolve_yesterday(base_date):
    """Resolve 'yesterday' to an absolute date, handling month/year boundaries."""
    if not base_date:
        return None
    from datetime import datetime, timedelta
    y, mo, d = base_date
    dt = datetime(y, mo, d)
    yesterday = dt - timedelta(days=1)
    return (yesterday.year, yesterday.month, yesterday.day)


def format_date(date_tuple):
    if not date_tuple:
        return None
    y, mo, d = date_tuple
    month_name = MONTH_NAMES.get(mo, "")
    return f"{month_name} {d}, {y}"


def main():
    print("=" * 78)
    print("M30-31 Phase 4: Minimal Controlled Experiment — lc_conv-41_10")
    print("=" * 78)
    print()
    
    # Load data
    m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    m30_30 = json.loads((PROJECT_ROOT / "results/m30_30/treatment_comparison.json").read_text(encoding="utf-8"))
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    locomo_by_id = {c["sample_id"]: c for c in locomo}
    
    qid = "lc_conv-41_10"
    conv_key = "conv-41"
    
    # Gold and question
    question = "When did Maria meet Jean?"
    gold = "February 24, 2023"
    
    # Baseline: M30-3 Reader output
    baseline_output = m30_3_results.get(qid, {}).get("reader_text", "")
    print(f"Baseline (M30-3): {baseline_output}")
    
    # Treatment A: M30-30 Reader output (improved retrieval)
    treatment_a_output = m30_30["reader_a_results"].get(qid, {}).get("reader_output", "")
    print(f"Treatment A (M30-30): {treatment_a_output}")
    
    # D7:5 timestamp
    conv_data = locomo_by_id.get(conv_key, {}).get("conversation", {})
    skey = "session_7"
    s7_dt = conv_data.get(f"{skey}_date_time", "")
    s7_parsed = parse_session_datetime(s7_dt)
    print(f"D7:5 timestamp: {s7_dt} -> {s7_parsed}")
    
    # Apply Temporal Resolution to Treatment A output
    # "yesterday" from 2023-02-25 = 2023-02-24 = February 24, 2023
    treatment_output = treatment_a_output
    if "yesterday" in treatment_a_output.lower() and s7_parsed:
        yesterday = resolve_yesterday(s7_parsed)
        yesterday_str = format_date(yesterday)
        treatment_output = treatment_a_output.replace("yesterday", yesterday_str)
    
    print(f"Treatment (A + Temporal Resolution): {treatment_output}")
    print()
    
    # Score
    grader = OfficialLongMemEvalGrader()
    qtype = "temporal-reasoning"
    
    print("Scorer results:")
    tests = [
        ("Baseline", baseline_output),
        ("Treatment A", treatment_a_output),
        ("Treatment A + Temporal Resolution", treatment_output),
    ]
    
    for name, output in tests:
        try:
            result = grader.grade(qid, qtype, output, gold)
            print(f"  {name}: {output[:60]} -> is_correct={result.is_correct}")
        except Exception as e:
            print(f"  {name}: {output[:60]} -> ERROR: {e}")
    
    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    
    # Check if the treatment recovered the case
    treatment_result = grader.grade(qid, qtype, treatment_output, gold)
    baseline_result = grader.grade(qid, qtype, baseline_output, gold)
    
    if treatment_result.is_correct and not baseline_result.is_correct:
        print("IMPROVEMENT CONFIRMED: lc_conv-41_10 recovered")
    elif treatment_result.is_correct and baseline_result.is_correct:
        print("NO CHANGE: Both correct")
    elif not treatment_result.is_correct and not baseline_result.is_correct:
        print("NO IMPROVEMENT: Both incorrect")
    else:
        print("REGRESSION: Treatment made it worse")


if __name__ == "__main__":
    main()
