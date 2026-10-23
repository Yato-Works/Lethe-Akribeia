#!/usr/bin/env python3
"""M30-31 Phase 4c: Temporal Resolution + Answer Extraction.

The scorer requires exact match for temporal-reasoning questions.
Reader outputs full sentences like "Maria met Jean February 24, 2023 while volunteering."

Fix: After Temporal Resolution, extract just the answer (date/fact) from the output.
This is a deterministic post-processing step, not a scorer change.

Test on lc_conv-41_10:
- Reader output: "Maria met Jean yesterday while volunteering."
- After Temporal Resolution: "Maria met Jean February 24, 2023 while volunteering."
- After Answer Extraction: "February 24, 2023"
- Expected: is_correct=True
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}

RELATIVE_EXPRESSIONS = [
    (re.compile(r"\bnext month\b", re.I), "next_month"),
    (re.compile(r"\blast month\b", re.I), "last_month"),
    (re.compile(r"\bthis month\b", re.I), "this_month"),
    (re.compile(r"\bnext year\b", re.I), "next_year"),
    (re.compile(r"\blast year\b", re.I), "last_year"),
    (re.compile(r"\bone year ago\b", re.I), "one_year_ago"),
    (re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", re.I), "years_ago"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    (re.compile(r"\btomorrow\b", re.I), "tomorrow"),
    (re.compile(r"\btoday\b", re.I), "today"),
    (re.compile(r"\bnext week\b", re.I), "next_week"),
    (re.compile(r"\blast week\b", re.I), "last_week"),
]


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


def resolve_relative(expr_type, base_date, expr_text=None):
    if not base_date:
        return None
    y, mo, d = base_date
    
    if expr_type == "next_month":
        dt = datetime(y, mo, 1) + timedelta(days=32)
        return (dt.year, dt.month, None)
    if expr_type == "last_month":
        dt = datetime(y, mo, 1) - timedelta(days=1)
        return (dt.year, dt.month, None)
    if expr_type == "this_month":
        return (y, mo, None)
    if expr_type == "next_year":
        return (y + 1, None, None)
    if expr_type == "last_year":
        return (y - 1, None, None)
    if expr_type == "one_year_ago":
        dt = datetime(y, mo, d) - timedelta(days=365)
        return (dt.year, dt.month, dt.day)
    if expr_type == "years_ago" and expr_text:
        m = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago", expr_text, re.I)
        if m:
            num_str = m.group(1).lower()
            num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                       "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            if num_str.isdigit():
                n = int(num_str)
            else:
                n = num_map.get(num_str, 0)
            if n > 0:
                dt = datetime(y, mo, d) - timedelta(days=365 * n)
                return (dt.year, dt.month, dt.day)
        return None
    if expr_type == "yesterday":
        dt = datetime(y, mo, d) - timedelta(days=1)
        return (dt.year, dt.month, dt.day)
    if expr_type == "tomorrow":
        dt = datetime(y, mo, d) + timedelta(days=1)
        return (dt.year, dt.month, dt.day)
    if expr_type == "today":
        return (y, mo, d)
    return None


def format_resolved_date(resolved):
    """Format date as 'Month Day, Year' (American format, matching LongMemEval gold)."""
    if not resolved:
        return None
    y, mo, d = resolved
    if mo is None:
        return str(y)
    month_name = MONTH_NAMES.get(mo, "")
    if d is None:
        return f"{month_name} {y}"
    return f"{month_name} {d}, {y}"


def detect_relative_expression(text):
    results = []
    text_lower = text.lower()
    for pattern, expr_type in RELATIVE_EXPRESSIONS:
        for m in pattern.finditer(text_lower):
            results.append((expr_type, m.start(), m.end()))
    return results


def resolve_temporal_expressions(text, base_date):
    """Resolve relative date expressions in text."""
    new_text = text
    expressions = detect_relative_expression(text)
    
    for expr_type, start, end in expressions:
        expr_text = text[start:end]
        resolved = resolve_relative(expr_type, base_date, expr_text)
        if resolved:
            resolved_str = format_resolved_date(resolved)
            new_text = new_text.replace(expr_text, resolved_str)
    
    return new_text


def extract_answer(text, question_type="temporal"):
    """Extract the answer from a full sentence.
    
    For temporal questions: extract just the date.
    For other questions: extract the most relevant fact.
    """
    if question_type == "temporal":
        # Look for date patterns
        date_patterns = [
            # "February 24, 2023" or "24 February 2023"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}",
            # "2023" (year only)
            r"\b(?:19|20)\d{2}\b",
        ]
        for pattern in date_patterns:
            m = re.search(pattern, text, re.I)
            if m:
                return m.group(0)
    
    # For non-temporal, try to extract after common prefixes
    prefixes = [
        r"(?:is|was|are|were|did|does|do)\s+",
        r"(?:the answer is|answer is)\s+",
    ]
    for prefix in prefixes:
        m = re.search(prefix + r"(.+?)(?:\.|$)", text, re.I)
        if m:
            return m.group(1).strip()
    
    return text


def main():
    print("=" * 78)
    print("M30-31 Phase 4c: Temporal Resolution + Answer Extraction")
    print("=" * 78)
    print()
    
    # Load data
    m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    locomo_by_id = {c["sample_id"]: c for c in locomo}
    
    qid = "lc_conv-41_10"
    conv_key = "conv-41"
    question = "When did Maria meet Jean?"
    gold = "February 24, 2023"
    
    # Reader output (from M30-30 Treatment A)
    reader_output = "Maria met Jean yesterday while volunteering."
    print(f"Reader output: {reader_output}")
    
    # Get D7:5 timestamp
    conv_data = locomo_by_id.get(conv_key, {}).get("conversation", {})
    skey = "session_7"
    s7_dt = conv_data.get(f"{skey}_date_time", "")
    s7_parsed = parse_session_datetime(s7_dt)
    print(f"D7:5 timestamp: {s7_dt} -> {s7_parsed}")
    
    # Step 1: Temporal Resolution
    resolved_text = resolve_temporal_expressions(reader_output, s7_parsed)
    print(f"After Temporal Resolution: {resolved_text}")
    
    # Step 2: Answer Extraction
    extracted_answer = extract_answer(resolved_text, "temporal")
    print(f"After Answer Extraction: {extracted_answer}")
    print()
    
    # Score
    grader = OfficialLongMemEvalGrader()
    qtype = "temporal-reasoning"
    
    print("Scorer results:")
    tests = [
        ("Raw Reader output", reader_output),
        ("After Temporal Resolution", resolved_text),
        ("After Answer Extraction", extracted_answer),
    ]
    
    for name, output in tests:
        result = grader.grade(qid, qtype, output, gold)
        print(f"  {name}: {repr(output[:60])} -> is_correct={result.is_correct}")
    
    print()
    
    # Final result
    result = grader.grade(qid, qtype, extracted_answer, gold)
    if result.is_correct:
        print("IMPROVEMENT CONFIRMED")
    else:
        print("NO IMPROVEMENT")


if __name__ == "__main__":
    main()
