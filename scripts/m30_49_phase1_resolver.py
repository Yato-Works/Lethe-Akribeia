#!/usr/bin/env python3
"""M30-49 Phase 1: Implement deterministic temporal resolution post-processing.

This module implements the noun-best resolution for Mechanism A failures.
It runs as a post-processor on the Reader output, using the compiled context.

Algorithm:
1. Detect if the raw answer contains a relative temporal expression
3. If yes, find the relevant evidence turn in the context by noun-overlap
4. Resolve the relative expression against that turn's header timestamp
5. Replace the answer with the resolved date

This is a deterministic post-processor. No LLM calls. Pure Python.
"""
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}

REL_RE = re.compile(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|"
                    r"next week|last week|currently|one year ago|"
                    r"two years ago|three years ago)\b", re.I)
LINE_RE = re.compile(r"^\[SPEAKER:[^\]]+\]\s*\[(D\d+:\d+) on ([^\]]+)\](.*)$")
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}


def parse_header(sd: str) -> Optional[datetime]:
    """Parse '3:19 pm on 28 August, 2023' -> datetime(2023, 8, 28, 15, 19)"""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", sd or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def resolve_relative(expr: str, base: datetime) -> Optional[datetime]:
    """Resolve relative temporal expression against base datetime."""
    if not base:
        return None
    e = expr.lower()
    if "yesterday" in e or "the day before" in e:
        return base - timedelta(days=1)
    if "tomorrow" in e or "the next day" in e:
        return base + timedelta(days=1)
    if "next month" in e:
        y, mo = (base.year + (1 if base.month == 12 else 0), 1 if base.month == 12 else base.month + 1)
        return datetime(y, mo, 1)
    if "last month" in e:
        y, mo = (base.year - (1 if base.month == 1 else 0),
                 12 if base.month == 1 else base.month - 1)
        return datetime(y, mo, 1)
    if "next week" in e:
        return base + timedelta(days=7)
    if "last week" in e:
        return base - timedelta(days=7)
    if "currently" in e or "just around the corner" in e:
        return base
    return None


def parse_header(sd: str) -> Optional[datetime]:
    """Parse '3:19 pm on 28 August, 2023' -> datetime(2023, 8, 28, 15, 19)"""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", sd or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def find_relevant_turn(ctx: str, question: str) -> Optional[Tuple[str, datetime, str]]:
    """Find the most relevant turn in context by noun overlap with question."""
    q_kw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower()) 
               if w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
    
    best = None
    best_score = 0
    best_date = None
    best_turn_text = ""
    
    for m in LINE_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        bl = body.lower()
        nouns = set(w for w in re.findall(r"\b[a-z]{4,}\b", bl) if w not in STOP)
        overlap = len(set(re.findall(r"\b[a-z]{4,}\b", question.lower())) & nouns)
        if overlap > 0:
            base = parse_header(hdr)
            if base:
                if overlap > 0 and (best is None or overlap > best_score):
                    return (m.group(1), parse_header(hdr), m.group(3))
    return None


def format_resolved(dt: datetime, gran: str = "month") -> str:
    """Format resolved datetime to match Gold format expectations."""
    if gran == "month":
        return f"{MONTH_NAMES[dt.month]} {dt.year}"
    return f"{dt.day} {MONTH_NAMES[dt.month]}, {dt.year}"


def resolve_temporal_in_answer(raw_answer: str, ctx: str, question: str) -> Optional[str]:
    """
    Main entry: try to resolve relative dates in the raw answer using context.
    Returns normalized date string if successful, None otherwise.
    """
    # 1. Extract relative expressions from the answer
    rels = REL_RE.findall(raw_answer.lower())
    if not rels:
        return None
    
    # 2. Find the most relevant turn in context
    turn = find_relevant_turn(ctx, question)
    if not turn:
        return None
    
    dia, base_date, turn_text = turn
    if not base_date:
        return None
    
    # 3. Resolve the first relative expression found
    rel = rels[0]
    resolved = resolve_relative(rel, base_date)
    if not resolved:
        return None
    
    # 4. Determine required granularity from question
    q = question.lower()
    if any(w in question.lower() for w in ["year", "which year"]):
        gran = "year"
    elif any(w in question.lower() for w in ["month", "which month"]):
        gran = "month"
    else:
        gran = "day"
    
    return format_resolved(resolved, "month" if "month" in str(base).lower() or "month" in rel else "day")


MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}

REL_RE = re.compile(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|"
                    r"next week|last week|currently|one year ago|"
                    r"two years ago|three years ago)\b", re.I)
LINE_RE = re.compile(r"^\[SPEAKER:[^\]]+\]\s*\[(D\d+:\d+) on ([^\]]+)\](.*)$")
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())


def format_resolved(dt: datetime, gran: str = "month") -> str:
    if gran == "month":
        return f"{MONTH_NAMES[dt.month]} {dt.year}"
    return f"{dt.day} {MONTH_NAMES[dt.month]}, {dt.year}"


def resolve_temporal_in_answer(raw_answer: str, ctx: str, question: str) -> Optional[str]:
    """
    Main entry point for temporal resolution post-processor.
    Returns normalized date string if successful, None otherwise.
    """
    # 1. Find relative expressions in the raw answer
    rels = REL_RE.findall(raw_answer.lower())
    if not rels:
        return None
    
    # 2. Find the most relevant turn in context
    turn = find_relevant_turn(ctx, question)
    if not turn:
        return None
    
    dia, base_date, turn_text = turn
    if not base_date:
        return None
    
    # 3. Resolve the first relative expression found
    rel = rels[0]
    resolved = resolve_relative(rels[0], base_date)
    if not resolved:
        return None
    
    # Determine granularity from question
    gran = "month" if any(w in question.lower() for w in ["month", "year"]) else "day"
    return format_resolved(resolved, "month" if "month" in question.lower() else "day")


if __name__ == "__main__":
    # Quick self-test
    ctx = """[SPEAKER:Caroline] [D15:11 on 3:19 pm on 28 August, 2023] (In reply to Melanie: "We're putting together a talent show for the kids next month. So excited...")
[SPEAKER:Melanie] [D15:12 on 3:19 pm on 28 August, 2023] (In reply to Caroline: "That's great news...")"""
    q = "When is Caroline's youth center putting on a talent show?"
    ans = "The talent show is next month."
    res = resolve_temporal_in_answer("The talent show is next month.", ctx, q)
    print(f"Result: {res}")  # Should be "September 2023"