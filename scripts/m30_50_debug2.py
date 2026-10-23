#!/usr/bin/env python3
"""Debug: trace the post_process on lc_conv-42_57."""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july",
    "august","september","october","november","december"], start=1)}
LINE_RE = re.compile(r"^\[SPEAKER:[^\]]+\]\s*\[(D\d+:\d+) on ([^\]]+)\](.*)$")

def parse_header(sd):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", sd or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None

prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
bp = next(r for r in prod["rows"] if r["qid"] == "lc_conv-42_57")
raw = bp["predicted"]
gold = bp["gold"]
print(f"Raw: {raw[:120]!r}")
print(f"Gold: {gold!r}")

locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
turns, questions, ir_records = adapter.load_conversation(sids.index("conv-42"))
q = next(q for q in questions if q.question_id == "conv-42-qa-057")
pcc = adapter.compiler.compile(q.question, ir_records)
ctx = pcc.context_text

# Find ALL turns with 'tomorrow' in body
count = 0
for m in LINE_RE.finditer(ctx):
    dia, hdr, body = m.group(1), m.group(2), m.group(3)
    if "tomorrow" in body.lower() or "next day" in body.lower():
        count += 1
        base = parse_header(hdr)
        if base:
            resolved = base + timedelta(days=1)
            resolved_str = f"{resolved.day} {resolved.strftime('%B')}, {resolved.year}"
            reader_match = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", raw)
            reader_date = reader_match.group(1) if reader_match else None
            print(f"Turn {dia}: hdr={hdr}, base={base.date()}, resolved={resolved.date()}")
            print(f"  resolved_str: {resolved_str}")
            print(f"  reader_date:  {reader_date}")
            print(f"  match: {reader_match and reader_date != resolved_str}")

print(f"\nTotal turns with 'tomorrow'/'next day' in body: {count}")
print(f"Current logic: if reader_date != resolved_str -> return resolved_str")
print(f"But if reader said CORRECT date, we still return it... and if WRONG, we fix it.")

# The real issue: reader said "10 November 2022"
# Correct evidence turn: D26:19 with tomorrow -> Nov 5
# But there may be other "tomorrow" turns in context
