#!/usr/bin/env python3
"""Debug: why does turn selection pick D28:32 over D26:19 for lc_conv-42_57?"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
exec(open("scripts/m30_50_phase1_postprocess.py").read().split("def main")[0])

locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
turns, questions, ir_records = adapter.load_conversation(sids.index("conv-42"))
q = next(q for q in questions if q.question_id == "conv-42-qa-057")
pcc = adapter.compiler.compile(q.question, ir_records)
ctx = pcc.context_text

print(f"Question: {q.question}")
print()

qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", q.question.lower())
          if w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())
qkw = qkw - STOP

print(f"Question keywords: {sorted(qkw)}")
print()

# Find all turns with temporal expressions
for line in ctx.split("\n"):
    m = TURN_RE.match(line.strip())
    if not m:
        continue
    dia, hdr, body = m.group(1), m.group(2), m.group(3)
    if re.search(r"\b(yesterday|tomorrow|next month|last month|currently)\b", body, re.I):
        bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in STOP)
        score = len(qkw & bkw)
        print(f"  {dia} [{hdr}] score={score} overlap={qkw & bkw}")
        print(f"    body: {body[:120]}")
        print()
