#!/usr/bin/env python3
import sys, re, json
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
exec(open("scripts/m30_51_phase2_fallback.py").read().split("def main")[0])

locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

turns, questions, ir_records = adapter.load_conversation(sids.index("conv-48"))
q = next(q for q in questions if q.question_id == "conv-48-qa-012")
pcc = adapter.compiler.compile(q.question, ir_records)
ctx = pcc.context_text

print(f"Question: {q.question}")
print(f"Category: {q.category}")
result = deterministic_temporal_resolve(ctx, q.question, q.category)
print(f"Result: {result!r}")

# Check candidates
candidates = []
for line in ctx.split("\n"):
    m = TURN_RE.match(line.strip())
    if not m: continue
    dia, hdr, body = m.group(1), m.group(2), m.group(3)
    rels = REL_RE.findall(body)
    if not rels: continue
    base = parse_header(hdr)
    print(f"  Candidate: {dia} hdr={hdr} rels={rels} base={base}")
    candidates.append((dia, base, rels))

print(f"Total candidates: {len(candidates)}")
