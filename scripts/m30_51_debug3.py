#!/usr/bin/env python3
import sys, re, json
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
exec(open("scripts/m30_51_phase2_fallback.py").read().split("def main")[0])

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

# Test on lc_conv-42_57 (where reader gave wrong date 10 November)
turns, questions, ir_records = adapter.load_conversation(sids.index("conv-42"))
q = next(q for q in questions if q.question_id == "conv-42-qa-057")
pcc = adapter.compiler.compile(q.question, ir_records)
ctx = pcc.context_text

print(f"Question: {q.question}")
result = deterministic_temporal_resolve(ctx, q.question, q.category)
print(f"Result: {result!r}")
print(f"Gold: 5 November, 2022.")
print()

# Show top candidates
qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", q.question.lower())
          if w not in STOP and w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
print(f"qkw: {sorted(qkw)}")

candidates = []
for line in ctx.split("\n"):
    m = TURN_RE.match(line.strip())
    if not m: continue
    dia, hdr, body = m.group(1), m.group(2), m.group(3)
    rels = REL_RE.findall(body)
    if not rels: continue
    base = parse_header(hdr)
    if not base: continue
    bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in STOP)
    score = len(qkw & bkw)
    candidates.append((score, dia, base, rels, body))

candidates.sort(key=lambda c: (-c[0], 0))
for c in candidates[:5]:
    print(f"  {c[1]}: score={c[0]} base={c[2].date()} rels={c[3]}")
    print(f"    body: {c[4][:100]}")

# Now test resolve
if candidates:
    best = candidates[0]
    print(f"\nBest turn: {best[1]} with rels={best[3]}")
    resolved = resolve_relative(best[3][0], best[2])
    print(f"Resolved: {resolved.date() if resolved else None}")
    if resolved:
        print(f"Formatted: {resolved.day} {resolved.strftime('%B')}, {resolved.year}")
