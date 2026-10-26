#!/usr/bin/env python3
"""M30-51 Phase 3: Check what evidence actually reaches the compiled context.

For the Mechanism A cases where the scorer ranks the correct turn high,
is that turn actually IN the compiled context where the reader can see it?

We check:
1. Does D26:19 (lc_conv-42_57) appear in compiled ctx? Is it near the top?
2. Does the reader see the right turn or get distracted by higher-scoring turns?
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

# From M30-48: correct evidence for lc_conv-42_57 is D26:19
# (header 4 Nov 2022, body "Sure! I'll come over tomorrow if...")
# Gold: 5 November, 2022

MECH_A_CASES = {
    "lc_conv-26_63": {"conv": "conv-26", "cqid": "conv-26-qa-063", "gold_turn": "D15:11"},
    "lc_conv-42_57": {"conv": "conv-42", "cqid": "conv-42-qa-057", "gold_turn": "D26:19"},
    "lc_conv-47_43": {"conv": "conv-47", "cqid": "conv-47-qa-043", "gold_turn": "D20:17"},  # or D20:18
    "lc_conv-47_65": {"conv": "conv-47", "cqid": "conv-47-qa-065", "gold_turn": "D31:1"},
    "lc_conv-48_12": {"conv": "conv-48", "cqid": "conv-48-qa-012", "gold_turn": "D2:7"},
    "lc_conv-50_52": {"conv": "conv-50", "cqid": "conv-50-qa-052", "gold_turn": "D24:17"},
}

locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\]")

for qid, info in MECH_A_CASES.items():
    idx = sids.index(info["conv"])
    turns, questions, ir_records = adapter.load_conversation(idx)
    q = next(q for q in questions if q.question_id == info["cqid"])
    pcc = adapter.compiler.compile(q.question, ir_records)
    ctx = pcc.context_text

    print(f"\n{'='*60}")
    print(f"{qid}: gold turn = {info['gold_turn']}")
    print(f"  Question: {q.question}")
    
    # Find gold turn in compiled context
    lines = ctx.split("\n")
    found_at = None
    for i, line in enumerate(lines):
        if info["gold_turn"] in line:
            found_at = i
            print(f"  Gold turn found at line {i}: {line[:100]}")
            # Check if it has a relative expr or explicit date
            if re.search(r"(tomorrow|yesterday|next month|currently)", line, re.I):
                print(f"    -> contains relative expression")

    if found_at is None:
        print(f"  *** Gold turn NOT in compiled context ***")
    else:
        print(f"  Position: {found_at}/{len(lines)} lines ({found_at/len(lines)*100:.1f}%)")
    
    # Check what the reader actually output
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
    bp = next(r for r in prod["rows"] if r["qid"] == qid)
    print(f"  Reader output: {bp['predicted'][:80]!r}")
