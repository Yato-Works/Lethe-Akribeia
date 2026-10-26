#!/usr/bin/env python3
"""M30-46 Phase 2b: Why does the reader abstain when evidence is at position 0.001?

Take lc_conv-48_12 (evidence at rel 0.001, reader says "no mention").
Print the first 1500 chars of the ACTUAL compiled context, so we can see
what the reader sees at the top.

This is the decisive diagnostic before any position experiment.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

TARGETS = {
    "conv-48-qa-012": "lc_conv-48_12",
    "conv-47-qa-065": "lc_conv-47_65",
    "conv-26-qa-063": "lc_conv-26_63",
}


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            if q.question_id not in TARGETS:
                continue
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text
            print("=" * 78)
            print(f"{TARGETS[q.question_id]}")
            print("=" * 78)
            print(f"Question: {q.question}")
            print(f"Gold:     {q.ground_truth}")
            print(f"Context length: {len(ctx)}")
            print()
            print("--- FIRST 1200 CHARS ---")
            print(ctx[:1200])
            print()
            print("--- LINES CONTAINING A DATE HEADER (first 6) ---")
            for ln in ctx.split("\n")[:40]:
                if re.search(r"\[D\d+:\d+ on", ln):
                    print(f"  {ln[:140]}")
            print()

    out = PROJECT_ROOT / "results/m30_46"


if __name__ == "__main__":
    main()
