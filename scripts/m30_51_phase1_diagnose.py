#!/usr/bin/env python3
"""M30-51 Phase 1: Diagnose EvidenceScorer scores for mechanism A cases."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer
from artificial_memory.core.ir.memory_types import ApexMemoryUnit

locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
scorer = UniversalEvidenceScorer()

MECH_A = {
    "lc_conv-26_63": {"conv": "conv-26", "q": "conv-26-qa-063", "rel": "next month"},
    "lc_conv-42_57": {"conv": "conv-42", "q": "conv-42-qa-057", "rel": "tomorrow"},
    "lc_conv-47_43": {"conv": "conv-47", "q": "conv-47-qa-043", "rel": "next month"},
    "lc_conv-47_65": {"conv": "conv-47", "q": "conv-47-qa-065", "rel": "currently"},
    "lc_conv-48_12": {"conv": "conv-48", "q": "conv-48-qa-012", "rel": "yesterday"},
    "lc_conv-50_52": {"conv": "conv-50", "q": "conv-50-qa-052", "rel": "next month"},
}

for qid, info in MECH_A.items():
    print(f"\n{'='*70}")
    print(f"{qid}  (question asks for '{info['rel']}')")
    print(f"{'='*70}")

    for idx, sid in enumerate(sids):
        if sid != info["conv"]:
            continue
        turns, questions, ir_records = adapter.load_conversation(idx)
        q = next(q for q in questions if q.question_id == info["q"])
        print(f"Question: {q.question}")

        for i, rec in enumerate(ir_records):
            body = rec.raw_content.lower()
            if not re.search(r"\b(yesterday|tomorrow|next month|last month|currently)\b", body, re.I):
                continue
            unit = ApexMemoryUnit(ir=rec)
            score = scorer.compute_score(q.question, unit)
            rel_in_body = re.search(r"\b(yesterday|tomorrow|next month|last month|currently)\b", body, re.I)
            rel_expr = rel_in_body.group(0) if rel_in_body else "?"
            print(f"  IR[{i}]: total={score.total_score:.1f}  lex={score.s_lexical:.0f} "
                  f"act={score.s_actor:.0f} temp={score.s_temporal:.0f} prov={score.s_provenance:.0f}  "
                  f"rel='{rel_expr}'")
            print(f"    raw: {rec.raw_content[:120]}")
