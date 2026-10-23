#!/usr/bin/env python3
"""Debug M30-52: why did lc_conv-26_63 stop being RECOVERED."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

scorer = UniversalEvidenceScorer()

# Load production results
prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
baseline = {r["qid"]: r for r in prod["rows"]}

# Load dataset
locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

qid_to_conv = {}
conv_ids_needed = set()
for qid in baseline:
    m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
    if not m: continue
    conv = f"conv-{m.group(1)}"
    conv_ids_needed.add(conv)
    qid_to_conv[qid] = f"{conv}-qa-{int(m.group(2)):03d}"

for idx, sid in enumerate(sids):
    if sid not in conv_ids_needed: continue
    turns, questions, ir_records = adapter.load_conversation(idx)
    for q in questions:
        cqid = q.question_id
        if cqid == "conv-26-qa-063":
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text

            print(f"Question: {q.question}")
            print(f"\n--- Context turns ---")
            TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)")
            candidates = []
            for m in TURN_RE.finditer(ctx):
                dia, hdr, body = m.group(1), m.group(2), m.group(3)
                rels = re.findall(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|next week|last week|currently)\b", body, re.I)
                if rels:
                    body_lower = body.lower()
                    actor_match = sum(1 for a in scorer.KNOWN_ACTORS if f" {a} " in f" {body_lower} ")
                    # Event entity
                    event_seed = ["carol", "youth", "center", "talent", "show"]
                    event_match = sum(1 for e in event_seed if e in body_lower)
                    print(f"  {dia} hdr={hdr[:30]} actors={actor_match} event={event_match} rels={rels} body={body[:80]!r}")
                    candidates.append((dia, hdr, body, rels, actor_match, event_match))

            # Now score the question
            print(f"\n--- Event entity extraction ---")
            q_lower = re.sub(r'\b(when|did|do|does|is|are|was|were|will|would|could|should|can|may|might)\b', '',
                             q.question, flags=re.I).strip(' ?')
            words = re.findall(r"\b[a-z]{4,}\b", q_lower.lower())
            event_seed = [w for w in words if w not in STOP]
            print(f"  Extracted: {event_seed}")
