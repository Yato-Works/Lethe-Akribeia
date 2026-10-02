import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()

# List of target questions to deeply inspect
target_qids = [
    # Cat 1
    ('conv-41-qa-018', 2), ('conv-41-qa-040', 2), ('conv-30-qa-031', 1), ('conv-30-qa-023', 1),
    ('conv-26-qa-040', 0), ('conv-41-qa-007', 2), ('conv-49-qa-048', 8), ('conv-41-qa-009', 2),
    # Cat 4
    ('conv-26-qa-140', 0), ('conv-26-qa-141', 0), ('conv-26-qa-085', 0), ('conv-26-qa-111', 0),
    ('conv-26-qa-089', 0), ('conv-43-qa-137', 4), ('conv-47-qa-145', 6),
    # Cat 2
    ('conv-41-qa-020', 2), ('conv-42-qa-043', 3), ('conv-43-qa-016', 4), ('conv-43-qa-020', 4),
    ('conv-50-qa-050', 9),
    # Cat 3
    ('conv-26-qa-059', 0), ('conv-26-qa-069', 0), ('conv-42-qa-066', 3), ('conv-42-qa-068', 3),
    ('conv-42-qa-073', 3), ('conv-44-qa-052', 5)
]

# Cache loaded convs
conv_cache = {}

for qid, c_idx in target_qids:
    if c_idx not in conv_cache:
        turns, questions, ir = adapter.load_conversation(conv_idx=c_idx)
        conv_cache[c_idx] = (turns, {q.question_id: q for q in questions}, ir)
    turns, q_dict, ir = conv_cache[c_idx]
    q = q_dict[qid]
    pcc = adapter.compiler.compile(q.question, ir)
    
    print("=" * 60)
    print(f"[{qid}] Category {q.category}")
    print(f"Q: {q.question}")
    print(f"Gold: {q.ground_truth}")
    print(f"Evidence IDs: {q.evidence_ids}")
    for evid in q.evidence_ids:
        t = next((t for t in turns if t.dia_id == evid), None)
        if t:
            in_pcc = t.text[:30] in pcc.context_text or (t.speaker in pcc.context_text and t.dia_id in pcc.context_text)
            print(f"  [Turn {evid}] (In PCC: {in_pcc}) {t.speaker}: {t.text}")
