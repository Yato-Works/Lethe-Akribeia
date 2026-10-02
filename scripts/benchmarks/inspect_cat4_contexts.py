import sys
sys.stdout.reconfigure(encoding='utf-8')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns26, q26, ir26 = adapter.load_conversation(conv_idx=0)

target_qids = [
    'conv-26-qa-085', 'conv-26-qa-088', 'conv-26-qa-089',
    'conv-26-qa-094', 'conv-26-qa-111', 'conv-26-qa-140',
    'conv-26-qa-141', 'conv-26-qa-142'
]

for qid in target_qids:
    q = next(q for q in q26 if q.question_id == qid)
    print(f"\n==========================================")
    print(f"[{qid}] Q: {q.question}")
    print(f"GT: {q.ground_truth}")
    print(f"Evidence IDs: {q.evidence_ids}")
    for evid in q.evidence_ids:
        t = next((t for t in turns26 if t.dia_id == evid), None)
        if t:
            print(f"  [Evidence {evid}] {t.speaker}: {t.text}")
    pcc = adapter.compiler.compile(q.question, ir26)
    gt_in_pcc = any(w.lower() in pcc.context_text.lower() for w in q.ground_truth.split() if len(w) > 3)
    print(f"Context length: {len(pcc.context_text)}, GT words in context: {gt_in_pcc}")
