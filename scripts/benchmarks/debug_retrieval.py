from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter, _evidence_present

adapter = LoCoMoAdapter()
weights = EvidenceScoreWeights()

conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
print(f"Loaded conv {conv_idx}: {len(turns)} turns, {len(questions)} questions")

target_qids = ['conv-41-qa-007', 'conv-41-qa-009', 'conv-41-qa-018', 'conv-41-qa-024', 'conv-41-qa-040']
for q in questions:
    if q.question_id in target_qids:
        print(f"\n==========================================")
        print(f"Question [{q.question_id}]: {q.question}")
        print(f"Ground Truth: {q.ground_truth}")
        print(f"Category: {q.category}")
        print(f"Evidence IDs: {q.evidence_ids}")
        pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
        print(f"Context length: {len(pcc.context_text)} chars")
        oracle_hit, missing = _evidence_present(pcc.context_text, turns, q.evidence_ids)
        print(f"Oracle Hit: {oracle_hit}, missing: {missing}")
        gt_words = [w.lower() for w in q.ground_truth.replace(',', ' ').replace('.', ' ').split() if len(w) > 2]
        found = [w for w in gt_words if w in pcc.context_text.lower()]
        print(f"GT words {gt_words} -> found in context: {found}")
        for evid in q.evidence_ids:
            t = next((t for t in turns if t.dia_id == evid), None)
            if t:
                print(f"  [Evidence Turn {evid}]: {t.speaker}: {t.text}")
                in_pcc = t.text.strip()[:40] in pcc.context_text
                print(f"    -> Turn in pcc: {in_pcc}")
