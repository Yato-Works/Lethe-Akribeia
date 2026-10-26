import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.pipeline.budgeted_compilation import BudgetedCompilationPipeline

pipeline = BudgetedCompilationPipeline()
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))

target_qids = [
    "gpt4_68e94287", "gpt4_88806d6e", "gpt4_93f6379c", "gpt4_45189cb4", "gpt4_2f56ae70",
    "gpt4_1e4a8aec", "gpt4_4929293b", "eac54add", "6e984302", "982b5123",
    "gpt4_1d80365e", "2ebe6c90", "a3045048", "0bc8ad93"
]

correct = 0
total = len(target_qids)

print(f"Evaluating {total} previously failed Temporal Reasoning questions...")
for qid in target_qids:
    it = next(item for item in data if item["question_id"] == qid)
    res = pipeline.run(it)
    pred = res.response
    gt = it["answer"]
    
    # Simple check against GT
    from artificial_memory.evaluation.eval_harness import evaluate_answer
    eval_res = evaluate_answer(it["question"], pred, gt)
    is_correct = eval_res.get("correct", False)
    if is_correct:
        correct += 1
        tag = "[CORRECT]"
    else:
        tag = "[WRONG]"
        
    print(f"{tag} [{qid}] Q: {it['question'][:60]}...")
    print(f"       GT:   {gt[:80]}")
    print(f"       Pred: {pred[:80]}")
    if res.metrics.get("deterministic_override"):
        print("       (Resolved via 0-token Deterministic Override!)")

print("=" * 60)
print(f"Rescued: {correct} / {total} ({correct / total * 100:.1f}%)")
