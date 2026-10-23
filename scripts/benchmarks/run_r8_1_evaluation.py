import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.compiler.operators.query_operator_compiler import QueryOperatorCompiler
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter

REPO = Path(__file__).resolve().parents[2]
r8_path = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R8.json"
dataset_path = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"

with open(r8_path, "r", encoding="utf-8") as f:
    r8_data = json.load(f)

with open(dataset_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

dev_map = {item["question_id"]: item for item in dev_data}
qoc = QueryOperatorCompiler()

updated_results = []
correct_count = 0
total_tokens = 0
overrides = 0

print("=" * 60)
print("Evaluating Lethe-Akribeia R8.1 (Query Operator Algebra)")
print("=" * 60)

for r in r8_data["results"]:
    qid = r["question_id"]
    qtype = r["question_type"]
    q = r["question"]
    gt = r["ground_truth"]

    # Check if newly handled by QueryOperatorCompiler
    if qid in dev_map:
        item = dev_map[qid]
        records = []
        for sid, sess, dt in zip(item["haystack_session_ids"], item["haystack_sessions"], item["haystack_dates"]):
            for m in sess:
                records.append(StructuredIR(
                    entity="dialogue",
                    property="message",
                    value=m["content"],
                    raw_content=f"[{sid} on {dt}] {m['role'].capitalize()}: {m['content']}",
                    time_scope=dt,
                    source=m["role"],
                    metadata={"session_id": sid}
                ))

        op_res = qoc.compile_and_execute(q, records)
        if op_res.has_result and op_res.direct_answer:
            # Verified deterministic override by Query Operator Algebra
            pred = op_res.direct_answer
            is_ok = bool(LongMemEvalAdapter.score_answer(
                question_type=qtype,
                gt=gt,
                predicted_answer=pred,
                is_abstention_gt=False,
                pcc_is_abstention=False,
            ))
            
            # Print if this rescued a previously failed item
            if not r.get("is_correct", False):
                print(f"🎉 RESCUED: [{qtype}] {qid}")
                print(f"   Q: {q[:70]}...")
                print(f"   GT: {gt}")
                print(f"   PRED: {pred}")
                print(f"   OPERATOR: {op_res.operator_name}")
                print(f"   IS_CORRECT: {is_ok}\n")

            r["predicted_answer"] = pred
            r["is_correct"] = is_ok
            r["deterministic_override"] = True
            r["prompt_tokens"] = 0  # 0-token deterministic resolution
            r["operator_name"] = op_res.operator_name

    updated_results.append(r)
    if r["is_correct"]:
        correct_count += 1
    total_tokens += r.get("prompt_tokens", 0)
    if r.get("deterministic_override"):
        overrides += 1

total_q = len(updated_results)
acc = correct_count / total_q * 100.0
mean_tok = total_tokens / total_q
drr = overrides / total_q * 100.0

by_type = {}
for r in updated_results:
    t = r["question_type"]
    by_type.setdefault(t, {"correct": 0, "total": 0})
    by_type[t]["total"] += 1
    if r["is_correct"]:
        by_type[t]["correct"] += 1

type_breakdown = {}
for t, st in by_type.items():
    type_breakdown[t] = {
        "correct": st["correct"],
        "total": st["total"],
        "accuracy": st["correct"] / st["total"] * 100.0,
    }

output_data = {
    "benchmark": "longmemeval_dev250",
    "split": "dev",
    "pipeline": "v4_budgeted_cognitive_compiler",
    "condition": "R8.1",
    "model": "qwen2.5:7b-instruct",
    "total_questions": total_q,
    "correct_answers": correct_count,
    "accuracy": acc,
    "mean_prompt_tokens": mean_tok,
    "deterministic_overrides": overrides,
    "deterministic_resolution_rate": drr,
    "elapsed_seconds": r8_data.get("elapsed_seconds", 0),
    "per_type_breakdown": type_breakdown,
    "results": updated_results,
}

out_full = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R8.1.json"
out_ckpt = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R8.1_ckpt.json"

with open(out_full, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

with open(out_ckpt, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("=" * 60)
print(f"R8.1 EVALUATION SUMMARY")
print("=" * 60)
print(f"Total Questions:          {total_q}")
print(f"Correct Answers:          {correct_count} / {total_q}")
print(f"Overall Accuracy:         {acc:.2f}%")
print(f"Mean Context Tokens:      {mean_tok:.1f} tokens")
print(f"Deterministic Overrides:  {overrides} / {total_q}")
print(f"Deterministic Res. Rate:  {drr:.2f}% (DRR)")
print("-" * 60)
print("Per-Type Breakdown:")
for t, st in by_type.items():
    print(f"  {t:26s}: {st['correct']:2d} / {st['total']:2d} ({st['correct']/st['total']*100:.2f}%)")
print("=" * 60)
print(f"Saved R8.1 results to {out_full}")
