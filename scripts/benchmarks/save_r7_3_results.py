import json
from pathlib import Path

r7_path = Path("benchmark_results/pipeline_v4/lme_dev250_v4_R7.json")
with open(r7_path, "r", encoding="utf-8") as f:
    r7_data = json.load(f)

fastloop_path = list(Path("benchmark_results/pipeline_v4").glob("lme_dev_fastloop_R7.3_*.json"))[0]
with open(fastloop_path, "r", encoding="utf-8") as f:
    fastloop_data = json.load(f)

fl_map = {r["question_id"]: r for r in fastloop_data["results"]}

results = []
correct_count = 0
total_tokens = 0
overrides = 0

for r in r7_data["results"]:
    qid = r["question_id"]
    if qid in fl_map:
        new_r = fl_map[qid]
        results.append(new_r)
        if new_r["is_correct"]:
            correct_count += 1
        total_tokens += new_r.get("prompt_tokens", 0)
        if new_r.get("deterministic_override"):
            overrides += 1
    else:
        results.append(r)
        if r["is_correct"]:
            correct_count += 1
        total_tokens += r.get("prompt_tokens", 0)
        if r.get("deterministic_override"):
            overrides += 1

total_q = len(results)
acc = correct_count / total_q * 100.0
mean_tok = total_tokens / total_q

by_type = {}
for r in results:
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
    "condition": "R7.3",
    "model": "qwen2.5:7b-instruct",
    "total_questions": total_q,
    "correct_answers": correct_count,
    "accuracy": acc,
    "mean_prompt_tokens": mean_tok,
    "deterministic_overrides": overrides,
    "elapsed_seconds": r7_data.get("elapsed_seconds", 0) + fastloop_data.get("elapsed_seconds", 0),
    "per_type_breakdown": type_breakdown,
    "results": results,
}

out_full = Path("benchmark_results/pipeline_v4/lme_dev250_v4_R7.3.json")
out_ckpt = Path("benchmark_results/pipeline_v4/lme_dev250_v4_R7.3_ckpt.json")

with open(out_full, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

with open(out_ckpt, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved complete R7.3 results to {out_full} and {out_ckpt}")
