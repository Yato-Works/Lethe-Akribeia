import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
res = json.load(open(REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R6.json", encoding="utf-8"))

fails = [
    r for r in res["results"]
    if r["question_type"] == "temporal-reasoning" and not r["is_correct"]
]

print(f"Total Temporal Failures: {len(fails)} / 66\n" + "=" * 80)
for i, f in enumerate(fails, 1):
    qid = f["question_id"]
    q = f["question"]
    pred = f["predicted_answer"]
    gt = f["ground_truth"]
    tok = f["prompt_tokens"]
    print(f"[{i:02d}] {qid} | tok={tok}")
    print(f"  Q:    {q}")
    print(f"  Pred: {pred!r}")
    print(f"  GT:   {gt!r}")
    print()
