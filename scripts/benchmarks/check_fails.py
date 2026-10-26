import json

with open("benchmark_results/pipeline_v4/lme_dev250_v4_pipeline.json", encoding="utf-8") as f:
    d = json.load(f)

for r in d["results"]:
    if not r["is_correct"]:
        print("="*60)
        print(f"QID: {r['question_id']}")
        print(f"Q:  {r['question']}")
        print(f"GT: {r['ground_truth']}")
        print(f"PR: {r['predicted_answer']}")
