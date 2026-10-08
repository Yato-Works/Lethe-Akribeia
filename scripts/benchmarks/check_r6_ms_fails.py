import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
path = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R6.json"
data = json.loads(path.read_text(encoding="utf-8"))

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = {it["question_id"]: it for it in json.load(f)}

fails = [r for r in data["results"] if r["question_type"] == "multi-session" and not r["is_correct"]]
print(f"Total multi-session fails: {len(fails)}")

for i, r in enumerate(fails):
    qid = r["question_id"]
    item = all_items.get(qid, {})
    q = item.get("question", "")
    ans = r.get("ground_truth", "")
    pred = r.get("predicted", "")[:60].replace("\n", " ")
    gold_sids = item.get("answer_session_ids", [])
    print(f"[{i+1:02d}] {qid} | GT: {ans} | Pred: {pred}")
    print(f"     Q: {q}")
    print(f"     Gold SIDs: {gold_sids}")
