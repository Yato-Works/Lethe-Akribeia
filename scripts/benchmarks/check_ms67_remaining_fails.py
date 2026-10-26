import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

res_file = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev_ms67_R6_coverage.json"
data = json.loads(res_file.read_text(encoding="utf-8"))

dataset = {it["question_id"]: it for it in json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))}

fails = [r for r in data["results"] if not r["is_correct"]]
print(f"Total remaining fails: {len(fails)} / {len(data['results'])}\n")

for i, r in enumerate(fails):
    qid = r["question_id"]
    item = dataset.get(qid, {})
    q = item.get("question", "")
    ans = r.get("ground_truth", "")
    pred = r.get("predicted", "")
    print(f"[{i+1:02d}] {qid} | GT: {ans!r} | Pred: {pred!r}")
    print(f"     Q: {q}")
    # Check if gold session ids are in compiled context
    gold_sids = item.get("answer_session_ids", [])
    ctx = r.get("compiled_context", "")
    # Read context from cache if available
    ctx_cache = REPO / "cache" / "v4_contexts" / "R6" / f"{qid}.txt"
    if ctx_cache.exists():
        ctx = ctx_cache.read_text(encoding="utf-8")
    sids_in_ctx = [sid for sid in gold_sids if sid in ctx]
    print(f"     Gold SIDs in ctx: {len(sids_in_ctx)}/{len(gold_sids)} ({sids_in_ctx})")
    print()
