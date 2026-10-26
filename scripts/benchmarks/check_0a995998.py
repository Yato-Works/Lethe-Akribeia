import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.coverage.concept_expander import AbstractConceptExpander
from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
import importlib.util
spec = importlib.util.spec_from_file_location("eval_v4", str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"))
eval_v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_v4)
retriever = eval_v4.V4CandidateRetriever()

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = {it["question_id"]: it for it in json.load(f)}

it = all_items["0a995998"]
q = it["question"]
all_turns = []
for s_idx, sess in enumerate(it["haystack_sessions"]):
    sid = it["haystack_session_ids"][s_idx]
    s_date = it["haystack_dates"][s_idx]
    for turn in sess:
        all_turns.append((sid, s_date, turn.get("role"), turn.get("content")))

subs = AbstractConceptExpander.generate_subqueries(q)
print("Subqueries generated:", len(subs), subs)

engine = MultiSessionCoverageEngine()
cov_turns, cov_scores, guaranteed_sids = engine.retrieve_convergent_evidence(q, all_turns, retriever)
print("Guaranteed SIDs:", guaranteed_sids)
target_sid = "answer_afa9873b_2"
print(f"Target {target_sid} in guaranteed?", target_sid in guaranteed_sids)
print(f"Target {target_sid} in top 35 turns?", target_sid in {t[0] for t in cov_turns[:35]})
