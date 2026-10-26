import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory_gen.compiler.coverage.concept_expander import AbstractConceptExpander
from eval_v4_pipeline_dev import V4CandidateRetriever

data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "0a995998")

print("Q:", it["question"])
print("GT:", it["answer"])
print("Gold SIDs:", it.get("answer_session_ids", []))

all_turns = []
sids = it.get("haystack_session_ids", [])
dates = it.get("haystack_dates", [])
for s_idx, session in enumerate(it.get("haystack_sessions", [])):
    sid = sids[s_idx]
    s_date = dates[s_idx]
    for turn in session:
        all_turns.append((sid, s_date, turn.get("role", "user"), turn.get("content", "")))

retriever = V4CandidateRetriever()
cov_engine = MultiSessionCoverageEngine()

top_turns, turn_scores, guaranteed_sids = cov_engine.retrieve_convergent_evidence(
    it["question"], all_turns, retriever, reference_date=it.get("question_date", "")
)

top_sids = {t[0] for t in top_turns[:32]} | guaranteed_sids
print("Retrieved Gold SIDs in top_sids:", [s for s in it["answer_session_ids"] if s in top_sids])
print("Missing Gold SIDs from top_sids:", [s for s in it["answer_session_ids"] if s not in top_sids])

for sid in it["answer_session_ids"]:
    sess_turns = [t for t in all_turns if t[0] == sid]
    print(f"\n--- Gold SID {sid} (in top_sids: {sid in top_sids}) ---")
    for t in sess_turns:
        if t[2] == "user":
            print(f"  USER: {t[3][:120]}")
