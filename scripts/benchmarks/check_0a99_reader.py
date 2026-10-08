import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_v4_pipeline_dev import build_prompt, FROZEN_MODEL, OllamaAnswerer
from artificial_memory.compiler.samplers.per_session_sampler import PerSessionSampler
from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from eval_v4_pipeline_dev import V4CandidateRetriever

data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "0a995998")

all_turns = []
for s_idx, session in enumerate(it['haystack_sessions']):
    sid = it['haystack_session_ids'][s_idx]
    s_date = it['haystack_dates'][s_idx]
    for turn in session:
        all_turns.append((sid, s_date, turn.get('role', 'user'), turn.get('content', '')))

cov_engine = MultiSessionCoverageEngine()
retriever = V4CandidateRetriever()
sampler = PerSessionSampler()

top_turns, turn_scores, guaranteed_sids = cov_engine.retrieve_convergent_evidence(
    it['question'], all_turns, retriever, base_top_k=60
)
top_sids = {t[0] for t in top_turns[:32]} | guaranteed_sids
candidate_turns = [t for t in all_turns if t[0] in top_sids]

sampled = sampler.sample(it['question'], candidate_turns, max_total_tokens=2200)

lines = []
for t in sampled:
    lines.append(f"[{t.session_id} on {t.session_date}] {t.speaker.capitalize()}: {t.content}")
ctx_str = "\n".join(lines)

from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
planner = CognitiveExecutionPlanner()
plan = planner.plan(it["question"], [], reference_date=it.get("question_date", ""))

prompt = build_prompt(
    cond="R6",
    plan=plan,
    is_aggregation=True,
    compiled_ctx_str=ctx_str,
    q=it["question"],
    q_date=it.get("question_date", "")
)

answerer = OllamaAnswerer(model=FROZEN_MODEL)
resp = answerer.answer(it["question"], prompt)
print("Question:", it["question"])
print("GT Answer:", it["answer"])
print("Reader Response:", resp.text)
