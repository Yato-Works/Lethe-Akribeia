import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

# Let's inspect what was compiled for 0a995998
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "0a995998")

from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory_gen.compiler.samplers.per_session_sampler import PerSessionSampler
from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from eval_v4_pipeline_dev import V4CandidateRetriever

extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
retriever = V4CandidateRetriever()
graph_builder = GraphBuilder()
compiler = BudgetedCognitiveCompiler()
coprocessor = CoProcessorEngine()
sampler = PerSessionSampler(turns_per_session=3)
coverage_engine = MultiSessionCoverageEngine()

all_turns = []
all_records = []
sids = it.get("haystack_session_ids", [])
dates = it.get("haystack_dates", [])
for s_idx, session in enumerate(it.get("haystack_sessions", [])):
    sid = sids[s_idx]
    s_date = dates[s_idx]
    for turn in session:
        speaker = turn.get("role", "user")
        content = turn.get("content", "")
        header = f"[{sid} on {s_date}]"
        line = f"{header} {speaker.capitalize()}: {content}"
        all_turns.append((sid, s_date, speaker, content))
        for r in extractor.extract(content, default_source=speaker):
            r.raw_content = line
            r.time_scope = s_date
            all_records.append(r)

plan = planner.plan(it["question"], all_records, reference_date=it.get("question_date", ""))
top_turns, turn_scores, guaranteed_sids = coverage_engine.retrieve_convergent_evidence(
    it["question"], all_turns, retriever, reference_date=it.get("question_date", ""), base_top_k=60
)
top_sids = {t[0] for t in top_turns[:32]} | guaranteed_sids
candidate_turns = [t for t in all_turns if t[0] in top_sids]

sampled_turns = sampler.sample(it["question"], candidate_turns, max_total_tokens=2200)

# Check which gold turns are in sampled_turns
print("--- Checking Sampled Turns for Gold Sessions ---")
for t in sampled_turns:
    if t.session_id in it["answer_session_ids"]:
        if any(w in t.content.lower() for w in ["pick up", "return", "dry clean", "boots", "zara", "exchanged"]):
            print(f"[{t.session_id}] {t.speaker}: {t.content[:100]}")
