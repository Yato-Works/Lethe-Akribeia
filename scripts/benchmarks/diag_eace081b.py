"""Deep-dive eace081b: why does compiled context drop the Oahu session?

Run: python scripts/benchmarks/diag_eace081b.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.compiler.graph.graph_builder import GraphBuilder
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

spec = importlib.util.spec_from_file_location(
    "eval_v4_pipeline_dev",
    str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
retriever = mod.V4CandidateRetriever()

extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
graph_builder = GraphBuilder()
compiler = BudgetedCognitiveCompiler()
coprocessor = CoProcessorEngine()

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = json.load(f)

item = next(x for x in all_items if x["question_id"] == "eace081b")
q = item["question"]
q_date = item.get("question_date", "")
print(f"Q: {q} (date={q_date}) GT={item.get('answer')}")

all_turns = []
all_records = []
dates = item.get("haystack_dates", [])
sids = item.get("haystack_session_ids", [])
for s_idx, session in enumerate(item.get("haystack_sessions", [])):
    s_date = dates[s_idx] if s_idx < len(dates) else ""
    sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
    for turn in session:
        speaker = turn.get("role", "user")
        content = turn.get("content", "")
        header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
        line = f"{header} {speaker.capitalize()}: {content}"
        all_turns.append((sid, s_date, speaker, content))
        for r in extractor.extract(content, default_source=speaker):
            r.raw_content = line
            r.time_scope = s_date
            all_records.append(r)

plan = planner.plan(q, all_records, reference_date=q_date)
print(f"plan: mode={plan.dominant_mode} budget={plan.recommended_budget}")

top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=35, reference_date=q_date)
print("\n-- top 15 ranked turns --")
for i, t in enumerate(top_turns[:15]):
    print(f"  #{i+1} {t[0]} ({t[2]}) score={turn_scores.get((t[0], t[3]), 0):.2f}: {t[3][:110]}")

top_sids = {t[0] for t in top_turns[:18]}
candidate_turns = [t for t in all_turns if t[0] in top_sids]
if not candidate_turns:
    candidate_turns = top_turns[:15]

# turns from the Oahu session
print("\n-- answer_8a791264_2 turns (source session) --")
for t in all_turns:
    if t[0] == "answer_8a791264_2":
        in_cand = t in candidate_turns
        rank = next((i + 1 for i, x in enumerate(top_turns) if x == t), None)
        score = turn_scores.get((t[0], t[3]), 0)
        print(f"  cand={in_cand} rank={rank} score={score:.2f} ({t[2]}): {t[3][:140]}")
        if "oahu" in t[3].lower():
            print("      ^^^ CONTAINS OAHU")

cand_sids = {c[0] for c in candidate_turns}
header_sid_re = re.compile(r"^\[([^\s\]]+)")
cand_records = [r for r in all_records
                if (m := header_sid_re.match(r.raw_content or "")) and m.group(1) in cand_sids]
print(f"\ncandidate sessions: {sorted(cand_sids)}  cand_records={len(cand_records)}")

t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(candidate_turns)}
anchor_ids, anchor_scores = [], {}
for t in top_turns:
    nid = t_map.get((t[0], t[3]))
    if nid and nid not in anchor_ids:
        anchor_ids.append(nid)
        anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)
print(f"anchors ({len(anchor_ids)}): {anchor_ids}")

bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
edg = graph_builder.build(candidate_turns, cand_records)
print(f"\nEDG nodes: {len(edg.nodes)}")
for nid, node in list(edg.nodes.items())[:40] if isinstance(edg.nodes, dict) else []:
    pass

bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
print(f"bundles formed: {len(bundles)}")
for i, b in enumerate(bundles):
    text = getattr(b, "text", "") or ""
    print(f"  bundle#{i} nodes={getattr(b, 'node_ids', None)} score={getattr(b, 'score', None)} len={len(text.split())}w")
    print(f"    {text[:150].replace(chr(10), ' | ')}")

is_multi = bool(plan.requires_event_aggregation or plan.aggregation_demand >= 0.50 or plan.temporal_demand >= 0.85)
budget = plan.recommended_budget
compiled = compiler.compile(question=q, bundles=bundles, token_budget=budget,
                            co_processor_bundle=bundle, anchor_scores=anchor_scores,
                            is_multi_session=is_multi)
ctx = compiled.context_text
print(f"\n-- compiled context ({len(ctx.split())}w, budget={budget}) --")
print(ctx)
print("\nOAHU in context:", "oahu" in ctx.lower())
