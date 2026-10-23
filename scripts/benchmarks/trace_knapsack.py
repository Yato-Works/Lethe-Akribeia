"""Live trace of the knapsack greedy for ONE case (3249768e, rank-1 anchor lost).

Reproduces compile()'s selection loop verbatim with verbose pick logging so
the dynamics (density race, budget fill, gate skips) are observable.
Run: python scripts/benchmarks/trace_knapsack.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

spec = importlib.util.spec_from_file_location(
    "ev", str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

QID = "3249768e"

items = json.loads((REPO / "datasets" / "external" / "longmemeval_s_cleaned.json")
                   .read_text(encoding="utf-8"))
item = {x["question_id"]: x for x in items}[QID]
q = item["question"]
q_date = item.get("question_date", "")

extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
retriever = ev.V4CandidateRetriever()
graph_builder = GraphBuilder()
coprocessor = CoProcessorEngine()

all_turns, all_records = [], []
dates = item.get("haystack_dates", [])
sids = item.get("haystack_session_ids", [])
for s_idx, session in enumerate(item.get("haystack_sessions", [])):
    s_date = dates[s_idx] if s_idx < len(dates) else ""
    sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
    for turn in session:
        speaker = turn.get("role", "user")
        content = turn.get("content", "")
        header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
        all_turns.append((sid, s_date, speaker, content))
        for rec in extractor.extract(content, default_source=speaker):
            rec.raw_content = f"{header} {speaker.capitalize()}: {content}"
            rec.time_scope = s_date
            all_records.append(rec)

plan = planner.plan(q, all_records, reference_date=q_date)
top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=35, reference_date=q_date)
top_sids = {t[0] for t in top_turns[:18]}
candidate_turns = [t for t in all_turns if t[0] in top_sids] or top_turns[:15]
token_budget = plan.recommended_budget

cand_sids = {c[0] for c in candidate_turns}
header_sid_re = re.compile(r"^\[([^\s\]]+)")
cand_records = [r for r in all_records
                if (m := header_sid_re.match(r.raw_content or ""))
                and m.group(1) in cand_sids]

t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(candidate_turns)}
anchor_ids, anchor_scores = [], {}
for t in top_turns:
    nid = t_map.get((t[0], t[3]))
    if nid and nid not in anchor_ids:
        anchor_ids.append(nid)
        anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)
for i, t in enumerate(candidate_turns):
    nid = f"{t[0]}:t{i}"
    if nid not in anchor_scores and (t[0], t[3]) in turn_scores:
        anchor_scores[nid] = turn_scores[(t[0], t[3])]

edg = graph_builder.build(candidate_turns, cand_records)
bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
cert_tokens = (len(bundle.assembled_certificate_block.split()) * 4 // 3
               if bundle.assembled_certificate_block else 0)
eff = max(150, token_budget - cert_tokens)
print(f"Q: {q}")
print(f"budget={token_budget} cert={cert_tokens} effective={eff} bundles={len(bundles)}")

# ---- verbatim copy of compile()'s greedy loop (is_multi=False branch), logged ----
expansions = UniversalEvidenceScorer.SEMANTIC_EXPANSIONS
stop_words = (UniversalEvidenceScorer.STOP_WORDS | {
    "can", "could", "would", "should", "some", "any", "suggest", "suggestion", "suggestions",
    "recommend", "recommendation", "recommendations", "advice", "tips", "ideas", "thoughts",
    "opinion", "what", "where", "when", "which", "who", "whom", "how", "tell", "help", "give",
    "think", "wondering", "please", "look", "looking", "interested"})
raw_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", q.lower())) - stop_words
if not raw_words:
    raw_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", q.lower()))
q_words = set(raw_words)
for w in raw_words:
    for suffix in ["ing", "ed", "es", "s"]:
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            q_words.add(w[:-len(suffix)])
    if w in expansions:
        q_words.update(expansions[w])

valid_candidates = [b for b in bundles if b.total_tokens <= eff]
oversized = [(b.bundle_id, b.total_tokens) for b in bundles if b.total_tokens > eff]
print(f"valid={len(valid_candidates)} excluded_oversized={oversized}")
covered_words: set = set()
selected_sids: set = set()
current_tokens = 0
picked = 0
while valid_candidates:
    best_bundle, best_density, best_gain = None, -1.0, 0.0
    for b in valid_candidates:
        if current_tokens + b.total_tokens > eff:
            continue
        b_words, b_sids = set(), set()
        for n in b.member_nodes:
            b_words.update(re.findall(r"\b[a-zA-Z0-9_-]+\b", n.raw_text.lower()))
            b_sids.add(n.session_id)
        overlap = b_words & q_words
        in_sel = bool(b_sids & selected_sids)
        anc = max([anchor_scores.get(n.node_id, 0.0) for n in b.member_nodes] + [0.0])
        if not overlap and not in_sel and anc <= 0.0:
            continue
        new_words = overlap - covered_words
        if not in_sel and not new_words and anc <= 0.0:
            continue
        gain = anc * 2.5 + len(new_words) * 8.0 + len(overlap) * 2.0
        gain += 6.0 if in_sel else 0.0  # single-session coherence (is_multi=False)
        density = gain / (max(80, b.total_tokens) ** 0.15)
        if density > best_density:
            best_density, best_bundle, best_gain = density, b, gain
    if not best_bundle or best_gain <= 0.0:
        print("STOP: no positive-gain bundle fits")
        break
    picked += 1
    print(f"pick#{picked}: {best_bundle.bundle_id} tok={best_bundle.total_tokens} "
          f"gain={best_gain:.1f} density={best_density:.2f} "
          f"room_after={eff - current_tokens - best_bundle.total_tokens}")
    current_tokens += best_bundle.total_tokens
    for n in best_bundle.member_nodes:
        covered_words.update(re.findall(r"\b[a-zA-Z0-9_-]+\b", n.raw_text.lower()))
        selected_sids.add(n.session_id)
    valid_candidates = [b for b in valid_candidates if b.bundle_id != best_bundle.bundle_id]
selected_ids = set()
# re-run capture: collect picks in a list
print(f"done: picked={picked} tokens={current_tokens}/{eff}")


