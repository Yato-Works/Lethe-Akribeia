import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import load_locomo, conversation_turns
from eval_v4_pipeline_dev import V4CandidateRetriever
from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory.compiler.graph.graph_builder import GraphBuilder

data = load_locomo(REPO / "datasets" / "external" / "locomo10.json")
turns, dia_map = conversation_turns(data[0]["conversation"])
q = "How long has Caroline had her current group of friends for?"
ret = V4CandidateRetriever()
cov = MultiSessionCoverageEngine()
gb = GraphBuilder()

top, ts, g = cov.retrieve_convergent_evidence(q, turns, ret)
cand = [t for t in turns if t[0] in {x[0] for x in top[:28]}]
t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(cand)}
a_ids = [t_map[(t[0], t[3])] for t in top if (t[0], t[3]) in t_map]
a_scores = {nid: ts.get((t[0], t[3]), 1.0) for t in top if (nid := t_map.get((t[0], t[3])))}

edg = gb.build(cand, [])
bundles = edg.form_bundles(a_ids, 150)

def _max_anc(b):
    return max((a_scores.get(n.node_id, 0.0) for n in b.member_nodes), default=0.0)

sorted_b = sorted([b for b in bundles if _max_anc(b) > 0.0], key=_max_anc, reverse=True)

gt = dia_map.get("D3:13")
print("Target D3:13 text:", gt)

for budget in [1000, 1200, 1400, 1600]:
    cur = 0
    sel = []
    for b in sorted_b[:25]:
        if cur + b.total_tokens <= budget:
            sel.append(b)
            cur += b.total_tokens
    in_sel = any(gt[:30] in n.raw_text for b in sel for n in b.member_nodes)
    print(f"Budget {budget}: tokens={cur}, bundles={len(sel)}, D3:13 in sel={in_sel}")
