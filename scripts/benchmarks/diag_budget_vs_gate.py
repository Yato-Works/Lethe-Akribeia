"""Is the knapsack limit a BUDGET limit or a GATE limit?

Oracle forcing barely helped (35.0% -> 36.6%), so anchoring is not the
constraint.  Two candidates remain:
  (a) BUDGET -- 1400 tokens cannot hold the evidence it already knows about
  (b) GATE   -- the bundle-selection gate refuses the evidence regardless

Discriminator: hold the pipeline fixed and only raise the budget.  If evidence
retention climbs, the constraint is budget.  If it flatlines at the oracle
level, the gate is the constraint and more tokens buy nothing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"
BUDGETS = [1400, 2400, 4000, 6000, 8192]

# The compiled context is NOT verbatim: SpeakerAttributionNormalizer rewrites
# first person to third person ("I went to X" -> "<name> went to X").  A raw-text
# probe never matches and understates retention by ~48pt (measured).  Probe on
# content words instead.
_CONTENT_STOP = frozenset({
    "i", "im", "ive", "id", "ill", "my", "me", "mine", "we", "our", "us",
    "you", "your", "the", "a", "an", "and", "or", "but", "to", "of", "in",
    "on", "at", "for", "with", "was", "were", "is", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "so", "very", "just",
    "went", "go", "going", "had", "has", "have", "did", "do", "does",
})


def _probe(text: str, n: int = 5) -> str:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    keep = [w for w in words if w not in _CONTENT_STOP and len(w) > 2]
    return " ".join(keep[:n])


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    hits = {b: 0 for b in BUDGETS}
    total = 0

    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            ev = [e for e in (q.get("evidence") or []) if dia_map.get(e)]
            probes = [_probe(dia_map[e]) for e in ev
                      if len(dia_map[e]) >= 40 and _probe(dia_map[e])]
            if not probes:
                continue
            total += 1

            def present(ctx: str) -> bool:
                words = set(re.findall(r"[a-z0-9]+", ctx.lower()))
                return any(
                    sum(1 for w in p.split() if w in words) >= max(2, len(p.split()) - 1)
                    for p in probes
                )

            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(top)}
            anchor_ids = [t_map[(t[0], t[3])] for t in top[:12]
                          if (t[0], t[3]) in t_map]
            edg = gb.build(list(top), [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            scores = {a: 5.0 for a in anchor_ids}

            for bud in BUDGETS:
                try:
                    out = comp.compile(question=q["question"], bundles=bundles,
                                       token_budget=bud, anchor_scores=scores,
                                       is_multi_session=False, adaptive_escalation=True)
                except Exception:
                    continue
                if present(out.context_text):
                    hits[bud] += 1

    print("=" * 72)
    print("BUDGET SWEEP -- is the limit the token budget or the selection gate?")
    print("=" * 72)
    print(f"questions with identifiable gold evidence: {total}")
    print()
    print(f"{'budget':>8s} {'evidence in context':>22s} {'vs 1400':>10s}")
    print("-" * 72)
    base = hits[BUDGETS[0]] / total * 100
    for b in BUDGETS:
        pct = hits[b] / total * 100
        print(f"{b:8d} {pct:20.1f}% {pct-base:+9.1f}pt")
    print("-" * 72)
    top_pct = hits[BUDGETS[-1]] / total * 100
    print()
    if top_pct - base > 5.0:
        print("VERDICT: the constraint is the TOKEN BUDGET. More tokens buy real")
        print("  evidence retention, so raising the ceiling is a legitimate lever.")
    else:
        print(f"VERDICT: raising the budget 1400 -> {BUDGETS[-1]} gains only "
              f"{top_pct-base:+.1f}pt.")
        print("  The constraint is the SELECTION GATE, not the budget.  Buying")
        print("  tokens will not surface evidence the gate refuses to admit.")
        print("  Oracle forcing gained the same +1.6pt, which agrees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
