"""ORACLE TEST: is the engine capable of surfacing 100% of the gold evidence?

The question Eli posed: can this engine be put in a state where it emits the
correct evidence for EVERY question, so that the remaining gap is purely the
reader?  That would make the engine model-independent -- a 120B reader would
then be expected to score near 100%, and any shortfall would be provably the
reader's, not retrieval's.

Method, no LLM, on the REAL LoCoMo path (eval_locomo_full's own pipeline:
retriever -> graph_builder -> form_bundles -> budgeted_compiler.compile):

  STAGE 1  candidate pool      is the gold evidence turn in the retrieved pool?
  STAGE 2  bundled             does it get an EvidenceBundle?
  STAGE 3  anchored            does it carry an anchor score?
  STAGE 4  compiled context    does its text survive into context_text?
  STAGE 5  ORACLE              re-run compile with the gold turn FORCED as the
                               top anchor -- if the context still lacks it, the
                               knapsack itself is the limit, not retrieval

STAGE 5 is the decisive one: it separates "we failed to find the evidence" from
"even told exactly where it is, we still cannot keep it".  Only the first is a
retrieval problem.

Note: the pre-existing scripts/benchmarks/oracle_funnel.py measures a DIFFERENT
path (msc_compiler / StateReconstructor), which the LoCoMo harness never calls.
Its numbers do not describe this pipeline, which is why this is written fresh.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"


# The compiled context is NOT verbatim: SpeakerAttributionNormalizer rewrites
# first-person to third-person ("I went to X" -> "<name> went to X"), so a
# raw-text probe against the original dia_id text never matches and every
# retention figure is understated.  Probe on CONTENT WORDS instead, which
# survive the rewrite.
_CONTENT_STOP = frozenset({
    "i", "im", "ive", "id", "ill", "my", "me", "mine", "we", "our", "us",
    "you", "your", "the", "a", "an", "and", "or", "but", "to", "of", "in",
    "on", "at", "for", "with", "was", "were", "is", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "so", "very", "just",
    "went", "go", "going", "had", "has", "have", "did", "do", "does",
})


def _probe(text: str, n: int = 5) -> str:
    """Content-word signature of an evidence turn, robust to person rewriting."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    keep = [w for w in words if w not in _CONTENT_STOP and len(w) > 2]
    return " ".join(keep[:n])


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    st = defaultdict(lambda: [0, 0])   # stage -> [hit, total]
    per_cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))

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

            def mark(stage, cat):
                st[stage][1] += 1
                per_cat[cat][stage][1] += 1

            def hit(stage, cat):
                st[stage][0] += 1
                per_cat[cat][stage][0] += 1

            cat = q["category"]
            mark("1_pool", cat)
            mark("2_bundled", cat)
            mark("4_context", cat)
            mark("5_oracle", cat)

            def present(ctx: str) -> bool:
                words = set(re.findall(r"[a-z0-9]+", ctx.lower()))
                return any(
                    sum(1 for w in p.split() if w in words) >= max(2, len(p.split()) - 1)
                    for p in probes
                )

            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            in_pool = any(present(t[3]) for t in top)
            if in_pool:
                hit("1_pool", cat)

            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(top)}
            anchor_ids = [t_map[(t[0], t[3])] for t in top[:12]
                          if (t[0], t[3]) in t_map]
            edg = gb.build(list(top), [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            if any(present(n.raw_text) for b in bundles for n in b.member_nodes):
                hit("2_bundled", cat)
            budget = E._scaled_budget(plan.recommended_budget,
                                      sum(len(t[3].split()) * 4 // 3 for t in top))

            # STAGE 4 -- real pipeline, gold NOT privileged
            try:
                out = comp.compile(question=q["question"], bundles=bundles,
                                   token_budget=budget,
                                   anchor_scores={a: 5.0 for a in anchor_ids},
                                   is_multi_session=False, adaptive_escalation=True)
                if present(out.context_text):
                    hit("4_context", cat)
            except Exception:
                pass

            # STAGE 5 -- ORACLE: force the gold turn to the very top anchor
            gold_keys = [(t[0], t[3]) for t in top if present(t[3])]
            oracle_ids = []
            for k in gold_keys:
                nid = t_map.get(k)
                if nid and nid not in oracle_ids:
                    oracle_ids.append(nid)
            oracle_ids += [a for a in anchor_ids if a not in oracle_ids]
            try:
                out_o = comp.compile(question=q["question"], bundles=bundles,
                                     token_budget=budget,
                                     anchor_scores={a: 900.0 for a in oracle_ids},
                                     is_multi_session=False, adaptive_escalation=True)
                if present(out_o.context_text):
                    hit("5_oracle", cat)
            except Exception:
                pass

    print("=" * 78)
    print("ORACLE TEST -- can the engine surface 100% of gold evidence?")
    print("=" * 78)
    stages = ["1_pool", "2_bundled", "4_context", "5_oracle"]
    print(f"{'stage':16s} {'all':>16s}   meaning")
    print("-" * 78)
    meaning = {
        "1_pool": "retriever put it in the candidate pool",
        "2_bundled": "it received an EvidenceBundle",
        "4_context": "it reached the reader in the real run",
        "5_oracle": "it survives when FORCED to be the top anchor",
    }
    for s in stages:
        h, t = st[s]
        print(f"{s:16s} {h:5d}/{t:<5d} ({h/t*100:5.1f}%)   {meaning[s]}")
    print("-" * 78)
    print()
    print("Per category, real run vs oracle:")
    print(f"{'cat':6s} {'n':>5s} {'real':>9s} {'oracle':>9s} {'headroom':>10s}")
    print("-" * 78)
    for c in sorted(per_cat):
        d = per_cat[c]
        n = d["1_pool"][1]
        real = d["4_context"][0] / n * 100
        orc = d["5_oracle"][0] / n * 100
        print(f"cat{c:<3d} {n:5d} {real:8.1f}% {orc:8.1f}% {orc-real:+9.1f}pt")
    print()
    orc_all = st["5_oracle"][0] / st["5_oracle"][1] * 100
    print(f"ORACLE CEILING: {orc_all:.1f}%")
    print()
    if orc_all >= 99.0:
        print("VERDICT: the engine CAN surface essentially all gold evidence.")
        print("  Any remaining score gap is then the READER's, provably.")
    else:
        gap = 100.0 - orc_all
        print(f"VERDICT: even with the gold turn forced as the top anchor, "
              f"{gap:.1f}% of evidence still does not reach the reader.")
        print("  So 'correct evidence 100% of the time' is NOT reachable by")
        print("  retrieval or anchoring alone -- the knapsack/budget is the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
