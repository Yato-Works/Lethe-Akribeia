"""Why does compile drop evidence that is already in the candidate pool?

21.9% of evidence present in the retrieved pool does not reach the context.
Diagnose the bundles that lose: were they even considered, and if not, which
gate rejected them.  Three candidate gates in budgeted_compiler.compile:

  G0 valid_candidates  total_tokens > effective_budget
  G1 admission         no query overlap AND not in a selected session
                       AND anchor score <= 0
  G2 zero-new-words    contributes no new query words, no anchor score,
                       not continuing a selected session

A bundle that passes G1 but loses on DENSITY is a different problem from one
rejected by a gate: the first means the ranking is wrong, the second means the
gate is too strict.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
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

STOP = frozenset({
    "i", "im", "ive", "id", "ill", "my", "me", "mine", "we", "our", "us",
    "you", "your", "the", "a", "an", "and", "or", "but", "to", "of", "in",
    "on", "at", "for", "with", "was", "were", "is", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "so", "very", "just",
    "went", "go", "going", "had", "has", "have", "did", "do", "does",
})


def probe(text: str, n: int = 5) -> str:
    w = [x for x in re.findall(r"[a-z0-9]+", str(text).lower())
         if x not in STOP and len(x) > 2]
    return " ".join(w[:n])


def present(ctx: str, probes) -> bool:
    words = set(re.findall(r"[a-z0-9]+", ctx.lower()))
    return any(sum(1 for w in p.split() if w in words) >= max(2, len(p.split()) - 1)
               for p in probes)


def _bundle_in_context(bundle, context: str) -> bool:
    """Is this bundle's primary content represented in the context?

    Content-word based, because SpeakerAttributionNormalizer rewrites
    first person to third person, so verbatim matching understates by ~48pt.
    """
    text = " ".join(n.raw_text for n in bundle.member_nodes)
    return present(context, [probe(text)])


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    why = Counter()
    total_in_pool = 0
    total_lost = 0
    examples = []

    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            ev = [e for e in (q.get("evidence") or []) if dia_map.get(e)]
            probes = [probe(dia_map[e]) for e in ev
                      if len(dia_map[e]) >= 40 and probe(dia_map[e])]
            if not probes:
                continue

            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            gold_turns = [t for t in top if present(t[3], probes)]
            if not gold_turns:
                continue
            total_in_pool += len(gold_turns)

            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(top)}
            anchor_ids = [t_map[(t[0], t[3])] for t in top[:12]
                          if (t[0], t[3]) in t_map]
            edg = gb.build(list(top), [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            budget = E._scaled_budget(plan.recommended_budget,
                                      sum(len(t[3].split()) * 4 // 3 for t in top))
            try:
                out = comp.compile(question=q["question"], bundles=bundles,
                                   token_budget=budget,
                                   anchor_scores={a: 5.0 for a in anchor_ids},
                                   is_multi_session=False, adaptive_escalation=True)
            except Exception:
                continue

            for t in gold_turns:
                # Is THIS turn's own bundle represented in the context?
                # Compare content words, since the context is person-rewritten.
                b = next((x for x in bundles
                          if any(n.raw_text == t[3] for n in x.member_nodes)), None)
                if b is not None and _bundle_in_context(b, out.context_text):
                    continue
                total_lost += 1

                if b is None:
                    why["no_bundle"] += 1
                    continue
                if b.total_tokens > budget:
                    why["G0_over_budget"] += 1
                    continue
                anc = max((5.0 for n in b.member_nodes
                           if n.node_id in anchor_ids), default=0.0)
                b_words = set(re.findall(r"[a-z0-9]+", " ".join(
                    n.raw_text.lower() for n in b.member_nodes)))
                q_words = set(re.findall(r"[a-z0-9]+", q["question"].lower()))
                has_overlap = bool(b_words & q_words)
                if not has_overlap and anc <= 0.0:
                    why["G1_no_overlap_no_anchor"] += 1
                elif anc <= 0.0 and not has_overlap:
                    why["G2_zero_new_words"] += 1
                else:
                    why["passed_gate_lost_on_density"] += 1
                    if len(examples) < 4:
                        examples.append((q["question"], t[3][:60],
                                         b.total_tokens, anc, has_overlap))

    print("=" * 74)
    print("WHY DOES COMPILE DROP EVIDENCE THAT WAS IN THE POOL?")
    print("=" * 74)
    print(f"gold turns present in candidate pool : {total_in_pool}")
    print(f"  of those, dropped by compile       : {total_lost} "
          f"({total_lost/total_in_pool*100:.1f}%)")
    print()
    print("reason the losing bundle was not selected:")
    for reason, c in why.most_common():
        print(f"  {reason:34s} {c:5d}  ({c/total_lost*100:5.1f}%)")
    print()
    if examples:
        print("examples that PASSED the gates and still lost (density problem):")
        for qq, txt, tok, anc, ov in examples:
            print(f"  Q   : {qq[:56]}")
            print(f"  turn: {txt}")
            print(f"  tokens={tok} anchor={anc} query_overlap={ov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
