"""What can the deterministic committer NOT answer?  Classify the abstentions.

The committer claims 76.7% of cat2 on the real pipeline context.  Before
broadening engine coverage, find out what shapes it refuses and why -- a refusal
is either a question shape no engine owns (broadenable) or an evidence problem
(not broadenable).  Only the first justifies new engine work.

Prediction discipline, BEFORE running:
  predicted  refusals split roughly 60% question-shape / 40% evidence
  mechanism  commit_answer returns a `detail` string for every refusal, so the
             distribution of those strings is the answer directly
  if it fails most refusals are evidence-side, then engine work is the wrong
             lever and the next investment is retrieval/memory structure
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
from artificial_memory_gen.skills.answer_committer import commit_answer  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    reasons = Counter()
    examples = {}
    n = 0
    claimed = 0

    for conv in data:
        turns, _ = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{k}" for k, t in enumerate(top)}
            anchors = [t_map[(t[0], t[3])] for t in top[:12]
                       if (t[0], t[3]) in t_map]
            edg = gb.build(list(top), [])
            bundles = edg.form_bundles(anchors, max_bundle_tokens=150)
            budget = E._scaled_budget(plan.recommended_budget,
                                      sum(len(t[3].split()) * 4 // 3 for t in top))
            try:
                out = comp.compile(question=q["question"], bundles=bundles,
                                   token_budget=budget,
                                   anchor_scores={a: 5.0 for a in anchors},
                                   is_multi_session=False, adaptive_escalation=True)
                ctx = out.context_text
            except Exception:
                continue

            n += 1
            c = commit_answer(q["question"], ctx, q["category"])
            if c.used:
                claimed += 1
                continue
            detail = str(getattr(c, "detail", "") or "")
            reason = detail.split("(")[0].strip()[:60] or "(no detail)"
            reasons[reason] += 1
            if reason not in examples:
                examples[reason] = (q["question"], str(q["answer"]), detail[:70])

    print("=" * 78)
    print(f"COMMITTER COVERAGE over {n} questions (real pipeline context)")
    print("=" * 78)
    print(f"  claimed    : {claimed} ({claimed/n*100:.1f}%)")
    print(f"  abstained  : {n-claimed} ({(n-claimed)/n*100:.1f}%)")
    print()
    print("abstention reasons:")
    print(f"{'reason':46s} {'n':>5s} {'share':>7s}")
    print("-" * 78)
    for reason, cnt in reasons.most_common(18):
        print(f"{reason:46s} {cnt:5d} {cnt/max(1,n-claimed)*100:6.1f}%")
        qq, gt, det = examples[reason]
        print(f"    e.g. {qq[:56]}")
        print(f"         GT {gt[:40]}  | detail: {det}")
    print()
    print("PREDICTION CHECK (stated before running):")
    print("  predicted 60% question-shape / 40% evidence")
    print(f"  actual   {claimed/max(1,n)*100:.0f}% claimed overall")
    print()
    shape_like = sum(c for r, c in reasons.items()
                     if any(k in r.lower() for k in
                            ("inference", "no deterministic", "shape", "open")))
    print(f"  question-shape refusals : {shape_like} "
          f"({shape_like/max(1,n-claimed)*100:.1f}%)  <- broadenable")
    print(f"  other refusals          : {n-claimed-shape_like}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
