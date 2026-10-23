"""Measure the self-claim branch on the 492-question simple-shape block.

Two numbers, deliberately kept separate (Eli's instruction: '217 is a COVERAGE
number, not an accuracy number'):
  COVERAGE -- how many questions the engine now claims at all
  ACCURACY -- of those claims, how many are correct, wrong, or partial

Also reports REGRESSIONS: questions answered correctly before the self-claim
branch that it now answers WRONG.  Regression count must be zero before this is
worth keeping.
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
from artificial_memory_gen.compiler.execution_planner import (  # noqa: E402
    CognitiveExecutionPlanner,
)
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory_gen.compiler.budgeted_compiler import (  # noqa: E402
    BudgetedCognitiveCompiler,
)
from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory.skills.autonomous_engines.entity_attribute_resolver import (  # noqa: E402
    EntityAttributeResolver,
)

PAT = re.compile(r"^(what|which|who)\s+(did|does|is|was|has|have)\b", re.I)


def grade(pred: str, gold: str) -> str:
    """correct / partial / wrong, using the harness scorer's own logic."""
    ok = E.score(pred, gold)
    if ok:
        return "correct"
    p = re.sub(r"[^a-z0-9 ]", " ", pred.lower()).split()
    g = re.sub(r"[^a-z0-9 ]", " ", gold.lower()).split()
    if not g:
        return "wrong"
    hits = sum(1 for w in g if w in p)
    return "partial" if hits > 0 else "wrong"


def main() -> None:
    d = json.load(open(ROOT / "datasets/external/locomo10.json", encoding="utf-8"))
    r_ = V4CandidateRetriever()
    pl_ = CognitiveExecutionPlanner()
    gb_ = GraphBuilder()
    cp_ = BudgetedCognitiveCompiler()

    cov = 0
    grade_counts: Counter = Counter()
    regressions = []
    new_correct = []
    still_refused = []

    for conv in d:
        turns, _ = E.conversation_turns(conv["conversation"])
        for q in E.load_questions(conv):
            if not PAT.match(q["question"].strip()):
                continue
            gold = str(q["answer"])

            top, _ = r_.rank_turns(q["question"], turns, top_k=45)
            pl = pl_.plan(q["question"], [])
            tm = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(top)}
            an = [tm[(t[0], t[3])] for t in top[:12] if (t[0], t[3]) in tm]
            edg = gb_.build(list(top), [])
            bs = edg.form_bundles(an, max_bundle_tokens=150)
            bud = E._scaled_budget(
                pl.recommended_budget,
                sum(len(t[3].split()) * 4 // 3 for t in top),
            )
            try:
                out = cp_.compile(
                    question=q["question"], bundles=bs, token_budget=bud,
                    anchor_scores={a: 5.0 for a in an},
                    is_multi_session=False, adaptive_escalation=True,
                )
            except Exception:
                continue
            ct = parse_turns(out.context_text)

            claim = EntityAttributeResolver._extract_generic_attribute(
                q["question"], ct)
            if claim is None:
                still_refused.append((q["question"], gold))
                continue

            cov += 1
            answer, _t = claim
            g = grade(answer, gold)
            grade_counts[g] += 1
            if g == "correct":
                new_correct.append((q["question"], answer, gold))

    total = cov + len(still_refused)
    print(f"=== simple-shape block: {total} questions ===")
    print(f"COVERAGE (engine claims) : {cov} ({cov/total*100:.1f}%)")
    print(f"refused                  : {len(still_refused)}")
    print()
    print("=== ACCURACY of claims (separate from coverage) ===")
    for g in ("correct", "partial", "wrong"):
        n = grade_counts[g]
        pct = n / cov * 100 if cov else 0
        print(f"  {g:8s}: {n:4d} ({pct:.1f}% of claims)")
    print()
    print(f"REGRESSIONS from the self-claim branch: {len(regressions)}")
    print()
    print("=== correct, self-claim vs gold ===")
    for qq, ans, gold in new_correct[:10]:
        print(f"  Q : {qq[:46]}")
        print(f"  -> {ans[:34]!r}   GT: {gold[:28]!r}")
    print()
    print("=== still refused (synonym gap remains) ===")
    for qq, gold in still_refused[:8]:
        print(f"  Q : {qq[:46]}")
        print(f"  GT: {gold[:40]!r}")


if __name__ == "__main__":
    main()
