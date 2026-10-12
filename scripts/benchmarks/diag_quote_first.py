"""Does a quote-first reader recover specific-name answers the current reader drops?

cat4's READER_FAILED block is dominated by specific named details the reader
abstains on even though the gold is in the context (cov 1.00):
  'What was Melanie's favorite book from her childhood?'  GT "Charlotte's Web"
  'What kind of pot did Mel and her kids make with clay?'  GT 'a cup with a dog face on it'
  'What activity did Caroline used to do with her dad?'   GT 'Horseback riding'

Hypothesis: a two-step reader (first quote the line that answers, then answer
from it) gives the model a concrete anchor instead of asking it to recall a
proper noun out of 1400 tokens.

Prediction discipline, BEFORE running:
  predicted  3 of 3 recovered
  mechanism  the gold is verbatim in the context at cov 1.00; the failure is
             retrieval-within-context, not absence
  if it fails abstention is not a context-search problem and quoting will not
             help, so the next lever is the reader itself, not the prompt
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from artificial_memory.research.benchmarks.llm import OllamaAnswerer  # noqa: E402

QUOTE_PROMPT = (
    'You are answering this question about a conversation:\n"{question}"\n\n'
    "STEP 1. Quote the single line from the evidence that answers it, "
    "verbatim, in quotation marks.\n"
    "STEP 2. On a new line, write ANSWER: followed by the short answer "
    "taken from that line.\n\n"
    "=== EVIDENCE ===\n{context}\n\nSTEP 1 QUOTE:\n"
)


def main() -> int:
    import json
    data = json.loads((ROOT / "datasets" / "external" / "locomo10.json").read_text(encoding="utf-8"))
    recorded = {r["question_id"]: r
                for r in json.loads((ROOT / "benchmark_results" /
                                     "locomo_full_20261012_211601.json").read_text(encoding="utf-8"))["results"]}

    targets = []
    for conv in data:
        for q in E.load_questions(conv):
            r = recorded.get(q["question_id"])
            if (r and r["category"] == 4 and not r["is_correct"]
                    and str(r["predicted_answer"]).strip().lower() == "i don't know."):
                targets.append((conv, q))
    print(f"cat4 abstentions with evidence available: {len(targets)}")
    print()

    answerer = OllamaAnswerer(num_ctx=8192)
    from eval_v4_pipeline_dev import V4CandidateRetriever
    from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
    from artificial_memory.compiler.graph.graph_builder import GraphBuilder
    from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler

    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    wins = 0
    for conv, q in targets[:6]:
        turns, _ = E.conversation_turns(conv["conversation"])
        top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
        plan = planner.plan(q["question"], [])
        t_map = {(t[0], t[3]): f"{t[0]}:t{k}" for k, t in enumerate(top)}
        anchors = [t_map[(t[0], t[3])] for t in top[:12] if (t[0], t[3]) in t_map]
        edg = gb.build(list(top), [])
        bundles = edg.form_bundles(anchors, max_bundle_tokens=150)
        budget = E._scaled_budget(plan.recommended_budget,
                                  sum(len(t[3].split()) * 4 // 3 for t in top))
        out = comp.compile(question=q["question"], bundles=bundles,
                           token_budget=budget,
                           anchor_scores={a: 5.0 for a in anchors},
                           is_multi_session=False, adaptive_escalation=True)

        base = answerer.answer(q["question"],
                               E.build_reader_prompt(q["question"], out.context_text)).text.strip()
        quoted = answerer.answer(
            q["question"], QUOTE_PROMPT.format(question=q["question"],
                                               context=out.context_text)).text.strip()

        ok_base = E.score(base, q["answer"])
        ok_q = E.score(quoted, q["answer"])
        wins += 1 if ok_q and not ok_base else 0
        print(f"Q   : {q['question'][:58]}")
        print(f"  GT    : {str(q['answer'])[:48]}")
        print(f"  base  : {base[:48]!r}  {'OK' if ok_base else '--'}")
        print(f"  quote : {quoted[:48]!r}  {'OK' if ok_q else '--'}")
        print()

    print("=" * 70)
    print(f"quote-first recovered {wins} of 6 that the current reader abstained on")
    print("PREDICTION: 3 of 3 recovered -> quote-first is a real lever" if wins >= 3
          else "PREDICTION MISSED: quote-first is not the lever; the reader itself is")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
