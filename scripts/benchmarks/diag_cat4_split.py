"""Is cat4's failure 'evidence missing' or 'cannot state the specific detail'?

cat4 has 290 failures and only 1.7% are temporal, so the date work does not
apply.  The observed shape is a specific-name recall problem: GT "Charlotte's
Web" vs PRED "I don't know."; GT "a cup with a dog face on it" vs "I don't
know.".

Splits cat4 failures by whether the gold's content words are actually IN the
compiled context:
  in context + abstained   -> reader saw it and still gave up  (reader problem)
  in context + wrong       -> reader saw it and misread it       (reader problem)
  NOT in context + abstained -> evidence never arrived           (retrieval problem)
  NOT in context + wrong   -> guessed without evidence          (retrieval problem)

Only the last two are a retrieval target.  No LLM calls -- the context is
rebuilt from the same pipeline the harness uses.
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
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"
RUN = ROOT / "benchmark_results" / "locomo_full_20261012_211601.json"

STOP = frozenset({
    "i", "im", "ive", "id", "ill", "my", "me", "mine", "we", "our", "us",
    "you", "your", "the", "a", "an", "and", "or", "but", "to", "of", "in",
    "on", "at", "for", "with", "was", "were", "is", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "so", "very", "just",
    "went", "go", "going", "had", "has", "have", "did", "do", "does",
    "when", "what", "where", "which", "who", "how", "why", "would", "could",
    "should", "likely", "still", "want", "wants", "she", "her", "he", "his",
    "they", "them", "their", "melanie", "caroline", "jon", "gina",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "yesterday",
    "today", "tomorrow", "week", "weeks", "weekend", "month", "months",
    "year", "years", "ago", "before", "after", "last", "next",
})


def content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if w not in STOP and len(w) > 2 and not w.isdigit()}


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    recorded = {r["question_id"]: r
                for r in json.loads(RUN.read_text(encoding="utf-8"))["results"]}
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    buckets = Counter()
    examples = {}

    for conv in data:
        turns, _ = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            r = recorded.get(q["question_id"])
            if r is None or r["category"] != 4 or r["is_correct"]:
                continue

            gold_words = content(q["answer"])
            if not gold_words:
                continue

            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{k}" for k, t in enumerate(top)}
            anchors = [t_map[(t[0], t[3])] for t in top[:12]
                       if (t[0], t[3]) in t_map]
            edg = gb.build(list(top), [])
            bundles = edg.form_bundles(anchors, max_bundle_tokens=150)
            budget = E._scaled_budget(
                plan.recommended_budget,
                sum(len(t[3].split()) * 4 // 3 for t in top))
            try:
                out = comp.compile(
                    question=q["question"], bundles=bundles, token_budget=budget,
                    anchor_scores={a: 5.0 for a in anchors},
                    is_multi_session=False, adaptive_escalation=True)
                ctx_words = set(re.findall(r"[a-z0-9]+", out.context_text.lower()))
            except Exception:
                continue

            # how much of the gold is recoverable from the context
            cov = len(gold_words & ctx_words) / len(gold_words)
            in_ctx = cov >= 0.5
            abstained = str(r["predicted_answer"]).strip().lower() in (
                "i don't know.", "", "not mentioned")

            if in_ctx and abstained:
                key = "A_in_ctx_abstained"
            elif in_ctx:
                key = "B_in_ctx_wrong"
            elif abstained:
                key = "C_not_in_ctx_abstained"
            else:
                key = "D_not_in_ctx_wrong"
            buckets[key] += 1
            if key not in examples:
                examples[key] = (q["question"], q["answer"],
                                 str(r["predicted_answer"]), round(cov, 2))

    print("=" * 76)
    print("cat4 FAILURES: is it retrieval or the reader?")
    print("=" * 76)
    total = sum(buckets.values())
    labels = {
        "A_in_ctx_abstained": "saw the evidence, still abstained   (READER)",
        "B_in_ctx_wrong": "saw the evidence, misread it        (READER)",
        "C_not_in_ctx_abstained": "evidence never arrived, abstained (RETRIEVAL)",
        "D_not_in_ctx_wrong": "evidence never arrived, guessed    (RETRIEVAL)",
    }
    print(f"cat4 failures analysed: {total}")
    print()
    for k in ("A_in_ctx_abstained", "B_in_ctx_wrong",
              "C_not_in_ctx_abstained", "D_not_in_ctx_wrong"):
        c = buckets[k]
        print(f"  {labels[k]:44s} {c:4d}  ({c/max(1,total)*100:5.1f}%)")
        if k in examples:
            qq, gt, pred, cov = examples[k]
            print(f"      e.g. {qq[:56]}")
            print(f"           GT {gt[:44]}  | PRED {pred[:30]}  | gold cov {cov}")
    reader = buckets["A_in_ctx_abstained"] + buckets["B_in_ctx_wrong"]
    retr = buckets["C_not_in_ctx_abstained"] + buckets["D_not_in_ctx_wrong"]
    print()
    print(f"  READER-side   : {reader} ({reader/max(1,total)*100:.1f}%)")
    print(f"  RETRIEVAL-side: {retr} ({retr/max(1,total)*100:.1f}%)")
    print()
    if reader > retr:
        print("VERDICT: most cat4 failures saw the evidence and still failed.")
        print("  Retrieval work will not move cat4 much; the reader will.")
    else:
        print("VERDICT: most cat4 failures never received the evidence.")
        print("  Retrieval is the lever for cat4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
